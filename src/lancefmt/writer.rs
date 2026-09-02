//! M1: dataset writer for the Lance v2.1 subset.
//!
//! Emits datasets interoperable with the official crate: MiniBlock pages with
//! Flat value compression (FixedSizeList nests Flat for dense vectors), a
//! single fragment, manifest v1 with an inline Overwrite transaction.

use std::path::Path;

use arrow::array::{
    Array, FixedSizeListArray, Float32Array, Float64Array, Int64Array, PrimitiveArray, UInt8Array,
    UInt32Array, UInt64Array,
};
use arrow::datatypes::{DataType, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use prost::Message;

use crate::StorageError;
use crate::StorageResult;
use crate::lancefmt::pb::encodings::ColumnEncoding;
use crate::lancefmt::pb::encodings::column_encoding as ce;
use crate::lancefmt::pb::encodings21::compressive_encoding::Compression;
use crate::lancefmt::pb::encodings21::{
    CompressiveEncoding, FixedSizeList as FslCompressive, Flat, MiniBlockLayout, PageLayout,
    page_layout,
};
use crate::lancefmt::pb::filev2 as lfv2;
use crate::lancefmt::pb::filev2::{ColumnMetadata, column_metadata};
use crate::lancefmt::pb::table::Transaction;
use crate::lancefmt::pb::table::manifest::{DataStorageFormat, WriterVersion};
use crate::lancefmt::pb::table::transaction::{Operation, Overwrite};
use crate::lancefmt::pb::table::{DataFile, DataFragment, Manifest};

use super::schema::{SchemaMeta, to_lance_schema};

const FILE_MAJOR: u16 = 2;
const FILE_MINOR: u16 = 1;
const DATA_FORMAT_VERSION: &str = "2.1";
const CHUNK_ALIGNMENT: usize = 8;
const BUFFER_ALIGNMENT: usize = 64;
const FILL_BYTE: u8 = 0xFE;
/// Per-chunk data budget: keeps the u16 chunk metadata (`((bytes/8)-1)<<4 | ln`)
/// and the u16 in-chunk buffer sizes well inside their limits.
const CHUNK_DATA_BUDGET: usize = 16 * 1024;

fn pad_to(buf: &mut Vec<u8>, alignment: usize) {
    while !buf.len().is_multiple_of(alignment) {
        buf.push(FILL_BYTE);
    }
}

fn direct_encoding(type_url: &'static str, value: Vec<u8>) -> lfv2::Encoding {
    let any = prost_types::Any {
        type_url: type_url.to_string(),
        value,
    };
    lfv2::Encoding {
        location: Some(lfv2::encoding::Location::Direct(lfv2::DirectEncoding {
            encoding: any.encode_to_vec(),
        })),
    }
}

struct Chunk {
    /// `((chunk_bytes / 8) - 1) << 4 | log2(values)`; log2(values) is 0 for the
    /// final chunk of a page.
    metadata: u16,
    bytes: Vec<u8>,
}

struct EncodedPage {
    metadata_buffer: Vec<u8>,
    data_buffer: Vec<u8>,
    rows: u64,
    num_items: u64,
    value_compression: Compression,
}

fn log2_usize(v: usize) -> StorageResult<u16> {
    u16::try_from(v.ilog2()).map_err(|_| StorageError::Overflow("log2 chunk".into()))
}

fn build_chunk(buffer: &[u8], ln: u16) -> StorageResult<Chunk> {
    let mut bytes = Vec::with_capacity(buffer.len() + 16);
    // [u16 num_levels (0: no rep/def support in the writer subset)]
    bytes.extend_from_slice(&0u16.to_le_bytes());
    // [u16 buffer_size]
    let size = u16::try_from(buffer.len()).map_err(|_| {
        StorageError::Overflow(format!("chunk buffer size {} exceeds u16", buffer.len()))
    })?;
    bytes.extend_from_slice(&size.to_le_bytes());
    pad_to(&mut bytes, CHUNK_ALIGNMENT);
    bytes.extend_from_slice(buffer);
    pad_to(&mut bytes, CHUNK_ALIGNMENT);
    let chunk_bytes = bytes.len();
    let divided = chunk_bytes / CHUNK_ALIGNMENT;
    let metadata = ((u16::try_from(divided - 1).map_err(|_| {
        StorageError::Overflow(format!("chunk bytes {chunk_bytes} exceed u16 metadata"))
    })?) << 4)
        | ln;
    Ok(Chunk { metadata, bytes })
}

/// Encodes `total_items` flat LE values into chunks of `items_per_chunk`
/// values; `full_chunk_ln` is the log2 value recorded for a full chunk
/// (log2 of values for scalar columns, log2 of rows for FixedSizeList
/// columns).
fn chunked_items(
    total_items: usize,
    items_per_chunk: usize,
    full_chunk_ln: u16,
    mut write_items: impl FnMut(&mut Vec<u8>, usize, usize),
) -> StorageResult<Vec<Chunk>> {
    let mut chunks = Vec::new();
    let mut start = 0usize;
    while start < total_items {
        let n = items_per_chunk.min(total_items - start);
        let mut buffer = Vec::with_capacity(n * 8);
        write_items(&mut buffer, start, n);
        let ln = if n == items_per_chunk {
            full_chunk_ln
        } else {
            0
        };
        chunks.push(build_chunk(&buffer, ln)?);
        start += n;
    }
    Ok(chunks)
}

fn chunk_bytes_le(
    values: &PrimitiveArray<arrow::datatypes::Float64Type>,
    values_per_chunk: usize,
) -> StorageResult<Vec<Chunk>> {
    let ln = log2_usize(values_per_chunk)?;
    chunked_items(values.len(), values_per_chunk, ln, |out, s, n| {
        for i in s..s + n {
            out.extend_from_slice(&values.value(i).to_le_bytes())
        }
    })
}

fn chunk_bytes_le_f32(
    values: &PrimitiveArray<arrow::datatypes::Float32Type>,
    values_per_chunk: usize,
) -> StorageResult<Vec<Chunk>> {
    let ln = log2_usize(values_per_chunk)?;
    chunked_items(values.len(), values_per_chunk, ln, |out, s, n| {
        for i in s..s + n {
            out.extend_from_slice(&values.value(i).to_le_bytes())
        }
    })
}

fn chunk_bytes_le_u32(
    values: &PrimitiveArray<arrow::datatypes::UInt32Type>,
    values_per_chunk: usize,
) -> StorageResult<Vec<Chunk>> {
    let ln = log2_usize(values_per_chunk)?;
    chunked_items(values.len(), values_per_chunk, ln, |out, s, n| {
        for i in s..s + n {
            out.extend_from_slice(&values.value(i).to_le_bytes())
        }
    })
}

fn chunk_bytes_le_u64(
    values: &PrimitiveArray<arrow::datatypes::UInt64Type>,
    values_per_chunk: usize,
) -> StorageResult<Vec<Chunk>> {
    let ln = log2_usize(values_per_chunk)?;
    chunked_items(values.len(), values_per_chunk, ln, |out, s, n| {
        for i in s..s + n {
            out.extend_from_slice(&values.value(i).to_le_bytes())
        }
    })
}

fn chunk_bytes_le_u8(
    values: &PrimitiveArray<arrow::datatypes::UInt8Type>,
    values_per_chunk: usize,
) -> StorageResult<Vec<Chunk>> {
    let ln = log2_usize(values_per_chunk)?;
    chunked_items(values.len(), values_per_chunk, ln, |out, s, n| {
        for i in s..s + n {
            out.extend_from_slice(&values.value(i).to_le_bytes())
        }
    })
}

fn chunk_bytes_le_i64(
    values: &PrimitiveArray<arrow::datatypes::Int64Type>,
    values_per_chunk: usize,
) -> StorageResult<Vec<Chunk>> {
    let ln = log2_usize(values_per_chunk)?;
    chunked_items(values.len(), values_per_chunk, ln, |out, s, n| {
        for i in s..s + n {
            out.extend_from_slice(&values.value(i).to_le_bytes())
        }
    })
}

fn encode_column(batch: &RecordBatch, col: usize) -> StorageResult<Vec<EncodedPage>> {
    let column = batch.column(col);
    let rows = column.len() as u64;
    let (chunks, value_compression, num_items) = match column.data_type() {
        DataType::Float64 => {
            let arr = column
                .as_any()
                .downcast_ref::<Float64Array>()
                .ok_or_else(|| StorageError::Invalid("float64 downcast failed".into()))?;
            // 512 x 8B = 4KiB per chunk
            (
                chunk_bytes_le(arr, 512)?,
                Compression::Flat(Flat {
                    bits_per_value: 64,
                    data: None,
                }),
                rows,
            )
        }
        DataType::Float32 => {
            let arr = column
                .as_any()
                .downcast_ref::<Float32Array>()
                .ok_or_else(|| StorageError::Invalid("float32 downcast failed".into()))?;
            // 1024 x 4B = 4KiB per chunk
            (
                chunk_bytes_le_f32(arr, 1024)?,
                Compression::Flat(Flat {
                    bits_per_value: 32,
                    data: None,
                }),
                rows,
            )
        }
        DataType::UInt8 => {
            let arr = column
                .as_any()
                .downcast_ref::<UInt8Array>()
                .ok_or_else(|| StorageError::Invalid("uint8 downcast failed".into()))?;
            // 16384 x 1B = 16KiB per chunk
            (
                chunk_bytes_le_u8(arr, 16384)?,
                Compression::Flat(Flat {
                    bits_per_value: 8,
                    data: None,
                }),
                rows,
            )
        }
        DataType::UInt32 => {
            let arr = column
                .as_any()
                .downcast_ref::<UInt32Array>()
                .ok_or_else(|| StorageError::Invalid("uint32 downcast failed".into()))?;
            // 1024 x 4B = 4KiB per chunk
            (
                chunk_bytes_le_u32(arr, 1024)?,
                Compression::Flat(Flat {
                    bits_per_value: 32,
                    data: None,
                }),
                rows,
            )
        }
        DataType::UInt64 => {
            let arr = column
                .as_any()
                .downcast_ref::<UInt64Array>()
                .ok_or_else(|| StorageError::Invalid("uint64 downcast failed".into()))?;
            // 1024 x 8B = 8KiB per chunk
            (
                chunk_bytes_le_u64(arr, 1024)?,
                Compression::Flat(Flat {
                    bits_per_value: 64,
                    data: None,
                }),
                rows,
            )
        }
        DataType::Int64 => {
            let arr = column
                .as_any()
                .downcast_ref::<Int64Array>()
                .ok_or_else(|| StorageError::Invalid("int64 downcast failed".into()))?;
            // 1024 x 8B = 8KiB per chunk
            (
                chunk_bytes_le_i64(arr, 1024)?,
                Compression::Flat(Flat {
                    bits_per_value: 64,
                    data: None,
                }),
                rows,
            )
        }
        DataType::FixedSizeList(child, dim) => {
            let dim: i32 = *dim;
            let list = column
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .ok_or_else(|| StorageError::Invalid("fsl downcast failed".into()))?;
            if list.null_count() != 0 {
                return Err(StorageError::UnsupportedFormat(
                    "lancefmt writer does not support nulls in FixedSizeList columns".into(),
                ));
            }
            let (bits, bytes_per_item) = match child.data_type() {
                DataType::Float64 => (64u64, 8usize),
                DataType::Float32 => (32u64, 4usize),
                other => {
                    return Err(StorageError::UnsupportedFormat(format!(
                        "lancefmt writer: unsupported fixed_size_list item type {other:?}"
                    )));
                }
            };
            let dim = dim as usize;
            let bytes_per_row = dim * bytes_per_item;
            // Rows per full chunk: a power of two so the chunk metadata can
            // record log2(rows). The u16 chunk metadata word caps a chunk at
            // 32768 bytes ((bytes/8 - 1) << 4 must fit in 12 bits), so a
            // single row of `bytes_per_row` bytes must fit in that budget.
            // When the 16KiB budget cannot hold even two rows (wide
            // vectors), `rows_per_chunk` collapses to 1 and log2(1) = 0 is
            // indistinguishable from the final-chunk marker; in that case
            // the column is written as one page per row, each page a single
            // final chunk (the shape the official writer emits for
            // single-chunk pages).
            if bytes_per_row > 32768 - 8 {
                return Err(StorageError::UnsupportedFormat(format!(
                    "lancefmt writer: fixed_size_list row of {bytes_per_row} bytes \
                     exceeds the 32768-byte chunk metadata limit"
                )));
            }
            let rows_per_chunk = (CHUNK_DATA_BUDGET / bytes_per_row).next_power_of_two();
            let fsl_compression = || {
                Compression::FixedSizeList(Box::new(FslCompressive {
                    items_per_value: dim as u64,
                    has_validity: false,
                    values: Some(Box::new(CompressiveEncoding {
                        compression: Some(Compression::Flat(Flat {
                            bits_per_value: bits,
                            data: None,
                        })),
                    })),
                }))
            };
            if rows_per_chunk <= 1 {
                let items = list.values();
                let mut pages = Vec::with_capacity(list.len());
                for row in 0..list.len() {
                    let mut buffer = Vec::with_capacity(bytes_per_row);
                    let start = row * dim;
                    match child.data_type() {
                        DataType::Float64 => {
                            let values =
                                items
                                    .as_any()
                                    .downcast_ref::<Float64Array>()
                                    .ok_or_else(|| {
                                        StorageError::Invalid("fsl child downcast failed".into())
                                    })?;
                            for i in start..start + dim {
                                buffer.extend_from_slice(&values.value(i).to_le_bytes());
                            }
                        }
                        DataType::Float32 => {
                            let values =
                                items
                                    .as_any()
                                    .downcast_ref::<Float32Array>()
                                    .ok_or_else(|| {
                                        StorageError::Invalid("fsl child downcast failed".into())
                                    })?;
                            for i in start..start + dim {
                                buffer.extend_from_slice(&values.value(i).to_le_bytes());
                            }
                        }
                        other => {
                            return Err(StorageError::UnsupportedFormat(format!(
                                "lancefmt writer: unsupported fixed_size_list item type {other:?}"
                            )));
                        }
                    }
                    let chunk = build_chunk(&buffer, 0)?;
                    pages.push(EncodedPage {
                        metadata_buffer: chunk.metadata.to_le_bytes().to_vec(),
                        data_buffer: chunk.bytes,
                        rows: 1,
                        num_items: 1,
                        value_compression: fsl_compression(),
                    });
                }
                return Ok(pages);
            }
            let items_per_chunk = rows_per_chunk * dim;
            let full_chunk_ln = log2_usize(rows_per_chunk)?;
            let total_items = list.values().len();
            let chunks = match child.data_type() {
                DataType::Float64 => {
                    let values = list
                        .values()
                        .as_any()
                        .downcast_ref::<Float64Array>()
                        .ok_or_else(|| StorageError::Invalid("fsl child downcast failed".into()))?;
                    chunked_items(total_items, items_per_chunk, full_chunk_ln, |buf, s, n| {
                        for i in s..s + n {
                            buf.extend_from_slice(&values.value(i).to_le_bytes());
                        }
                    })?
                }
                DataType::Float32 => {
                    let values = list
                        .values()
                        .as_any()
                        .downcast_ref::<Float32Array>()
                        .ok_or_else(|| StorageError::Invalid("fsl child downcast failed".into()))?;
                    chunked_items(total_items, items_per_chunk, full_chunk_ln, |buf, s, n| {
                        for i in s..s + n {
                            buf.extend_from_slice(&values.value(i).to_le_bytes());
                        }
                    })?
                }
                other => {
                    return Err(StorageError::UnsupportedFormat(format!(
                        "lancefmt writer: unsupported fixed_size_list item type {other:?}"
                    )));
                }
            };
            let compression = fsl_compression();
            (chunks, compression, rows)
        }
        other => {
            return Err(StorageError::UnsupportedFormat(format!(
                "lancefmt writer: unsupported column type {other:?}"
            )));
        }
    };

    let mut metadata_buffer = Vec::with_capacity(chunks.len() * 2);
    let mut data_buffer = Vec::new();
    for chunk in &chunks {
        metadata_buffer.extend_from_slice(&chunk.metadata.to_le_bytes());
        data_buffer.extend_from_slice(&chunk.bytes);
    }

    Ok(vec![EncodedPage {
        metadata_buffer,
        data_buffer,
        rows,
        num_items,
        value_compression,
    }])
}

fn page_layout(page: &EncodedPage) -> lfv2::Encoding {
    let layout = PageLayout {
        layout: Some(page_layout::Layout::MiniBlockLayout(MiniBlockLayout {
            rep_compression: None,
            def_compression: None,
            value_compression: Some(CompressiveEncoding {
                compression: Some(page.value_compression.clone()),
            }),
            dictionary: None,
            num_dictionary_items: 0,
            layers: vec![1], // REPDEF_ALL_VALID_ITEM
            num_buffers: 1,
            repetition_index_depth: 0,
            num_items: page.num_items,
            has_large_chunk: false,
        })),
    };
    direct_encoding("/lance.encodings21.PageLayout", layout.encode_to_vec())
}

/// Durable file write (#95-3): write to a unique tmp file next to `path`,
/// fsync, then rename over `path`. The rename is the atomic publish point;
/// the fsync makes the payload durable before the directory entry appears.
fn write_durable(path: &Path, bytes: &[u8]) -> StorageResult<()> {
    use std::io::Write;

    let mut name = path
        .file_name()
        .ok_or_else(|| StorageError::Invalid(format!("bad artifact path {path:?}")))?
        .to_os_string();
    name.push(format!(".{}.tmp", uuid::Uuid::new_v4().simple()));
    let tmp = path.with_file_name(name);

    {
        let mut f = std::fs::File::create(&tmp)
            .map_err(|e| StorageError::Io(format!("create tmp for {path:?}: {e}")))?;
        f.write_all(bytes)
            .map_err(|e| StorageError::Io(format!("write tmp for {path:?}: {e}")))?;
        f.sync_all()
            .map_err(|e| StorageError::Io(format!("fsync tmp for {path:?}: {e}")))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| StorageError::Io(format!("publish {path:?}: {e}")))?;
    Ok(())
}

/// fsyncs a directory so a completed rename is itself durable
/// (directory-entry durability, #95-3). No-op off unix.
fn fsync_dir(dir: &Path) -> StorageResult<()> {
    #[cfg(unix)]
    {
        let f = std::fs::File::open(dir)
            .map_err(|e| StorageError::Io(format!("open dir {dir:?} for fsync: {e}")))?;
        f.sync_all()
            .map_err(|e| StorageError::Io(format!("fsync dir {dir:?}: {e}")))?;
    }
    #[cfg(not(unix))]
    let _ = dir;
    Ok(())
}

/// Writes `batch` as a single-fragment Lance v2.1 dataset rooted at `dir`.
///
/// Durability & concurrency (#95-3): the whole write — manifest-version
/// allocation through commit-point publish — runs under a per-dataset write
/// mailbox (a blocking `std::sync::Mutex` held across several fsyncs), and
/// every artifact (data file, txn, manifest, version hint) is written
/// tmp + fsync + rename with a directory fsync, in the order
/// data → txn → manifest (commit) → hint. Readers observe either the
/// previous or the complete new dataset version.
///
/// Blocking: because of that lock, callers on Tokio must reach this through
/// `spawn_blocking` (as `LanceStorage::write_lance_batch_async` does), never
/// directly on an async executor thread.
pub fn write_dataset(batch: &RecordBatch, dir: &Path) -> StorageResult<()> {
    if batch.num_rows() == 0 {
        return Err(StorageError::Invalid(
            "lancefmt writer: empty batches are not supported".into(),
        ));
    }

    std::fs::create_dir_all(dir)
        .map_err(|e| StorageError::Io(format!("create dataset dir {dir:?}: {e}")))?;
    let data_dir = dir.join("data");
    let versions_dir = dir.join("_versions");
    let txn_dir = dir.join("_transactions");
    for d in [&data_dir, &versions_dir, &txn_dir] {
        std::fs::create_dir_all(d).map_err(|e| StorageError::Io(format!("create {d:?}: {e}")))?;
    }

    crate::commit::with_dataset_write_lock(dir, || {
        write_dataset_locked(batch, &data_dir, &versions_dir, &txn_dir)
    })
}

fn write_dataset_locked(
    batch: &RecordBatch,
    data_dir: &Path,
    versions_dir: &Path,
    txn_dir: &Path,
) -> StorageResult<()> {
    let arrow_schema = batch.schema().as_ref().clone();
    let (fields, schema_meta) = to_lance_schema(&arrow_schema)?;

    let uuid = uuid::Uuid::new_v4();
    let data_file_name = format!("{:024b}{}.lance", 0, uuid.simple());

    // ---- data file -------------------------------------------------------
    let mut file_bytes: Vec<u8> = Vec::new();
    let mut column_metadatas: Vec<(u64, Vec<u8>)> = Vec::new();

    for col in 0..batch.num_columns() {
        let pages = encode_column(batch, col)?;
        let mut page_entries: Vec<column_metadata::Page> = Vec::with_capacity(pages.len());
        for page in &pages {
            let page_encoding = page_layout(page);

            let mut buffers: Vec<&[u8]> = vec![&page.metadata_buffer, &page.data_buffer];
            let mut page_buffer_offsets = Vec::with_capacity(buffers.len());
            let mut page_buffer_sizes = Vec::with_capacity(buffers.len());
            for b in buffers.drain(..) {
                pad_to(&mut file_bytes, BUFFER_ALIGNMENT);
                page_buffer_offsets.push(file_bytes.len() as u64);
                page_buffer_sizes.push(b.len() as u64);
                file_bytes.extend_from_slice(b);
            }

            page_entries.push(column_metadata::Page {
                buffer_offsets: page_buffer_offsets,
                buffer_sizes: page_buffer_sizes,
                length: page.rows,
                encoding: Some(page_encoding),
                priority: 0,
            });
        }

        let column_metadata = ColumnMetadata {
            encoding: Some(direct_encoding(
                "/lance.encodings.ColumnEncoding",
                ColumnEncoding {
                    column_encoding: Some(ce::ColumnEncoding::Values(())),
                }
                .encode_to_vec(),
            )),
            pages: page_entries,
            buffer_offsets: vec![],
            buffer_sizes: vec![],
        };
        column_metadatas.push((0, column_metadata.encode_to_vec()));
    }

    // File layout mirrors the official writer: page buffers, global buffer,
    // column metadatas, column-metadata offset table, global-buffer offset
    // table, footer.
    let schema_bytes =
        super::schema::encode_schema_global(&fields, &schema_meta, batch.num_rows() as u64);

    pad_to(&mut file_bytes, BUFFER_ALIGNMENT);
    let global_pos = file_bytes.len() as u64;
    file_bytes.extend_from_slice(&schema_bytes);

    // column metadata section (immediately after the global buffers, like the
    // official writer)
    let col_meta0_pos = file_bytes.len() as u64;
    let mut cmo_entries = Vec::with_capacity(column_metadatas.len());
    for (_, meta) in &column_metadatas {
        let pos = file_bytes.len() as u64;
        file_bytes.extend_from_slice(meta);
        cmo_entries.push((pos, meta.len() as u64));
    }

    // column metadata offset table
    pad_to(&mut file_bytes, CHUNK_ALIGNMENT);
    let cmo_pos = file_bytes.len() as u64;
    for (pos, size) in &cmo_entries {
        file_bytes.extend_from_slice(&pos.to_le_bytes());
        file_bytes.extend_from_slice(&size.to_le_bytes());
    }

    // global-buffer offset table (single entry: the file schema), written
    // after the cmo table and patched in place
    let gbo_pos = file_bytes.len() as u64;
    file_bytes.extend_from_slice(&global_pos.to_le_bytes());
    file_bytes.extend_from_slice(&(schema_bytes.len() as u64).to_le_bytes());
    let global_entries_len = 1u32;

    // footer: [u64 col_meta0][u64 cmo][u64 gbo][u32 n_global][u32 n_cols]
    //          [u16 major][u16 minor]["LANC"]
    pad_to(&mut file_bytes, CHUNK_ALIGNMENT);
    file_bytes.extend_from_slice(&col_meta0_pos.to_le_bytes());
    file_bytes.extend_from_slice(&cmo_pos.to_le_bytes());
    file_bytes.extend_from_slice(&gbo_pos.to_le_bytes());
    file_bytes.extend_from_slice(&global_entries_len.to_le_bytes());
    file_bytes.extend_from_slice(&(column_metadatas.len() as u32).to_le_bytes());
    file_bytes.extend_from_slice(&FILE_MAJOR.to_le_bytes());
    file_bytes.extend_from_slice(&FILE_MINOR.to_le_bytes());
    file_bytes.extend_from_slice(b"LANC");

    let data_path = data_dir.join(&data_file_name);
    write_durable(&data_path, &file_bytes)?;
    fsync_dir(data_dir)?;

    // ---- manifest + transaction -----------------------------------------
    // Overwrite semantics (M4): an existing dataset receives a new manifest
    // version whose fragment set fully replaces the previous one; readers
    // (ours and official) open the highest version.
    let prev_version = latest_manifest_version(versions_dir)?;
    let next_version = prev_version + 1;
    // Fragment ids are unique per dataset; version numbering gives us a
    // deterministic fresh id per overwrite (fresh dataset -> 0).
    // NOTE (#95): this id == version-by-construction holds only while
    // Overwrite is the sole operation. When Append lands (RFC #81), fragment
    // ids must be allocated from `max_fragment_id` instead, since appended
    // fragments coexist with previous versions' fragments.
    let fragment_id = prev_version;

    let data_file = DataFile {
        path: data_file_name.clone(),
        fields: fields.iter().map(|f| f.id).collect(),
        column_indices: (0..fields.len() as i32).collect(),
        file_major_version: FILE_MAJOR as u32,
        file_minor_version: FILE_MINOR as u32,
        file_size_bytes: file_bytes.len() as u64,
        base_id: None,
    };
    let fragment = DataFragment {
        id: fragment_id,
        files: vec![data_file],
        overlays: vec![],
        deletion_file: None,
        row_id_sequence: None,
        last_updated_at_version_sequence: None,
        created_at_version_sequence: None,
        physical_rows: batch.num_rows() as u64,
    };
    let manifest = Manifest {
        fields,
        schema_metadata: schema_meta.clone(),
        fragments: vec![fragment],
        version: next_version,
        version_aux_data: 0,
        writer_version: Some(WriterVersion {
            library: "genegraph-storage".to_string(),
            version: env!("CARGO_PKG_VERSION").to_string(),
            prerelease: None,
            build_metadata: None,
        }),
        index_section: None,
        timestamp: Some(prost_types::Timestamp {
            seconds: chrono::Utc::now().timestamp(),
            nanos: chrono::Utc::now().timestamp_subsec_nanos() as i32,
        }),
        tag: String::new(),
        reader_feature_flags: 0,
        writer_feature_flags: 0,
        max_fragment_id: Some(fragment_id as u32),
        transaction_file: format!("{prev_version}-{uuid}.txn"),
        next_row_id: 0,
        data_format: Some(DataStorageFormat {
            file_format: "lance".to_string(),
            version: DATA_FORMAT_VERSION.to_string(),
        }),
        config: Default::default(),
        table_metadata: Default::default(),
        base_paths: vec![],
        branch: None,
        transaction_section: None,
    };

    let transaction = Transaction {
        read_version: prev_version,
        uuid: uuid.to_string(),
        tag: String::new(),
        transaction_properties: Default::default(),
        operation: Some(Operation::Overwrite(Overwrite {
            fragments: manifest.fragments.clone(),
            schema: manifest.fields.clone(),
            schema_metadata: schema_meta.clone(),
            config_upsert_values: Default::default(),
            initial_bases: vec![],
        })),
    };
    let txn_bytes = transaction.encode_to_vec();
    write_durable(
        &txn_dir.join(format!("{prev_version}-{uuid}.txn")),
        &txn_bytes,
    )?;
    fsync_dir(txn_dir)?;

    let manifest_bytes = manifest.encode_to_vec();
    let mut out: Vec<u8> = Vec::new();
    out.extend_from_slice(&(txn_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&txn_bytes);
    let manifest_pos = out.len() as u64;
    out.extend_from_slice(&(manifest_bytes.len() as u32).to_le_bytes());
    out.extend_from_slice(&manifest_bytes);
    out.extend_from_slice(&manifest_pos.to_le_bytes());
    out.extend_from_slice(&FILE_MINOR.to_le_bytes());
    out.extend_from_slice(&FILE_MAJOR.to_le_bytes());
    out.extend_from_slice(b"LANC");

    // The manifest rename is the dataset's commit point; the directory
    // fsync right after makes the published version durable.
    write_durable(&versions_dir.join(format!("{next_version}.manifest")), &out)?;
    fsync_dir(versions_dir)?;
    // The hint is a discovery accelerator, not the commit pointer; it is
    // still published atomically so a concurrent open never reads a
    // truncated hint (#95).
    write_durable(
        &versions_dir.join("latest_version_hint.json"),
        format!("{{\"version\":{next_version}}}").as_bytes(),
    )?;

    Ok(())
}

/// Highest `N.manifest` version currently present (0 for a fresh dataset).
fn latest_manifest_version(versions_dir: &Path) -> StorageResult<u64> {
    let entries = std::fs::read_dir(versions_dir)
        .map_err(|e| StorageError::Io(format!("read {versions_dir:?}: {e}")))?;
    let mut best = 0u64;
    for entry in entries {
        let entry = entry.map_err(|e| StorageError::Io(e.to_string()))?;
        let name = entry.file_name();
        let name_str = name.to_string_lossy().to_string();
        let Some(stem) = name_str.strip_suffix(".manifest") else {
            continue;
        };
        let v: u64 = stem.parse().map_err(|_| {
            StorageError::Invalid(format!("unparseable manifest name {name_str:?}"))
        })?;
        best = best.max(v);
    }
    Ok(best)
}

#[allow(dead_code)]
fn unused(_: &ArrowSchema, _: &SchemaMeta) {}
