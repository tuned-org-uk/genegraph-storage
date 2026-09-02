//! M2: dataset reader for the Lance v2.1 subset.
//!
//! Opens datasets written by `super::writer` (and by the official crate for
//! the encodings we support: MiniBlock pages with Flat, InlineBitpacking and
//! FixedSizeList value compression over all-valid repetition/definition
//! layers). Anything outside that subset is rejected with
//! [`StorageError::UnsupportedFormat`], never guessed (see #75).

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float64Array, UInt32Array};
use arrow::datatypes::{DataType, Field as ArrowField, Schema as ArrowSchema};
use arrow::record_batch::RecordBatch;
use prost::Message;

use crate::StorageError;
use crate::StorageResult;
use crate::lancefmt::pb::encodings21::compressive_encoding::Compression;
use crate::lancefmt::pb::encodings21::page_layout;
use crate::lancefmt::pb::encodings21::{MiniBlockLayout, PageLayout};
use crate::lancefmt::pb::filev2 as lfv2;
use crate::lancefmt::pb::filev2::{ColumnMetadata, column_metadata};
use crate::lancefmt::pb::table::{DataFile, Manifest};

use super::schema::from_lance_fields;

const MAGIC: &[u8; 4] = b"LANC";
const REPDEF_ALL_VALID_ITEM: i32 = 1;

fn unsupported(what: &str) -> StorageError {
    StorageError::UnsupportedFormat(format!("lancefmt reader: {what}"))
}

/// Reads the newest manifest from a dataset directory.
fn open_manifest(dir: &Path) -> StorageResult<Manifest> {
    let versions = dir.join("_versions");
    let entries = std::fs::read_dir(&versions)
        .map_err(|e| StorageError::Io(format!("read {:?}: {e}", versions)))?;
    let mut best: Option<(u64, std::path::PathBuf)> = None;
    for entry in entries {
        let entry = entry.map_err(|e| StorageError::Io(e.to_string()))?;
        let name = entry.file_name();
        let name = name.to_string_lossy().to_string();
        let Some(stem) = name.strip_suffix(".manifest") else {
            continue;
        };
        let version: u64 = stem
            .parse()
            .map_err(|_| StorageError::Invalid(format!("unparseable manifest name {name}")))?;
        if best.as_ref().is_none_or(|(v, _)| version > *v) {
            best = Some((version, entry.path()));
        }
    }
    let (_, manifest_path) =
        best.ok_or_else(|| StorageError::Invalid(format!("no manifest under {versions:?}")))?;

    let raw = std::fs::read(&manifest_path)
        .map_err(|e| StorageError::Io(format!("read {manifest_path:?}: {e}")))?;
    let n = raw.len();
    if n < 24 || &raw[n - 4..] != MAGIC {
        return Err(StorageError::Invalid(format!(
            "corrupt manifest {manifest_path:?}: bad magic"
        )));
    }
    let txn_len = u32::from_le_bytes(raw[0..4].try_into().unwrap()) as usize;
    let mstart = 4 + txn_len;
    if mstart + 4 > n {
        return Err(StorageError::Invalid("corrupt manifest: truncated".into()));
    }
    let mlen = u32::from_le_bytes(raw[mstart..mstart + 4].try_into().unwrap()) as usize;
    if mstart + 4 + mlen > n {
        return Err(StorageError::Invalid("corrupt manifest: truncated".into()));
    }
    Manifest::decode(&raw[mstart + 4..mstart + 4 + mlen])
        .map_err(|e| StorageError::Invalid(format!("manifest decode failed: {e}")))
}

struct Footer {
    cmo: u64,
    n_columns: u32,
}

fn parse_footer(data: &[u8]) -> StorageResult<Footer> {
    let n = data.len();
    if n < 40 || &data[n - 4..] != MAGIC {
        return Err(StorageError::Invalid(
            "corrupt data file: bad footer".into(),
        ));
    }
    let major = u16::from_le_bytes(data[n - 8..n - 6].try_into().unwrap());
    let minor = u16::from_le_bytes(data[n - 6..n - 4].try_into().unwrap());
    if major != 2 || minor > 1 {
        return Err(unsupported(&format!("file version {major}.{minor}")));
    }
    Ok(Footer {
        cmo: u64::from_le_bytes(data[n - 32..n - 24].try_into().unwrap()),
        n_columns: u32::from_le_bytes(data[n - 12..n - 8].try_into().unwrap()),
    })
}

fn decode_direct_any(encoding: &Option<lfv2::Encoding>) -> StorageResult<Vec<u8>> {
    match encoding.as_ref().and_then(|e| e.location.as_ref()) {
        Some(lfv2::encoding::Location::Direct(d)) => {
            let any = prost_types::Any::decode(d.encoding.as_slice())
                .map_err(|e| StorageError::Invalid(format!("bad encoding Any: {e}")))?;
            Ok(any.value)
        }
        _ => Err(unsupported("indirect or absent encodings")),
    }
}

/// One decoded chunk: `values` holds the logical values of the chunk.
enum ChunkValues {
    F64(Vec<f64>),
    U32(Vec<u32>),
}

fn decode_chunk(
    buffer: &[u8],
    compression: &Compression,
    values_in_chunk: u64,
) -> StorageResult<ChunkValues> {
    match compression {
        Compression::Flat(flat) => match flat.bits_per_value {
            64 => {
                let values = buffer
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| f64::from_le_bytes(*b))
                    .collect();
                Ok(ChunkValues::F64(values))
            }
            32 => {
                let values = buffer
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                Ok(ChunkValues::U32(values))
            }
            other => Err(unsupported(&format!("Flat bits_per_value={other}"))),
        },
        Compression::InlineBitpacking(ib) => match ib.uncompressed_bits_per_value {
            32 => {
                if buffer.len() < 4 {
                    return Err(StorageError::Invalid("bitpacked chunk too small".into()));
                }
                let width = u32::from_le_bytes(buffer[0..4].try_into().unwrap()) as usize;
                let packed = &buffer[4..];
                if !packed.len().is_multiple_of(4) {
                    return Err(StorageError::Invalid(
                        "bitpacked chunk not word-aligned".into(),
                    ));
                }
                let words: Vec<u32> = packed
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                let block_words = (1024usize * width).div_ceil(32);
                if words.len() < block_words {
                    return Err(StorageError::Invalid(
                        "bitpacked chunk shorter than one FL block".into(),
                    ));
                }
                let mut out = vec![0u32; 1024];
                // SAFETY: `out` has exactly the FL block size for u32 and
                // `words` holds one full block of packed words.
                unsafe {
                    use lance_bitpacking::BitPacking;
                    <u32 as BitPacking>::unchecked_unpack(width, &words[..block_words], &mut out);
                }
                let n = values_in_chunk as usize;
                if n > 1024 {
                    return Err(StorageError::Invalid(
                        "bitpacked chunk claims more than 1024 values".into(),
                    ));
                }
                Ok(ChunkValues::U32(out[..n].to_vec()))
            }
            other => Err(unsupported(&format!(
                "InlineBitpacking uncompressed_bits_per_value={other}"
            ))),
        },
        other => Err(unsupported(&format!("value compression {other:?}"))),
    }
}

fn decode_page(
    data: &[u8],
    page: &column_metadata::Page,
    expected_type: &DataType,
) -> StorageResult<ArrayRef> {
    if page.buffer_offsets.len() < 2 || page.buffer_sizes.len() < 2 {
        return Err(unsupported("pages without metadata+data buffers"));
    }
    let (m_off, m_len) = (
        page.buffer_offsets[0] as usize,
        page.buffer_sizes[0] as usize,
    );
    let (d_off, d_len) = (
        page.buffer_offsets[1] as usize,
        page.buffer_sizes[1] as usize,
    );
    if m_off + m_len > data.len() || d_off + d_len > data.len() {
        return Err(StorageError::Invalid(
            "page buffers out of file bounds".into(),
        ));
    }
    let metadata = &data[m_off..m_off + m_len];
    let page_data = &data[d_off..d_off + d_len];

    let encoding = decode_direct_any(&page.encoding)?;
    let layout = PageLayout::decode(encoding.as_slice())
        .map_err(|e| StorageError::Invalid(format!("page layout decode: {e}")))?;
    let Some(page_layout::Layout::MiniBlockLayout(miniblock)) = layout.layout else {
        return Err(unsupported("non-MiniBlock page layouts"));
    };
    validate_miniblock(&miniblock)?;

    let Some(compression) = miniblock
        .value_compression
        .as_ref()
        .and_then(|c| c.compression.as_ref())
    else {
        return Err(unsupported("pages without value compression"));
    };

    // Walk chunks: metadata is one u16 per chunk; the data buffer holds the
    // chunks back to back.
    let mut f64_out: Vec<f64> = Vec::new();
    let mut u32_out: Vec<u32> = Vec::new();
    let mut chunk_data_pos = 0usize;
    let mut vals_so_far: u64 = 0;
    let num_chunks = metadata.len() / 2;
    for k in 0..num_chunks {
        let word = u16::from_le_bytes(metadata[k * 2..k * 2 + 2].try_into().unwrap());
        let ln = word & 0xF;
        let chunk_bytes = (((word >> 4) as u64 + 1) * 8) as usize;
        let values_in_chunk = if ln == 0 {
            if num_chunks != k + 1 {
                return Err(StorageError::Invalid(
                    "non-final chunk with log_num_values=0".into(),
                ));
            }
            page.length - vals_so_far
        } else {
            1u64 << ln
        };
        vals_so_far += values_in_chunk;

        // in-chunk header: [u16 num_levels][u16 buffer_size][pad to 8]
        if chunk_data_pos + 4 > page_data.len() {
            return Err(StorageError::Invalid("chunk header out of bounds".into()));
        }
        let num_levels = u16::from_le_bytes(
            page_data[chunk_data_pos..chunk_data_pos + 2]
                .try_into()
                .unwrap(),
        );
        if num_levels != 0 {
            return Err(unsupported("chunks with repetition/definition levels"));
        }
        let buffer_size = u16::from_le_bytes(
            page_data[chunk_data_pos + 2..chunk_data_pos + 4]
                .try_into()
                .unwrap(),
        ) as usize;
        let buffer_start = chunk_data_pos + 8;
        let buffer_end = buffer_start + buffer_size;
        if buffer_end > chunk_data_pos + chunk_bytes || buffer_end > page_data.len() {
            return Err(StorageError::Invalid("chunk buffer out of bounds".into()));
        }
        let buffer = &page_data[buffer_start..buffer_end];

        match compression {
            Compression::FixedSizeList(fsl) => {
                let Some(Compression::Flat(flat)) =
                    fsl.values.as_ref().and_then(|c| c.compression.as_ref())
                else {
                    return Err(unsupported("FixedSizeList without Flat values"));
                };
                if flat.bits_per_value != 64 {
                    return Err(unsupported("FixedSizeList with non-64-bit items"));
                }
                match decode_chunk(buffer, &Compression::Flat(*flat), values_in_chunk)? {
                    ChunkValues::F64(mut items) => {
                        let items_expected =
                            values_in_chunk as usize * fsl.items_per_value as usize;
                        if items.len() != items_expected {
                            return Err(StorageError::DimensionMismatch {
                                expected: format!("{items_expected} items"),
                                found: format!("{} items", items.len()),
                            });
                        }
                        f64_out.append(&mut items);
                    }
                    ChunkValues::U32(_) => {
                        return Err(StorageError::Invalid("fsl decoded as u32".into()));
                    }
                }
            }
            other => {
                let values_in_chunk = if ln == 0 {
                    page.length - (vals_so_far - values_in_chunk)
                } else {
                    values_in_chunk
                };
                match decode_chunk(buffer, other, values_in_chunk)? {
                    ChunkValues::F64(mut v) => f64_out.append(&mut v),
                    ChunkValues::U32(mut v) => u32_out.append(&mut v),
                }
            }
        }
        chunk_data_pos += chunk_bytes;
    }
    if chunk_data_pos != page_data.len() {
        return Err(StorageError::Invalid(format!(
            "page data trailing bytes: {} of {} consumed",
            chunk_data_pos,
            page_data.len()
        )));
    }

    build_array(expected_type, f64_out, u32_out, page.length)
}

fn validate_miniblock(miniblock: &MiniBlockLayout) -> StorageResult<()> {
    if miniblock.has_large_chunk
        || miniblock.repetition_index_depth != 0
        || miniblock.rep_compression.is_some()
        || miniblock.def_compression.is_some()
        || miniblock.num_buffers != 1
    {
        return Err(unsupported(
            "large-chunk/repdef/multi-buffer miniblock pages",
        ));
    }
    if miniblock.layers.as_slice() != [REPDEF_ALL_VALID_ITEM] {
        return Err(unsupported(&format!(
            "repdef layers {:?} (only all-valid supported)",
            miniblock.layers
        )));
    }
    Ok(())
}

fn build_array(
    expected_type: &DataType,
    f64_out: Vec<f64>,
    u32_out: Vec<u32>,
    expected_rows: u64,
) -> StorageResult<ArrayRef> {
    let arr: ArrayRef = match expected_type {
        DataType::Float64 => {
            if !u32_out.is_empty() {
                return Err(StorageError::Invalid("u32 values in f64 column".into()));
            }
            if f64_out.len() as u64 != expected_rows {
                return Err(StorageError::DimensionMismatch {
                    expected: format!("{expected_rows} values"),
                    found: format!("{} values", f64_out.len()),
                });
            }
            Arc::new(Float64Array::from(f64_out))
        }
        DataType::UInt32 => {
            if !f64_out.is_empty() {
                return Err(StorageError::Invalid("f64 values in u32 column".into()));
            }
            if u32_out.len() as u64 != expected_rows {
                return Err(StorageError::DimensionMismatch {
                    expected: format!("{expected_rows} values"),
                    found: format!("{} values", u32_out.len()),
                });
            }
            Arc::new(UInt32Array::from(u32_out))
        }
        DataType::FixedSizeList(_child, dim) => {
            if !u32_out.is_empty() {
                return Err(StorageError::Invalid("u32 values in fsl column".into()));
            }
            let items_expected = expected_rows * *dim as u64;
            if f64_out.len() as u64 != items_expected {
                return Err(StorageError::DimensionMismatch {
                    expected: format!("{items_expected} items"),
                    found: format!("{} items", f64_out.len()),
                });
            }
            let values = Float64Array::from(f64_out);
            // Mirror official-reader behavior: the FSL child field is nullable.
            let child = ArrowField::new("item", DataType::Float64, true);
            Arc::new(FixedSizeListArray::new(
                Arc::new(child),
                *dim,
                Arc::new(values),
                None,
            ))
        }
        other => {
            return Err(unsupported(&format!("column type {other:?}")));
        }
    };
    Ok(arr)
}

fn decode_column(
    data: &[u8],
    cm: &ColumnMetadata,
    expected_type: &DataType,
    expected_rows: u64,
) -> StorageResult<ArrayRef> {
    let column_encoding = decode_direct_any(&cm.encoding)?;
    let parsed = crate::lancefmt::pb::encodings::ColumnEncoding::decode(column_encoding.as_slice())
        .map_err(|e| StorageError::Invalid(format!("column encoding decode: {e}")))?;
    match parsed.column_encoding {
        Some(crate::lancefmt::pb::encodings::column_encoding::ColumnEncoding::Values(())) => {}
        _ => return Err(unsupported("column-level encodings (zone indexes, blobs)")),
    }

    let mut pages: Vec<ArrayRef> = Vec::new();
    for page in &cm.pages {
        if page.length == 0 {
            continue;
        }
        pages.push(decode_page(data, page, expected_type)?);
    }
    let total: u64 = pages.iter().map(|a| a.len() as u64).sum();
    if total != expected_rows {
        return Err(StorageError::DimensionMismatch {
            expected: format!("{expected_rows} rows"),
            found: format!("{total} rows"),
        });
    }
    concat_pages(pages)
}

fn concat_pages(pages: Vec<ArrayRef>) -> StorageResult<ArrayRef> {
    let Some(first) = pages.first() else {
        return Err(StorageError::Invalid("column has no pages".into()));
    };
    if pages.len() == 1 {
        return Ok(first.clone());
    }
    let schema = Arc::new(ArrowSchema::new(vec![ArrowField::new(
        "c",
        first.data_type().clone(),
        true,
    )]));
    let batches = pages
        .iter()
        .map(|a| RecordBatch::try_new(schema.clone(), vec![a.clone()]))
        .collect::<Result<Vec<_>, _>>()
        .map_err(|e| StorageError::Lance(format!("page concat: {e}")))?;
    let combined = arrow::compute::concat_batches(&schema, &batches)
        .map_err(|e| StorageError::Lance(format!("page concat: {e}")))?;
    Ok(combined.column(0).clone())
}

/// Full scan of a dataset directory into a single `RecordBatch`.
pub fn scan_all(dir: &Path) -> StorageResult<RecordBatch> {
    let manifest = open_manifest(dir)?;
    let arrow_schema = from_lance_fields(&manifest.fields, &manifest.schema_metadata)?;
    let fragment = manifest
        .fragments
        .first()
        .ok_or_else(|| StorageError::Invalid("dataset has no fragments".into()))?;
    let data_file: &DataFile = fragment
        .files
        .first()
        .ok_or_else(|| StorageError::Invalid("fragment has no data files".into()))?;
    let data_path = dir.join("data").join(&data_file.path);
    let data = std::fs::read(&data_path)
        .map_err(|e| StorageError::Io(format!("read {data_path:?}: {e}")))?;
    let footer = parse_footer(&data)?;

    if footer.n_columns != data_file.column_indices.len() as u32 {
        return Err(StorageError::DimensionMismatch {
            expected: format!("{} columns in manifest", data_file.column_indices.len()),
            found: format!("{} columns in file", footer.n_columns),
        });
    }

    let table_len = footer.n_columns as usize * 16;
    if footer.cmo as usize + table_len > data.len() {
        return Err(StorageError::Invalid("cmo table out of bounds".into()));
    }
    let mut arrays: Vec<ArrayRef> = Vec::with_capacity(footer.n_columns as usize);
    for c in 0..footer.n_columns as usize {
        let base = footer.cmo as usize + c * 16;
        let pos = u64::from_le_bytes(data[base..base + 8].try_into().unwrap());
        let size = u64::from_le_bytes(data[base + 8..base + 16].try_into().unwrap());
        if pos as usize + size as usize > data.len() {
            return Err(StorageError::Invalid(
                "column metadata out of bounds".into(),
            ));
        }
        let cm = ColumnMetadata::decode(&data[pos as usize..pos as usize + size as usize])
            .map_err(|e| StorageError::Invalid(format!("column metadata decode: {e}")))?;
        let Some(field) = arrow_schema.fields().get(c) else {
            return Err(StorageError::Invalid("column index out of schema".into()));
        };
        arrays.push(decode_column(
            &data,
            &cm,
            field.data_type(),
            fragment.physical_rows,
        )?);
    }

    RecordBatch::try_new(Arc::new(arrow_schema), arrays)
        .map_err(|e| StorageError::Lance(format!("batch build: {e}")))
}
