//! M2: dataset reader for the Lance v2.1 subset.
//!
//! Opens datasets written by `super::writer` (and by the official crate for
//! the encodings we support: MiniBlock pages with Flat, InlineBitpacking and
//! FixedSizeList value compression over all-valid repetition/definition
//! layers). Anything outside that subset is rejected with
//! [`StorageError::UnsupportedFormat`], never guessed (see #75).

use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float32Array, Float64Array, UInt32Array};
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
    F32(Vec<f32>),
    U8(Vec<u8>),
    U32(Vec<u32>),
    U64(Vec<u64>),
    I64(Vec<i64>),
}

/// The scalar leaf type a column decodes into (FSL columns decode into their
/// item type).
#[derive(Clone, Copy, PartialEq, Debug)]
enum Leaf {
    F64,
    F32,
    U8,
    U32,
    U64,
    I64,
}

fn leaf_of(dt: &DataType) -> StorageResult<Leaf> {
    match dt {
        DataType::Float64 => Ok(Leaf::F64),
        DataType::Float32 => Ok(Leaf::F32),
        DataType::UInt8 => Ok(Leaf::U8),
        DataType::UInt32 => Ok(Leaf::U32),
        DataType::UInt64 => Ok(Leaf::U64),
        DataType::Int64 => Ok(Leaf::I64),
        DataType::FixedSizeList(child, _) => match child.data_type() {
            DataType::Float64 => Ok(Leaf::F64),
            DataType::Float32 => Ok(Leaf::F32),
            other => Err(unsupported(&format!("fsl item type {other:?}"))),
        },
        other => Err(unsupported(&format!("column type {other:?}"))),
    }
}

/// The FL block holds 1024 values per chunk; a zero-width (constant)
/// bitpacked chunk carries no packed words at all and decodes to zeros
/// (review PR #96, finding 5).
fn zeros_as<T>(values_in_chunk: u64) -> StorageResult<Vec<T>>
where
    T: Default + Clone,
{
    let n = values_in_chunk as usize;
    if n > 1024 {
        return Err(StorageError::Invalid(
            "bitpacked chunk claims more than 1024 values".into(),
        ));
    }
    Ok(vec![T::default(); n])
}

fn decode_chunk(
    buffer: &[u8],
    compression: &Compression,
    leaf: Leaf,
    values_in_chunk: u64,
) -> StorageResult<ChunkValues> {
    match compression {
        Compression::Flat(flat) => match (leaf, flat.bits_per_value) {
            (Leaf::F64, 64) => {
                let values = buffer
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| f64::from_le_bytes(*b))
                    .collect();
                Ok(ChunkValues::F64(values))
            }
            (Leaf::F32, 32) => {
                let values = buffer
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| f32::from_le_bytes(*b))
                    .collect();
                Ok(ChunkValues::F32(values))
            }
            (Leaf::I64, 64) => {
                let values = buffer
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| i64::from_le_bytes(*b))
                    .collect();
                Ok(ChunkValues::I64(values))
            }
            (Leaf::U64, 64) => {
                let values = buffer
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| u64::from_le_bytes(*b))
                    .collect();
                Ok(ChunkValues::U64(values))
            }
            (Leaf::U32, 32) => {
                let values = buffer
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|b| u32::from_le_bytes(*b))
                    .collect();
                Ok(ChunkValues::U32(values))
            }
            (Leaf::U8, 8) => Ok(ChunkValues::U8(buffer.to_vec())),
            (leaf, bits) => Err(unsupported(&format!(
                "Flat bits_per_value={bits} for leaf {leaf:?}"
            ))),
        },
        Compression::InlineBitpacking(ib) => match (leaf, ib.uncompressed_bits_per_value) {
            (Leaf::U32, 32) => {
                if buffer.len() < 4 {
                    return Err(StorageError::Invalid("bitpacked chunk too small".into()));
                }
                let width = u32::from_le_bytes(buffer[0..4].try_into().unwrap()) as usize;
                if width == 0 {
                    // constant / all-zero chunk: the packed block is empty
                    return Ok(ChunkValues::U32(zeros_as::<u32>(values_in_chunk)?));
                }
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
                // `words` holds one full block of packed words. A zero width
                // (constant chunk) has an empty block and is handled above
                // the unpack via the `block_words == 0` short-circuit.
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
            (Leaf::I64, 64) => {
                if buffer.len() < 4 {
                    return Err(StorageError::Invalid("bitpacked chunk too small".into()));
                }
                let width = u32::from_le_bytes(buffer[0..4].try_into().unwrap()) as usize;
                if width == 0 {
                    // constant / all-zero chunk: the packed block is empty
                    return Ok(ChunkValues::I64(zeros_as::<i64>(values_in_chunk)?));
                }
                let packed = &buffer[4..];
                if !packed.len().is_multiple_of(8) {
                    return Err(StorageError::Invalid(
                        "bitpacked chunk not word-aligned".into(),
                    ));
                }
                let words: Vec<u64> = packed
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| u64::from_le_bytes(*b))
                    .collect();
                let block_words = (1024usize * width).div_ceil(64);
                if words.len() < block_words {
                    return Err(StorageError::Invalid(
                        "bitpacked chunk shorter than one FL block".into(),
                    ));
                }
                let mut out = vec![0u64; 1024];
                // SAFETY: `out` has exactly the FL block size for u64 and
                // `words` holds one full block of packed words.
                unsafe {
                    use lance_bitpacking::BitPacking;
                    <u64 as BitPacking>::unchecked_unpack(width, &words[..block_words], &mut out);
                }
                let n = values_in_chunk as usize;
                if n > 1024 {
                    return Err(StorageError::Invalid(
                        "bitpacked chunk claims more than 1024 values".into(),
                    ));
                }
                Ok(ChunkValues::I64(
                    out[..n].iter().map(|v| *v as i64).collect(),
                ))
            }
            (Leaf::U64, 64) => {
                if buffer.len() < 4 {
                    return Err(StorageError::Invalid("bitpacked chunk too small".into()));
                }
                let width = u32::from_le_bytes(buffer[0..4].try_into().unwrap()) as usize;
                if width == 0 {
                    // constant / all-zero chunk: the packed block is empty
                    return Ok(ChunkValues::U64(zeros_as::<u64>(values_in_chunk)?));
                }
                let packed = &buffer[4..];
                if !packed.len().is_multiple_of(8) {
                    return Err(StorageError::Invalid(
                        "bitpacked chunk not word-aligned".into(),
                    ));
                }
                let words: Vec<u64> = packed
                    .as_chunks::<8>()
                    .0
                    .iter()
                    .map(|b| u64::from_le_bytes(*b))
                    .collect();
                let block_words = (1024usize * width).div_ceil(64);
                if words.len() < block_words {
                    return Err(StorageError::Invalid(
                        "bitpacked chunk shorter than one FL block".into(),
                    ));
                }
                let mut out = vec![0u64; 1024];
                // SAFETY: `out` has exactly the FL block size for u64 and
                // `words` holds one full block of packed words.
                unsafe {
                    use lance_bitpacking::BitPacking;
                    <u64 as BitPacking>::unchecked_unpack(width, &words[..block_words], &mut out);
                }
                let n = values_in_chunk as usize;
                if n > 1024 {
                    return Err(StorageError::Invalid(
                        "bitpacked chunk claims more than 1024 values".into(),
                    ));
                }
                Ok(ChunkValues::U64(out[..n].to_vec()))
            }
            (Leaf::U8, 8) => {
                if buffer.len() < 4 {
                    return Err(StorageError::Invalid("bitpacked chunk too small".into()));
                }
                let width = u32::from_le_bytes(buffer[0..4].try_into().unwrap()) as usize;
                if width > 8 {
                    return Err(StorageError::Invalid(format!(
                        "u8 bitpacked width {width} exceeds 8 bits"
                    )));
                }
                if width == 0 {
                    // constant / all-zero chunk: the packed block is empty
                    return Ok(ChunkValues::U8(zeros_as::<u8>(values_in_chunk)?));
                }
                let packed = &buffer[4..];
                let block_bytes = (1024usize * width).div_ceil(8);
                if packed.len() < block_bytes {
                    return Err(StorageError::Invalid(
                        "bitpacked chunk shorter than one FL block".into(),
                    ));
                }
                let mut out = vec![0u8; 1024];
                // SAFETY: `out` has exactly the FL block size for u8 and
                // `packed` starts with one full block of packed bytes.
                unsafe {
                    use lance_bitpacking::BitPacking;
                    <u8 as BitPacking>::unchecked_unpack(width, &packed[..block_bytes], &mut out);
                }
                let n = values_in_chunk as usize;
                if n > 1024 {
                    return Err(StorageError::Invalid(
                        "bitpacked chunk claims more than 1024 values".into(),
                    ));
                }
                Ok(ChunkValues::U8(out[..n].to_vec()))
            }
            (leaf, bits) => Err(unsupported(&format!(
                "InlineBitpacking uncompressed_bits_per_value={bits} for leaf {leaf:?}"
            ))),
        },
        other => Err(unsupported(&format!("value compression {other:?}"))),
    }
}

/// Per-leaf decoded value accumulators for one page.
#[derive(Default)]
struct ColumnValues {
    f64: Vec<f64>,
    f32: Vec<f32>,
    u8: Vec<u8>,
    u32: Vec<u32>,
    u64: Vec<u64>,
    i64: Vec<i64>,
}

impl ColumnValues {
    fn append(&mut self, values: ChunkValues) {
        match values {
            ChunkValues::F64(mut v) => self.f64.append(&mut v),
            ChunkValues::F32(mut v) => self.f32.append(&mut v),
            ChunkValues::U8(mut v) => self.u8.append(&mut v),
            ChunkValues::U32(mut v) => self.u32.append(&mut v),
            ChunkValues::U64(mut v) => self.u64.append(&mut v),
            ChunkValues::I64(mut v) => self.i64.append(&mut v),
        }
    }

    /// Rejects chunks decoded into any leaf type other than the expected
    /// ones for the column (guards against encoding/schema confusion).
    fn keep_only(&self, keep: &[&str]) -> StorageResult<()> {
        let checks: [(&str, usize); 6] = [
            ("f64", self.f64.len()),
            ("f32", self.f32.len()),
            ("u8", self.u8.len()),
            ("u32", self.u32.len()),
            ("u64", self.u64.len()),
            ("i64", self.i64.len()),
        ];
        for (name, len) in checks {
            if len > 0 && !keep.contains(&name) {
                return Err(StorageError::Invalid(format!(
                    "{name} values in {} column",
                    keep.join("|")
                )));
            }
        }
        Ok(())
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
    let leaf = leaf_of(expected_type)?;
    let mut out = ColumnValues::default();
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
                let expected_bits = match leaf {
                    Leaf::F64 => 64,
                    Leaf::F32 => 32,
                    other => {
                        return Err(unsupported(&format!("FixedSizeList with {other:?} items")));
                    }
                };
                if flat.bits_per_value != expected_bits {
                    return Err(unsupported(&format!(
                        "FixedSizeList with {}-bit items",
                        flat.bits_per_value
                    )));
                }
                let items_expected = values_in_chunk as usize * fsl.items_per_value as usize;
                match decode_chunk(buffer, &Compression::Flat(*flat), leaf, values_in_chunk)? {
                    ChunkValues::F64(mut items) => {
                        if items.len() != items_expected {
                            return Err(StorageError::DimensionMismatch {
                                expected: format!("{items_expected} items"),
                                found: format!("{} items", items.len()),
                            });
                        }
                        out.f64.append(&mut items);
                    }
                    ChunkValues::F32(mut items) => {
                        if items.len() != items_expected {
                            return Err(StorageError::DimensionMismatch {
                                expected: format!("{items_expected} items"),
                                found: format!("{} items", items.len()),
                            });
                        }
                        out.f32.append(&mut items);
                    }
                    _ => {
                        return Err(StorageError::Invalid(
                            "fsl decoded as non-float values".into(),
                        ));
                    }
                }
            }
            other => {
                let values_in_chunk = if ln == 0 {
                    page.length - (vals_so_far - values_in_chunk)
                } else {
                    values_in_chunk
                };
                let values = decode_chunk(buffer, other, leaf, values_in_chunk)?;
                out.append(values);
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

    build_array(expected_type, out, page.length)
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
    values: ColumnValues,
    expected_rows: u64,
) -> StorageResult<ArrayRef> {
    let arr: ArrayRef = match expected_type {
        DataType::Float64 => {
            values.keep_only(&["f64"])?;
            if values.f64.len() as u64 != expected_rows {
                return Err(StorageError::DimensionMismatch {
                    expected: format!("{expected_rows} values"),
                    found: format!("{} values", values.f64.len()),
                });
            }
            Arc::new(Float64Array::from(values.f64))
        }
        DataType::Float32 => {
            values.keep_only(&["f32"])?;
            if values.f32.len() as u64 != expected_rows {
                return Err(StorageError::DimensionMismatch {
                    expected: format!("{expected_rows} values"),
                    found: format!("{} values", values.f32.len()),
                });
            }
            Arc::new(Float32Array::from(values.f32))
        }
        DataType::UInt8 => {
            values.keep_only(&["u8"])?;
            if values.u8.len() as u64 != expected_rows {
                return Err(StorageError::DimensionMismatch {
                    expected: format!("{expected_rows} values"),
                    found: format!("{} values", values.u8.len()),
                });
            }
            Arc::new(arrow::array::UInt8Array::from(values.u8))
        }
        DataType::UInt32 => {
            values.keep_only(&["u32"])?;
            if values.u32.len() as u64 != expected_rows {
                return Err(StorageError::DimensionMismatch {
                    expected: format!("{expected_rows} values"),
                    found: format!("{} values", values.u32.len()),
                });
            }
            Arc::new(UInt32Array::from(values.u32))
        }
        DataType::UInt64 => {
            values.keep_only(&["u64"])?;
            if values.u64.len() as u64 != expected_rows {
                return Err(StorageError::DimensionMismatch {
                    expected: format!("{expected_rows} values"),
                    found: format!("{} values", values.u64.len()),
                });
            }
            Arc::new(arrow::array::UInt64Array::from(values.u64))
        }
        DataType::Int64 => {
            values.keep_only(&["i64"])?;
            if values.i64.len() as u64 != expected_rows {
                return Err(StorageError::DimensionMismatch {
                    expected: format!("{expected_rows} values"),
                    found: format!("{} values", values.i64.len()),
                });
            }
            Arc::new(arrow::array::Int64Array::from(values.i64))
        }
        DataType::FixedSizeList(child, dim) => {
            match child.data_type() {
                DataType::Float64 => {
                    values.keep_only(&["f64"])?;
                    let items_expected = expected_rows * *dim as u64;
                    if values.f64.len() as u64 != items_expected {
                        return Err(StorageError::DimensionMismatch {
                            expected: format!("{items_expected} items"),
                            found: format!("{} items", values.f64.len()),
                        });
                    }
                    let values = Float64Array::from(values.f64);
                    // Mirror official-reader behavior: the FSL child field is nullable.
                    let child = ArrowField::new("item", DataType::Float64, true);
                    Arc::new(FixedSizeListArray::new(
                        Arc::new(child),
                        *dim,
                        Arc::new(values),
                        None,
                    ))
                }
                DataType::Float32 => {
                    values.keep_only(&["f32"])?;
                    let items_expected = expected_rows * *dim as u64;
                    if values.f32.len() as u64 != items_expected {
                        return Err(StorageError::DimensionMismatch {
                            expected: format!("{items_expected} items"),
                            found: format!("{} items", values.f32.len()),
                        });
                    }
                    let values = Float32Array::from(values.f32);
                    let child = ArrowField::new("item", DataType::Float32, true);
                    Arc::new(FixedSizeListArray::new(
                        Arc::new(child),
                        *dim,
                        Arc::new(values),
                        None,
                    ))
                }
                other => {
                    return Err(unsupported(&format!("fsl item type {other:?}")));
                }
            }
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
