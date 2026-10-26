//! Numeric NPY v1-v3 array I/O.
//!
//! This module handles little-endian C-order Float64 and UInt64 arrays. It
//! does not encode registry metadata or named collections.

use std::path::Path;

use crate::{StorageError, StorageResult};

const MAGIC: &[u8; 6] = b"\x93NUMPY";

/// Writes a little-endian C-order Float64 NPY v1 array.
pub fn write_f64(path: &Path, shape: &[usize], values: &[f64]) -> StorageResult<()> {
    write_array(path, shape, values, "<f8", |v| v.to_le_bytes())
}

/// Reads a Float64 NPY array. Accepts NPY header versions 1, 2, and 3.
pub fn read_f64(path: &Path) -> StorageResult<(Vec<usize>, Vec<f64>)> {
    let (shape, bytes) = read_array(path, "<f8")?;
    let values = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|item| f64::from_le_bytes(*item))
        .collect();
    Ok((shape, values))
}

/// Writes a little-endian C-order UInt64 NPY v1 array.
pub fn write_u64(path: &Path, shape: &[usize], values: &[u64]) -> StorageResult<()> {
    write_array(path, shape, values, "<u8", |v| v.to_le_bytes())
}

/// Reads a UInt64 NPY array. Accepts NPY header versions 1, 2, and 3.
pub fn read_u64(path: &Path) -> StorageResult<(Vec<usize>, Vec<u64>)> {
    let (shape, bytes) = read_array(path, "<u8")?;
    let values = bytes
        .as_chunks::<8>()
        .0
        .iter()
        .map(|item| u64::from_le_bytes(*item))
        .collect();
    Ok((shape, values))
}

fn element_count(shape: &[usize]) -> StorageResult<usize> {
    shape.iter().try_fold(1usize, |n, &dim| {
        n.checked_mul(dim)
            .ok_or_else(|| StorageError::Overflow("NPY shape element count overflow".into()))
    })
}

fn write_array<T>(
    path: &Path,
    shape: &[usize],
    values: &[T],
    descr: &str,
    encode: impl Fn(&T) -> [u8; 8],
) -> StorageResult<()> {
    let count = element_count(shape)?;
    if count != values.len() {
        return Err(StorageError::DimensionMismatch {
            expected: format!("{count} values for shape {shape:?}"),
            found: format!("{} values", values.len()),
        });
    }
    let shape_text = match shape {
        [] => "".to_string(),
        [one] => format!("{one},"),
        many => many
            .iter()
            .map(usize::to_string)
            .collect::<Vec<_>>()
            .join(", "),
    };
    let prefix =
        format!("{{'descr': '{descr}', 'fortran_order': False, 'shape': ({shape_text}), }}");
    let padding = (16 - ((10 + prefix.len() + 1) % 16)) % 16;
    let header = format!("{prefix}{}\n", " ".repeat(padding));
    let header_len = u16::try_from(header.len())
        .map_err(|_| StorageError::Overflow("NPY v1 header exceeds 65535 bytes".into()))?;
    let payload_len = values
        .len()
        .checked_mul(8)
        .ok_or_else(|| StorageError::Overflow("NPY payload byte count overflow".into()))?;
    let capacity = 10usize
        .checked_add(header.len())
        .and_then(|n| n.checked_add(payload_len))
        .ok_or_else(|| StorageError::Overflow("NPY output size overflow".into()))?;
    let mut out = Vec::with_capacity(capacity);
    out.extend_from_slice(MAGIC);
    out.extend_from_slice(&[1, 0]);
    out.extend_from_slice(&header_len.to_le_bytes());
    out.extend_from_slice(header.as_bytes());
    for value in values {
        out.extend_from_slice(&encode(value));
    }
    std::fs::write(path, out).map_err(|e| StorageError::Io(format!("write NPY {path:?}: {e}")))
}

fn read_array(path: &Path, expected_descr: &str) -> StorageResult<(Vec<usize>, Vec<u8>)> {
    let data =
        std::fs::read(path).map_err(|e| StorageError::Io(format!("read NPY {path:?}: {e}")))?;
    if data.len() < 10 || &data[..6] != MAGIC {
        return Err(StorageError::Invalid(
            "invalid NPY magic or truncated header".into(),
        ));
    }
    let version = (data[6], data[7]);
    let (header_start, header_len) = match version {
        (1, 0) => (10usize, u16::from_le_bytes([data[8], data[9]]) as usize),
        (2, 0) | (3, 0) if data.len() >= 12 => (
            12usize,
            u32::from_le_bytes(data[8..12].try_into().unwrap()) as usize,
        ),
        _ => {
            return Err(StorageError::UnsupportedFormat(format!(
                "unsupported NPY version {version:?}"
            )));
        }
    };
    let header_end = header_start
        .checked_add(header_len)
        .filter(|&end| end <= data.len())
        .ok_or_else(|| StorageError::Invalid("truncated NPY header".into()))?;
    let header = std::str::from_utf8(&data[header_start..header_end])
        .map_err(|_| StorageError::Invalid("NPY header is not UTF-8/ASCII".into()))?;
    let descr = header_value(header, "'descr': '")?;
    if descr != expected_descr {
        return Err(StorageError::UnsupportedFormat(format!(
            "NPY dtype {descr:?} is unsupported; expected {expected_descr:?}"
        )));
    }
    let order = header_value(header, "'fortran_order': ")?;
    if order != "False" {
        return Err(StorageError::UnsupportedFormat(
            "Fortran-order NPY arrays are unsupported".into(),
        ));
    }
    let shape_start = header
        .find("'shape': (")
        .ok_or_else(|| StorageError::Invalid("NPY header has no shape".into()))?
        + 10;
    let shape_end = header[shape_start..]
        .find(')')
        .map(|i| shape_start + i)
        .ok_or_else(|| StorageError::Invalid("malformed NPY shape".into()))?;
    let shape_text = header[shape_start..shape_end].trim();
    let shape = if shape_text.is_empty() {
        Vec::new()
    } else {
        shape_text
            .split(',')
            .filter(|part| !part.trim().is_empty())
            .map(|part| {
                part.trim()
                    .parse::<usize>()
                    .map_err(|_| StorageError::Invalid("invalid NPY shape dimension".into()))
            })
            .collect::<StorageResult<Vec<_>>>()?
    };
    let count = element_count(&shape)?;
    let bytes_len = count
        .checked_mul(8)
        .ok_or_else(|| StorageError::Overflow("NPY data byte count overflow".into()))?;
    if data.len() - header_end != bytes_len {
        return Err(StorageError::Invalid(format!(
            "NPY payload has {} bytes; expected {bytes_len}",
            data.len() - header_end
        )));
    }
    Ok((shape, data[header_end..].to_vec()))
}

fn header_value<'a>(header: &'a str, prefix: &str) -> StorageResult<&'a str> {
    let start = header
        .find(prefix)
        .ok_or_else(|| StorageError::Invalid(format!("NPY header is missing {prefix}")))?
        + prefix.len();
    let end = header[start..]
        .find(['\'', ','])
        .map(|i| start + i)
        .ok_or_else(|| StorageError::Invalid(format!("malformed NPY header value {prefix}")))?;
    Ok(header[start..end].trim())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn temp_path(label: &str) -> std::path::PathBuf {
        std::env::temp_dir().join(format!("npy-{label}-{}.npy", uuid::Uuid::new_v4()))
    }

    fn convert_to_header_version(path: &Path, version: u8) {
        let original = std::fs::read(path).unwrap();
        let old_len = u16::from_le_bytes([original[8], original[9]]) as usize;
        let old_header = std::str::from_utf8(&original[10..10 + old_len]).unwrap();
        let body = old_header.trim_end();
        let prefix = body.trim_end_matches('\n').trim_end();
        let padding = (16 - ((12 + prefix.len() + 1) % 16)) % 16;
        let header = format!("{prefix}{}\n", " ".repeat(padding));
        let mut bytes = Vec::new();
        bytes.extend_from_slice(MAGIC);
        bytes.extend_from_slice(&[version, 0]);
        bytes.extend_from_slice(&(header.len() as u32).to_le_bytes());
        bytes.extend_from_slice(header.as_bytes());
        bytes.extend_from_slice(&original[10 + old_len..]);
        std::fs::write(path, bytes).unwrap();
    }

    #[test]
    fn reads_npy_v2_and_v3_headers() {
        for version in [2, 3] {
            let path = temp_path("version");
            write_f64(&path, &[2], &[2.5, -3.0]).unwrap();
            convert_to_header_version(&path, version);
            assert_eq!(read_f64(&path).unwrap(), (vec![2], vec![2.5, -3.0]));
        }
    }

    #[test]
    fn rejects_big_endian_and_fortran_order_dtypes() {
        let path = temp_path("layout");
        write_f64(&path, &[1], &[1.0]).unwrap();
        let mut bytes = std::fs::read(&path).unwrap();
        let header_end = 10 + u16::from_le_bytes([bytes[8], bytes[9]]) as usize;
        let header = std::str::from_utf8(&bytes[10..header_end]).unwrap();
        let dtype = header.find("<f8").unwrap() + 10;
        let fortran = header.find("False").unwrap() + 10;
        bytes[dtype] = b'>';
        std::fs::write(&path, &bytes).unwrap();
        assert!(matches!(
            read_f64(&path),
            Err(StorageError::UnsupportedFormat(_))
        ));

        bytes[dtype] = b'<';
        bytes[fortran..fortran + 5].copy_from_slice(b"True ");
        std::fs::write(&path, bytes).unwrap();
        assert!(matches!(
            read_f64(&path),
            Err(StorageError::UnsupportedFormat(_))
        ));
    }
}
