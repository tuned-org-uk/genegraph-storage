//! Zarr v3 ("zzarr") array IO over the [`zarrs`] crate (#3).
//!
//! Contract: Zarr v3 metadata (`zarr.json`), regular chunk grid, default
//! chunk key encoding (`/` separator), `bytes` (little endian) plus
//! optional `zstd` codecs. Unsupported encodings surface
//! [`StorageError::UnsupportedFormat`] — never guessed.
//!
//! All operations are synchronous; async wrappers belong at the serving
//! boundary (`spawn_blocking`), keeping this module runtime-free.

pub mod csr;

use std::ops::Range;
use std::path::Path;
use std::sync::Arc;

use zarrs::array::codec::ZstdCodec;
use zarrs::array::{Array, data_type};
use zarrs::array::{ArraySubset, DataType, Element, ElementOwned, FillValue};
use zarrs::filesystem::FilesystemStore;
use zarrs::storage::ReadableWritableListableStorageTraits;

use crate::{StorageError, StorageResult};

/// Element types this module can store. Maps a Rust type to its Zarr v3
/// data type and its zero fill value (the zzarr contract uses `0`).
pub trait ZzarrElement: Element + ElementOwned + Clone {
    /// The Zarr v3 data type.
    fn zzarr_data_type() -> DataType;
    /// The zero fill value.
    fn zzarr_fill() -> FillValue;
}

macro_rules! zzarr_element {
    ($t:ty, $dt:expr) => {
        impl ZzarrElement for $t {
            fn zzarr_data_type() -> DataType {
                $dt()
            }
            fn zzarr_fill() -> FillValue {
                FillValue::new((0 as $t).to_le_bytes().to_vec())
            }
        }
    };
}

zzarr_element!(f32, data_type::float32);
zzarr_element!(f64, data_type::float64);
zzarr_element!(i64, data_type::int64);
zzarr_element!(u64, data_type::uint64);

fn store_for(path: &Path) -> StorageResult<Arc<dyn ReadableWritableListableStorageTraits>> {
    let store: Arc<dyn ReadableWritableListableStorageTraits> =
        Arc::new(FilesystemStore::new(path).map_err(|e| StorageError::Io(e.to_string()))?);
    Ok(store)
}

fn map_open_err(err: impl std::fmt::Display) -> StorageError {
    StorageError::UnsupportedFormat(format!("not a readable Zarr v3 array: {}", err))
}

fn map_err(err: impl std::fmt::Display) -> StorageError {
    StorageError::Invalid(format!("zzarr error: {}", err))
}

/// A Zarr v3 array opened on the filesystem.
pub struct ZarrArray {
    array: Array<dyn ReadableWritableListableStorageTraits>,
}

impl std::fmt::Debug for ZarrArray {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("ZarrArray")
            .field("shape", &self.array.shape())
            .finish()
    }
}

impl ZarrArray {
    /// Array shape, outermost dimension first.
    pub fn shape(&self) -> Vec<u64> {
        self.array.shape().to_vec()
    }

    /// Read every element, row-major.
    pub fn read_all<T: ElementOwned>(&self) -> StorageResult<Vec<T>> {
        self.read_subset::<T>(
            &(0..self.array.shape().len())
                .map(|i| 0..self.array.shape()[i])
                .collect::<Vec<_>>(),
        )
    }

    /// Read a hyperrectangular subset, row-major.
    pub fn read_subset<T: ElementOwned>(&self, ranges: &[Range<u64>]) -> StorageResult<Vec<T>> {
        let subset = ArraySubset::new_with_ranges(ranges);
        self.array
            .retrieve_array_subset::<Vec<T>>(&subset)
            .map_err(map_err)
    }

    /// Append elements along the leading axis: resize, persist the new
    /// shape, then write the tail. Values must tile the trailing extent.
    pub fn append<T: Element>(&mut self, values: &[T]) -> StorageResult<()> {
        let shape = self.array.shape().to_vec();
        let trailing: u64 = shape[1..].iter().product::<u64>().max(1);
        if shape.is_empty() || !(values.len() as u64).is_multiple_of(trailing) {
            return Err(StorageError::Invalid(format!(
                "append length {} does not tile trailing extent {}",
                values.len(),
                trailing
            )));
        }
        let old_len = shape[0];
        let added = values.len() as u64 / trailing;
        let new_shape = shape
            .iter()
            .enumerate()
            .map(|(i, s)| if i == 0 { old_len + added } else { *s })
            .collect::<Vec<_>>();
        self.array.set_shape(new_shape).map_err(map_err)?;
        self.array.store_metadata().map_err(map_err)?;
        let mut ranges: Vec<Range<u64>> = Vec::new();
        ranges.push(old_len..old_len + added);
        ranges.extend(shape[1..].iter().map(|s| 0..*s));
        let subset = ArraySubset::new_with_ranges(&ranges);
        self.array
            .store_array_subset(&subset, values)
            .map_err(map_err)
    }
}

/// Open an existing Zarr v3 array directory.
///
/// Fails with [`StorageError::UnsupportedFormat`] when the directory does
/// not carry `zarr.json` or carries unreadable metadata.
pub fn open(path: &Path) -> StorageResult<ZarrArray> {
    if !path.join("zarr.json").is_file() {
        return Err(StorageError::UnsupportedFormat(format!(
            "not a Zarr v3 array (missing zarr.json): {}",
            path.display()
        )));
    }
    let store = store_for(path)?;
    let array = Array::open(store, "/").map_err(map_open_err)?;
    Ok(ZarrArray { array })
}

/// Create a Zarr v3 array and write every element.
///
/// `compress` selects the `bytes`-only or `bytes`+`zstd` codec chain.
/// Fails with [`StorageError::InvalidState`] when the path already holds
/// an array.
pub fn write_array<T: ZzarrElement>(
    path: &Path,
    shape: &[u64],
    chunks: &[u64],
    values: &[T],
    compress: bool,
) -> StorageResult<()> {
    if path.join("zarr.json").is_file() {
        return Err(StorageError::InvalidState(format!(
            "zzarr array already exists at {}",
            path.display()
        )));
    }
    let total: u64 = shape.iter().product();
    if total != values.len() as u64 {
        return Err(StorageError::Invalid(format!(
            "shape {:?} holds {} elements but {} values were given",
            shape,
            total,
            values.len()
        )));
    }
    std::fs::create_dir_all(path).map_err(|e| StorageError::Io(e.to_string()))?;
    let store = store_for(path)?;
    let mut builder = zarrs::array::ArrayBuilder::new(
        shape.to_vec(),
        chunks.to_vec(),
        T::zzarr_data_type(),
        T::zzarr_fill(),
    );
    if compress {
        builder.bytes_to_bytes_codecs(vec![Arc::new(ZstdCodec::new(3, false))]);
    }
    let array = builder.build(store, "/").map_err(map_open_err)?;
    array.store_metadata().map_err(map_err)?;
    let subset_all = array.subset_all();
    array
        .store_array_subset(&subset_all, values)
        .map_err(map_err)
}
