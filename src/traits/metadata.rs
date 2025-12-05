use std::path::PathBuf;

use crate::StorageError;
use crate::metadata::FileInfo;
use crate::metadata::GeneMetadata;
use crate::traits::backend::StorageBackend;

/// A trait to create metadata structures.
/// Instantiate structures with this trait, like for `GeneMetaData`
pub trait Metadata {
    /// constructor
    fn new(name_id: &str) -> Self;
    /// instantiate a new file info instance for this Metadata type
    fn new_fileinfo(
        &self,
        key: &str,
        filetype: &str,
        data_shape: (usize, usize),
        nnz: Option<usize>,
        size_bytes: Option<u64>,
    ) -> FileInfo;

    /// Standard pipeline object
    async fn seed_metadata<B: StorageBackend>(
        name_id: &str,
        nitems: usize,
        nfeatures: usize,
        storage: &B,
    ) -> Result<GeneMetadata, StorageError>;

    /// add a file to the metadata files
    fn add_file(self, key: &str, info: FileInfo) -> Self;

    // constructor helpers
    fn with_base(self, base_path: PathBuf) -> Self;
    fn with_dimensions(self, rows: usize, cols: usize) -> Self;
}
