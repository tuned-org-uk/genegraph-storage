#![allow(dead_code)]

use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;

use crate::StorageError;
use crate::traits::StorageBackend;

/// Represent a single file spec in the persistence directory
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileInfo {
    /// name of the file, can be equal to filetype if there is only one per type
    pub filename: String,
    /// see `Self::which_filetype(..)`: "rawinput" | "sub_centroids" | "lambdas" | "..."
    pub filetype: String,
    /// see `Self::which_format(..)`
    pub storage_format: String,
    pub rows: usize,
    pub cols: usize,
    pub nnz: Option<usize>,
    pub size_bytes: Option<u64>,
}

impl FileInfo {
    /// Create a file spec to add to the persistence directory
    pub fn new(
        filename: String,
        filetype: &str,
        data_shape: (usize, usize),
        nnz: Option<usize>,
        size_bytes: Option<u64>,
    ) -> Self {
        debug!(
            "FileInfo::new: filename={}, filetype={}, shape={}x{}, nnz={:?}",
            filename, filetype, data_shape.0, data_shape.1, nnz
        );
        Self {
            filename,
            filetype: filetype.into(),
            storage_format: Self::which_format(filetype),
            rows: data_shape.0,
            cols: data_shape.1,
            nnz,
            size_bytes,
        }
    }

    /// Assign the right format to the file type
    pub fn which_format(filetype: &str) -> String {
        match filetype {
            "dense" => String::from("lance fixed-row"),
            "sparse" => String::from("lance row-major"),
            "vector" => String::from("lance row-major"),
            _ => panic!("filetype not recognised {}", filetype),
        }
    }

    /// Assign the right filetype to the keyname of the file
    pub fn which_filetype(filetype: &str) -> String {
        match filetype {
            "rawinput" | "sub_centroids" => String::from("dense"),
            "adjacency" | "laplacian" | "signals" => String::from("sparse"),
            "lambdas" | "item_norms" | "norms" => String::from("vector"),
            _ => panic!("key not recognised {}", filetype),
        }
    }
}

/// Metadata for an ArrowSpace index persisted to Lance storage.
///
/// Tracks dataset dimensions, builder configuration, file locations, and pipeline context.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GeneMetadata {
    pub name_id: String,
    pub nrows: usize,
    pub ncols: usize,
    pub base: String,
    pub files: HashMap<String, FileInfo>,
    pub created_at: String,
}

impl GeneMetadata {
    /// Empty metadata object
    /// do not use in test, use seed_metadata_eigen instead
    pub fn new(name_id: &str) -> Self {
        info!("GeneMetadata::new: creating metadata for '{}'", name_id);
        Self {
            name_id: name_id.to_string(),
            nrows: 0,
            ncols: 0,
            base: String::from(""),
            files: HashMap::new(),
            created_at: chrono::Utc::now().to_rfc3339(),
        }
    }

    pub fn new_fileinfo(
        &self,
        key: &str,
        filetype: &str,
        data_shape: (usize, usize),
        nnz: Option<usize>,
        size_bytes: Option<u64>,
    ) -> FileInfo {
        FileInfo::new(
            format!("{}_{}.lance", self.name_id, key),
            filetype,
            (data_shape.0, data_shape.1),
            nnz,
            size_bytes,
        )
    }

    /// Read metadata file from JSON
    pub async fn read(path: PathBuf) -> Result<Self, StorageError> {
        info!("Reading metadata from {:?}", path);
        let s = fs::read_to_string(path).map_err(|e| StorageError::Io(e.to_string()))?;
        let md: GeneMetadata = serde_json::from_str(&s).map_err(StorageError::Serde)?;
        info!("Metadata read successfully");
        Ok(md)
    }

    /// Standard pipeline object
    pub async fn seed_metadata<B: StorageBackend>(
        name_id: &str,
        nitems: usize,
        nfeatures: usize,
        storage: &B,
    ) -> Result<GeneMetadata, StorageError> {
        info!(
            "GeneMetadata::seed_metadata: seeding metadata for '{}' with nitems={}, nfeatures={}",
            name_id, nitems, nfeatures
        );

        let mut md = Self::new(name_id)
            .with_base(storage.base_path())
            .with_dimensions(nitems, nfeatures);

        debug!("GeneMetadata::seed_metadata: registering files");

        let (key, filetype, rows, cols, nnz) = ("rawinput", "dense", nitems, nfeatures, None);
        debug!(
            "SpaceMetadata::seed_metadata_eigen: adding file {} ({}x{}, nnz={:?})",
            filetype, rows, cols, nnz
        );
        md = md.add_file(
            key,
            FileInfo::new(
                format!("{}_{}.lance", name_id, key),
                filetype,
                (rows, cols),
                nnz,
                None,
            ),
        );

        debug!("GeneMetadata::seed_metadata: saving metadata to storage");
        storage.save_metadata(&md).await?;

        info!(
            "GeneMetadata::seed_metadata: metadata seeded successfully for '{}'",
            name_id
        );
        Ok(md)
    }

    pub fn with_base(mut self, base_path: PathBuf) -> Self {
        self.base = base_path.to_string_lossy().to_string();
        self
    }

    pub fn with_dimensions(mut self, rows: usize, cols: usize) -> Self {
        debug!(
            "GeneMetadata::with_dimensions: setting dimensions to {}x{}",
            rows, cols
        );
        self.nrows = rows;
        self.ncols = cols;
        self
    }

    pub fn add_file(mut self, key: &str, info: FileInfo) -> Self {
        debug!(
            "GeneMetadata::add_file: adding file '{}' ({})",
            key, info.filename
        );
        self.files.insert(key.to_string(), info);
        self
    }
}
