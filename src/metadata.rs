#![allow(dead_code)]

use log::{debug, info};
use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, HashMap};
use std::path::PathBuf;

use crate::StorageError;
use crate::StorageResult;
use crate::traits::backend::StorageBackend;
use crate::traits::metadata::Metadata;

/// First-class collection kind (RFC #81-P1).
///
/// Every named collection in the metadata registry is either a vector space
/// (dense vector collections and scalar artifacts), a graph (edge/adjacency
/// artifacts), or a plain table (anything else).
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "kebab-case")]
pub enum CollectionKind {
    /// Dense vector collections (embeddings, maps, norms, lambdas, ...).
    VectorSpace,
    /// Graph data (edge lists, adjacency, laplacian).
    Graph,
    /// Anything else stored as a table.
    Table,
}

impl CollectionKind {
    /// Registry / descriptor string form (Generic Table API property value).
    pub const fn as_str(&self) -> &'static str {
        match self {
            CollectionKind::VectorSpace => "vector-space",
            CollectionKind::Graph => "graph",
            CollectionKind::Table => "table",
        }
    }

    /// Parses the string form ([`Self::as_str`]).
    pub fn parse(s: &str) -> StorageResult<Self> {
        match s {
            "vector-space" => Ok(CollectionKind::VectorSpace),
            "graph" => Ok(CollectionKind::Graph),
            "table" => Ok(CollectionKind::Table),
            other => Err(StorageError::Invalid(format!(
                "unknown collection kind '{other}' (expected vector-space, graph or table)"
            ))),
        }
    }

    /// Derives the kind from a legacy artifact filetype; the compat shim
    /// that turns the fixed artifact keys (`rawinput`, `adjacency`, ...)
    /// into pre-seeded collections (RFC #81-P1).
    pub fn for_filetype(filetype: &str) -> Option<Self> {
        match filetype {
            "dense" | "vectors" | "vector" => Some(CollectionKind::VectorSpace),
            "sparse" | "graph" => Some(CollectionKind::Graph),
            _ => None,
        }
    }
}

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
    /// First-class collection kind (RFC #81-P1); `None` for entries written
    /// before 0.28 (kind is then derived from `filetype`).
    #[serde(default)]
    pub kind: Option<CollectionKind>,
    /// User properties attached to the collection (RFC #81-P1), e.g. the
    /// `graph` linkage of a vector space (RFC #81-P4).
    #[serde(default)]
    pub properties: BTreeMap<String, String>,
}

impl FileInfo {
    /// Create a file spec to add to the persistence directory
    ///
    /// Fails with `StorageError::UnsupportedFormat` for unrecognised
    /// filetypes instead of panicking (issue #45).
    pub fn new(
        filename: String,
        filetype: &str,
        data_shape: (usize, usize),
        nnz: Option<usize>,
        size_bytes: Option<u64>,
    ) -> StorageResult<Self> {
        debug!(
            "FileInfo::new: filename={}, filetype={}, shape={}x{}, nnz={:?}",
            filename, filetype, data_shape.0, data_shape.1, nnz
        );
        Ok(Self {
            filename,
            filetype: filetype.into(),
            storage_format: Self::which_format(filetype)?,
            rows: data_shape.0,
            cols: data_shape.1,
            nnz,
            size_bytes,
            kind: CollectionKind::for_filetype(filetype),
            properties: BTreeMap::new(),
        })
    }

    /// Assign the right format to the file type
    pub fn which_format(filetype: &str) -> StorageResult<String> {
        match filetype {
            "dense" | "vectors" => Ok(String::from("lance fixed-row")),
            "sparse" => Ok(String::from("lance row-major")),
            "vector" => Ok(String::from("lance row-major")),
            "graph" => Ok(String::from("lance edge-list")),
            other => Err(StorageError::UnsupportedFormat(other.to_string())),
        }
    }

    /// Assign the right filetype to the keyname of the file
    pub fn which_filetype(filetype: &str) -> StorageResult<String> {
        match filetype {
            "rawinput" | "sub_centroids" | "dense" => Ok(String::from("dense")),
            "adjacency" | "laplacian" | "signals" | "sparse" => Ok(String::from("sparse")),
            "lambdas" | "item_norms" | "norms" | "vector" => Ok(String::from("vector")),
            "graph" | "vectors" => Ok(filetype.into()),
            other => Err(StorageError::UnsupportedFiletype(other.to_string())),
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
    /// Read metadata file from JSON
    pub async fn read(path: PathBuf) -> Result<Self, StorageError> {
        info!("Reading metadata from {:?}", path);
        let s = tokio::fs::read_to_string(path)
            .await
            .map_err(|e| StorageError::Io(e.to_string()))?;
        let md: GeneMetadata = serde_json::from_str(&s).map_err(StorageError::Serde)?;
        info!("Metadata read successfully");
        Ok(md)
    }
}

impl GeneMetadata {
    /// Inherent constructor: mirrors [`Metadata::new`] so the common
    /// construction chain works without importing the trait (0.25.0
    /// ergonomics).
    pub fn new(name_id: &str) -> Self {
        <Self as Metadata>::new(name_id)
    }

    /// Inherent builder step: mirrors [`Metadata::with_base`].
    pub fn with_base(self, base_path: PathBuf) -> Self {
        <Self as Metadata>::with_base(self, base_path)
    }

    /// Inherent builder step: mirrors [`Metadata::with_dimensions`].
    pub fn with_dimensions(self, rows: usize, cols: usize) -> Self {
        <Self as Metadata>::with_dimensions(self, rows, cols)
    }

    /// Inherent builder step: mirrors [`Metadata::add_file`].
    pub fn add_file(self, key: &str, info: FileInfo) -> Self {
        <Self as Metadata>::add_file(self, key, info)
    }

    /// Inherent helper: mirrors [`Metadata::new_fileinfo`].
    pub fn new_fileinfo(
        &self,
        key: &str,
        filetype: &str,
        data_shape: (usize, usize),
        nnz: Option<usize>,
        size_bytes: Option<u64>,
    ) -> StorageResult<FileInfo> {
        <Self as Metadata>::new_fileinfo(self, key, filetype, data_shape, nnz, size_bytes)
    }
}

impl Metadata for GeneMetadata {
    /// Empty metadata object
    /// do not use in test, use seed_metadata_eigen instead
    fn new(name_id: &str) -> Self {
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

    fn new_fileinfo(
        &self,
        key: &str,
        filetype: &str,
        data_shape: (usize, usize),
        nnz: Option<usize>,
        size_bytes: Option<u64>,
    ) -> StorageResult<FileInfo> {
        FileInfo::new(
            format!("{}_{}.lance", self.name_id, key),
            filetype,
            (data_shape.0, data_shape.1),
            nnz,
            size_bytes,
        )
    }

    /// Standard pipeline object
    async fn seed_metadata<B: StorageBackend>(
        name_id: &str,
        nitems: usize,
        nfeatures: usize,
        storage: &B,
    ) -> Result<GeneMetadata, StorageError> {
        info!(
            "GeneMetadata::seed_metadata: seeding metadata for '{}' with nitems={}, nfeatures={}",
            name_id, nitems, nfeatures
        );

        let md = Self::new(name_id)
            .with_base(storage.base_path())
            .with_dimensions(nitems, nfeatures);

        debug!("GeneMetadata::seed_metadata: saving metadata to storage");
        storage.save_metadata(&md).await?;

        info!(
            "GeneMetadata::seed_metadata: metadata seeded successfully for '{}'",
            name_id
        );
        Ok(md)
    }

    fn with_base(mut self, base_path: PathBuf) -> Self {
        self.base = base_path.to_string_lossy().to_string();
        self
    }

    fn with_dimensions(mut self, rows: usize, cols: usize) -> Self {
        debug!(
            "GeneMetadata::with_dimensions: setting dimensions to {}x{}",
            rows, cols
        );
        self.nrows = rows;
        self.ncols = cols;
        self
    }

    fn add_file(mut self, key: &str, info: FileInfo) -> Self {
        debug!(
            "GeneMetadata::add_file: adding file '{}' ({})",
            key, info.filename
        );
        self.files.insert(key.to_string(), info);
        self
    }
}
