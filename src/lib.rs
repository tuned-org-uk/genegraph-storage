#![allow(async_fn_in_trait)]
pub mod lance_storage_graph;
pub mod lancefmt;
pub mod metadata;
pub mod traits;

#[cfg(test)]
mod tests;

use std::fmt;

// Error Handling harness
//
// `#[non_exhaustive]` lets this crate add error variants in future releases
// without breaking downstream `match` expressions that carry a wildcard arm.
#[derive(Debug)]
#[non_exhaustive]
pub enum StorageError {
    Io(String),
    Serde(serde_json::Error),
    Parquet(String),
    Invalid(String),
    Lance(String),
    QueryError(String),
    /// A storage resource is in an unexpected state (e.g. the metadata path
    /// passed to a `save_*` call does not match the instance metadata path).
    InvalidState(String),
    /// A filetype does not map to a known storage format.
    UnsupportedFormat(String),
    /// A key/filetype is not recognised as one of the supported file types.
    UnsupportedFiletype(String),
    /// Dimensions recorded in schema metadata do not match the dimensions
    /// recorded in storage metadata.
    DimensionMismatch {
        expected: String,
        found: String,
    },
    /// A numeric value exceeds the range of the storage type it is written to.
    Overflow(String),
}

impl fmt::Display for StorageError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            StorageError::Io(msg) => write!(f, "IO error: {}", msg),
            StorageError::Serde(msg) => write!(f, "Serialization error: {}", msg),
            StorageError::Parquet(msg) => write!(f, "Parquet error: {}", msg),
            StorageError::Invalid(msg) => write!(f, "Invalid data: {}", msg),
            StorageError::Lance(msg) => write!(f, "Lance error: {}", msg),
            StorageError::QueryError(msg) => write!(f, "Query error: {}", msg),
            StorageError::InvalidState(msg) => write!(f, "Invalid state: {}", msg),
            StorageError::UnsupportedFormat(msg) => write!(f, "Unsupported format: {}", msg),
            StorageError::UnsupportedFiletype(msg) => write!(f, "Unsupported filetype: {}", msg),
            StorageError::DimensionMismatch { expected, found } => {
                write!(
                    f,
                    "Dimension mismatch: expected {}, found {}",
                    expected, found
                )
            }
            StorageError::Overflow(msg) => write!(f, "Overflow error: {}", msg),
        }
    }
}

impl std::error::Error for StorageError {}

pub type StorageResult<T> = Result<T, StorageError>;

// Logging harness
use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        // Read RUST_LOG env variable, default to "info" if not set
        let env = env_logger::Env::default().default_filter_or("debug");

        // don't panic if called multiple times across binaries
        let _ = env_logger::Builder::from_env(env).try_init();
    });
}
