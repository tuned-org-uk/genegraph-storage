#![allow(async_fn_in_trait)]
pub mod lance;
pub mod metadata;
pub mod traits;

#[cfg(test)]
mod tests;

use std::fmt;

// Error Handling harness
#[derive(Debug)]
pub enum StorageError {
    Io(String),
    Serde(serde_json::Error),
    Parquet(String),
    Invalid(String),
    Lance(String),
    QueryError(String),
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
