#[cfg(not(feature = "parquet"))]
use std::path::PathBuf;

use crate::traits::backend::StorageBackend;
use smartcore::linalg::basic::matrix::DenseMatrix;

#[cfg(not(feature = "parquet"))]
fn unopened_path(ext: &str) -> PathBuf {
    std::env::temp_dir()
        .join(format!("format-disabled-{}", uuid::Uuid::new_v4()))
        .join(format!("artifact.{ext}"))
}

#[cfg(not(feature = "parquet"))]
#[test]
fn disabled_parquet_load_rejects_before_file_lookup() {
    let path = unopened_path("parquet");
    let storage = crate::lance_storage_graph::LanceStorageGraph::new(
        std::env::temp_dir().to_string_lossy().to_string(),
        "feature_test".into(),
    )
    .unwrap();
    let error = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(storage.load_dense_from_file(&path))
        .unwrap_err();
    assert!(matches!(error, crate::StorageError::UnsupportedFormat(_)));
}

#[cfg(not(feature = "npy"))]
#[test]
fn disabled_npy_write_rejects_before_parent_creation() {
    let root = std::env::temp_dir().join(format!("npy-disabled-{}", uuid::Uuid::new_v4()));
    let path = root.join("missing").join("artifact.npy");
    let matrix = DenseMatrix::new(1, 1, vec![1.0], true).unwrap();
    let error = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(crate::lance_storage_graph::LanceStorageGraph::save_dense_to_file(&matrix, &path))
        .unwrap_err();
    assert!(matches!(error, crate::StorageError::UnsupportedFormat(_)));
    assert!(!root.exists());
}

#[cfg(not(feature = "parquet"))]
#[test]
fn disabled_parquet_write_rejects_before_parent_creation() {
    let root = std::env::temp_dir().join(format!("parquet-disabled-{}", uuid::Uuid::new_v4()));
    let path = root.join("missing").join("artifact.parquet");
    let matrix = DenseMatrix::new(1, 1, vec![1.0], true).unwrap();
    let error = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(crate::lance_storage_graph::LanceStorageGraph::save_dense_to_file(&matrix, &path))
        .unwrap_err();
    assert!(matches!(error, crate::StorageError::UnsupportedFormat(_)));
    assert!(!root.exists());
}

#[test]
fn lance_file_io_remains_available_without_optional_features() {
    let root = std::env::temp_dir().join(format!("lance-default-{}", uuid::Uuid::new_v4()));
    let path = root.join("dense.lance");
    let matrix = DenseMatrix::new(1, 1, vec![2.5], true).unwrap();
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(crate::lance_storage_graph::LanceStorageGraph::save_dense_to_file(&matrix, &path))
        .unwrap();
}
