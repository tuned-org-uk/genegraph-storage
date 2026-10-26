use crate::traits::backend::StorageBackend;
use smartcore::linalg::basic::matrix::DenseMatrix;

#[cfg(not(feature = "parquet"))]
#[tokio::test]
async fn disabled_parquet_load_rejects_before_file_lookup() {
    let root = crate::tests::tmp_dir("disabled-parquet-load").await;
    let path = root.join("artifact.parquet");
    let storage = crate::lance_storage_graph::LanceStorageGraph::new(
        root.to_string_lossy().to_string(),
        "feature_test".into(),
    )
    .unwrap();
    let error = storage.load_dense_from_file(&path).await.unwrap_err();
    assert!(matches!(error, crate::StorageError::UnsupportedFormat(_)));
}

#[cfg(not(feature = "npy"))]
#[tokio::test]
async fn disabled_npy_write_rejects_before_parent_creation() {
    let root = crate::tests::tmp_dir("disabled-npy-write").await;
    let parent = root.join("missing");
    let path = parent.join("artifact.npy");
    let matrix = DenseMatrix::new(1, 1, vec![1.0], true).unwrap();
    let error = crate::lance_storage_graph::LanceStorageGraph::save_dense_to_file(&matrix, &path)
        .await
        .unwrap_err();
    assert!(matches!(error, crate::StorageError::UnsupportedFormat(_)));
    assert!(!parent.exists());
}

#[cfg(not(feature = "parquet"))]
#[tokio::test]
async fn disabled_parquet_write_rejects_before_parent_creation() {
    let root = crate::tests::tmp_dir("disabled-parquet-write").await;
    let parent = root.join("missing");
    let path = parent.join("artifact.parquet");
    let matrix = DenseMatrix::new(1, 1, vec![1.0], true).unwrap();
    let error = crate::lance_storage_graph::LanceStorageGraph::save_dense_to_file(&matrix, &path)
        .await
        .unwrap_err();
    assert!(matches!(error, crate::StorageError::UnsupportedFormat(_)));
    assert!(!parent.exists());
}

#[tokio::test]
async fn unknown_extension_load_returns_unsupported_format() {
    let root = crate::tests::tmp_dir("unknown-extension-load").await;
    let path = root.join("artifact.txt");
    std::fs::write(&path, b"present").unwrap();
    let storage = crate::lance_storage_graph::LanceStorageGraph::new(
        root.to_string_lossy().to_string(),
        "feature_test".into(),
    )
    .unwrap();
    let error = storage.load_dense_from_file(&path).await.unwrap_err();
    assert!(matches!(error, crate::StorageError::UnsupportedFormat(_)));
}

#[tokio::test]
async fn unknown_extension_write_returns_unsupported_format_before_parent_creation() {
    let root = crate::tests::tmp_dir("unknown-extension-write").await;
    let parent = root.join("missing");
    let path = parent.join("artifact.txt");
    let matrix = DenseMatrix::new(1, 1, vec![1.0], true).unwrap();
    let error = crate::lance_storage_graph::LanceStorageGraph::save_dense_to_file(&matrix, &path)
        .await
        .unwrap_err();
    assert!(matches!(error, crate::StorageError::UnsupportedFormat(_)));
    assert!(!parent.exists());
}

#[cfg(not(any(feature = "parquet", feature = "zarr")))]
#[tokio::test]
async fn lance_file_io_remains_available_without_parquet_or_zarr() {
    let root = crate::tests::tmp_dir("lance-no-optional-formats").await;
    let path = root.join("dense.lance");
    let matrix = DenseMatrix::new(1, 1, vec![2.5], true).unwrap();
    crate::lance_storage_graph::LanceStorageGraph::save_dense_to_file(&matrix, &path)
        .await
        .unwrap();
}
