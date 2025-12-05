use crate::StorageError;
use crate::lance_graphdata::LanceStorageGraph;
use crate::metadata::{FileInfo, GeneMetadata};
use crate::tests::tmp_dir;
use crate::traits::backend::StorageBackend;
use crate::traits::metadata::Metadata;

use std::collections::HashMap;
use std::path::PathBuf;

#[tokio::test(flavor = "multi_thread")]
async fn test_metadata_roundtrip_basic() {
    // Setup: Create a temporary directory
    let name_id = "test_roundtrip";
    let base_path = tmp_dir(name_id).await;

    // Create storage instance
    let storage =
        LanceStorageGraph::new(base_path.to_string_lossy().to_string(), name_id.to_string());

    // Create original metadata with all fields populated
    let mut original_metadata = GeneMetadata {
        name_id: name_id.to_string(),
        nrows: 500,
        ncols: 128,
        base: base_path.to_string_lossy().to_string(),
        files: HashMap::new(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };

    // Add file entries to test HashMap serialization
    original_metadata.files.insert(
        "rawinput".to_string(),
        original_metadata.new_fileinfo("rawinput", "dense", (500, 128), None, Some(65536)),
    );

    original_metadata.files.insert(
        "lambdas".to_string(),
        original_metadata.new_fileinfo("lambdas", "vector", (500, 128), None, Some(65536)),
    );

    // Save metadata using StorageBackend trait
    let saved_path = storage
        .save_metadata(&original_metadata)
        .await
        .expect("Failed to save metadata");

    // Verify the file was created at the expected path
    assert!(saved_path.exists());
    assert_eq!(saved_path, storage.metadata_path());

    // Load metadata using StorageBackend::load_metadata
    let loaded_via_storage = storage
        .load_metadata()
        .await
        .expect("Failed to load metadata via storage");

    // Load metadata using GeneMetadata::read directly
    let loaded_via_read = GeneMetadata::read(saved_path.clone())
        .await
        .expect("Failed to read metadata directly");

    // Assert: Both loading methods should produce identical results
    assert_eq!(loaded_via_storage.name_id, loaded_via_read.name_id);
    assert_eq!(loaded_via_storage.nrows, loaded_via_read.nrows);
    assert_eq!(loaded_via_storage.ncols, loaded_via_read.ncols);
    assert_eq!(loaded_via_storage.base, loaded_via_read.base);
    assert_eq!(loaded_via_storage.files.len(), loaded_via_read.files.len());

    // Assert: Loaded metadata matches original
    assert_eq!(loaded_via_read.name_id, original_metadata.name_id);
    assert_eq!(loaded_via_read.nrows, original_metadata.nrows);
    assert_eq!(loaded_via_read.ncols, original_metadata.ncols);
    assert_eq!(loaded_via_read.base, original_metadata.base);
    assert_eq!(loaded_via_read.created_at, original_metadata.created_at);
    assert_eq!(loaded_via_read.files.len(), original_metadata.files.len());

    // Verify file entries were preserved correctly
    let loaded_rawinput = loaded_via_read
        .files
        .get("rawinput")
        .expect("rawinput not found");
    let original_rawinput = original_metadata.files.get("rawinput").unwrap();
    assert_eq!(loaded_rawinput.filename, original_rawinput.filename);
    assert_eq!(loaded_rawinput.filetype, original_rawinput.filetype);
    assert_eq!(loaded_rawinput.rows, original_rawinput.rows);
    assert_eq!(loaded_rawinput.cols, original_rawinput.cols);
    assert_eq!(loaded_rawinput.nnz, original_rawinput.nnz);
    assert_eq!(loaded_rawinput.size_bytes, original_rawinput.size_bytes);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metadata_roundtrip_with_seed_metadata() {
    // Setup
    let name_id = "test_seed_roundtrip";
    let base_path = tmp_dir(name_id).await;

    let storage =
        LanceStorageGraph::new(base_path.to_string_lossy().to_string(), name_id.to_string());

    // Use seed_metadata helper to create initial metadata
    let original_metadata = GeneMetadata::seed_metadata(
        name_id, 1000, // nitems
        256,  // nfeatures
        &storage,
    )
    .await
    .expect("Failed to seed metadata");

    // Verify seed created the file
    let metadata_path = storage.metadata_path();
    assert!(metadata_path.exists());

    // Read it back using GeneMetadata::read
    let loaded_metadata = GeneMetadata::read(metadata_path)
        .await
        .expect("Failed to read seeded metadata");

    // Assert dimensions match
    assert_eq!(loaded_metadata.name_id, name_id);
    assert_eq!(loaded_metadata.nrows, 1000);
    assert_eq!(loaded_metadata.ncols, 256);
    assert_eq!(loaded_metadata.base, original_metadata.base);

    // Verify the rawinput file entry was created by seed_metadata
    assert!(loaded_metadata.files.contains_key("rawinput"));
    let rawinput_info = loaded_metadata.files.get("rawinput").unwrap();
    assert_eq!(rawinput_info.filetype, "dense");
    assert_eq!(rawinput_info.rows, 1000);
    assert_eq!(rawinput_info.cols, 256);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metadata_roundtrip_empty_files() {
    // Test with minimal metadata (no file entries)
    let name_id = "test_empty";
    let base_path = tmp_dir(name_id).await;

    let storage =
        LanceStorageGraph::new(base_path.to_string_lossy().to_string(), name_id.to_string());

    let original_metadata = GeneMetadata {
        name_id: name_id.to_string(),
        nrows: 0,
        ncols: 0,
        base: base_path.to_string_lossy().to_string(),
        files: HashMap::new(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };

    // Save
    let saved_path = storage
        .save_metadata(&original_metadata)
        .await
        .expect("Failed to save empty metadata");

    // Read back
    let loaded_metadata = GeneMetadata::read(saved_path)
        .await
        .expect("Failed to read empty metadata");

    assert_eq!(loaded_metadata.name_id, original_metadata.name_id);
    assert_eq!(loaded_metadata.nrows, 0);
    assert_eq!(loaded_metadata.ncols, 0);
    assert!(loaded_metadata.files.is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metadata_roundtrip_with_sparse_info() {
    // Test with sparse matrix file entries (with nnz field)
    let name_id = "test_sparse";
    let base_path = tmp_dir(name_id).await;

    let storage =
        LanceStorageGraph::new(base_path.to_string_lossy().to_string(), name_id.to_string());

    let mut original_metadata = GeneMetadata {
        name_id: name_id.to_string(),
        nrows: 1000,
        ncols: 1000,
        base: base_path.to_string_lossy().to_string(),
        files: HashMap::new(),
        created_at: chrono::Utc::now().to_rfc3339(),
    };

    // Add sparse matrix file info
    original_metadata.files.insert(
        "laplacian".to_string(),
        FileInfo {
            filename: format!("{}_laplacian.lance", name_id),
            filetype: "sparse".to_string(),
            storage_format: "lance row-major".to_string(),
            rows: 1000,
            cols: 1000,
            nnz: Some(5000), // Sparse matrix with 5000 non-zero entries
            size_bytes: Some(120000),
        },
    );

    // Save and reload
    let saved_path = storage
        .save_metadata(&original_metadata)
        .await
        .expect("Failed to save metadata");

    let loaded_metadata = GeneMetadata::read(saved_path)
        .await
        .expect("Failed to read metadata");

    // Verify sparse matrix metadata
    let laplacian_info = loaded_metadata
        .files
        .get("laplacian")
        .expect("laplacian not found");
    assert_eq!(laplacian_info.filetype, "sparse");
    assert_eq!(laplacian_info.nnz, Some(5000));
    assert_eq!(laplacian_info.rows, 1000);
    assert_eq!(laplacian_info.cols, 1000);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metadata_read_nonexistent_file() {
    // Test error handling for missing file
    let nonexistent_path = PathBuf::from("/tmp/nonexistent_metadata_test_12345.json");

    let result = GeneMetadata::read(nonexistent_path).await;

    assert!(result.is_err());
    match result {
        Err(StorageError::Io(msg)) => {
            assert!(msg.contains("No such file") || msg.contains("cannot find"));
        }
        _ => panic!("Expected StorageError::Io, got {:?}", result),
    }
}
