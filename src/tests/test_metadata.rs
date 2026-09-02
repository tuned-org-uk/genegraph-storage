use crate::StorageError;
use crate::lance_storage_graph::LanceStorageGraph;
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
        original_metadata
            .new_fileinfo("rawinput", "dense", (500, 128), None, Some(65536))
            .expect("dense is a known filetype"),
    );

    original_metadata.files.insert(
        "lambdas".to_string(),
        original_metadata
            .new_fileinfo("lambdas", "vector", (500, 128), None, Some(65536))
            .expect("vector is a known filetype"),
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
    assert!(loaded_metadata.files.is_empty());
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

// ===== Issue #47: validate_initialized must return an error, not assert_eq!-panic =====

#[tokio::test(flavor = "multi_thread")]
async fn test_validate_initialized_mismatched_md_path_returns_error() {
    let name_id = "validate_mismatch";
    let base_path = tmp_dir(name_id).await;
    let storage =
        LanceStorageGraph::new(base_path.to_string_lossy().to_string(), name_id.to_string());

    let wrong_path = base_path.join("other_instance_metadata.json");
    let result = storage.validate_initialized(&wrong_path);

    match result {
        Err(StorageError::InvalidState(msg)) => {
            assert!(
                msg.contains("mismatch"),
                "message should mention the mismatch: {}",
                msg
            );
            assert!(
                msg.contains(&wrong_path.to_string_lossy().to_string()),
                "message should include the found path: {}",
                msg
            );
        }
        other => panic!(
            "expected StorageError::InvalidState for mismatched path, got {:?}",
            other.err()
        ),
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_validate_initialized_matching_path_without_metadata_returns_invalid() {
    let name_id = "validate_missing";
    let base_path = tmp_dir(name_id).await;
    let storage =
        LanceStorageGraph::new(base_path.to_string_lossy().to_string(), name_id.to_string());

    let result = storage.validate_initialized(&storage.metadata_path());
    assert!(
        matches!(result, Err(StorageError::Invalid(_))),
        "expected Invalid (metadata missing), got {:?}",
        result.err()
    );
}

// ===== Issue #45: FileInfo classification must return errors, not panic =====

#[test]
fn test_which_format_maps_known_filetypes() {
    assert_eq!(FileInfo::which_format("dense").unwrap(), "lance fixed-row");
    assert_eq!(FileInfo::which_format("sparse").unwrap(), "lance row-major");
    assert_eq!(FileInfo::which_format("vector").unwrap(), "lance row-major");
}

#[test]
fn test_which_format_rejects_unknown_filetype_with_error() {
    let result = FileInfo::which_format("bogus_format");
    assert!(
        matches!(result, Err(StorageError::UnsupportedFormat(ref s)) if s == "bogus_format"),
        "expected UnsupportedFormat error, got {:?}",
        result
    );
}

#[test]
fn test_which_filetype_maps_known_keys() {
    assert_eq!(FileInfo::which_filetype("rawinput").unwrap(), "dense");
    assert_eq!(FileInfo::which_filetype("sub_centroids").unwrap(), "dense");
    assert_eq!(FileInfo::which_filetype("adjacency").unwrap(), "sparse");
    assert_eq!(FileInfo::which_filetype("laplacian").unwrap(), "sparse");
    assert_eq!(FileInfo::which_filetype("signals").unwrap(), "sparse");
    assert_eq!(FileInfo::which_filetype("lambdas").unwrap(), "vector");
    assert_eq!(FileInfo::which_filetype("item_norms").unwrap(), "vector");
    assert_eq!(FileInfo::which_filetype("norms").unwrap(), "vector");
}

#[test]
fn test_which_filetype_rejects_unknown_key_with_error() {
    let result = FileInfo::which_filetype("bogus_key");
    assert!(
        matches!(result, Err(StorageError::UnsupportedFiletype(ref s)) if s == "bogus_key"),
        "expected UnsupportedFiletype error, got {:?}",
        result
    );
}

#[test]
fn test_fileinfo_new_rejects_unknown_filetype_with_error() {
    let result = FileInfo::new("bogus.lance".to_string(), "bogus", (10, 10), None, None);
    assert!(
        matches!(result, Err(StorageError::UnsupportedFormat(_))),
        "expected UnsupportedFormat error, got {:?}",
        result
    );
}

#[test]
fn test_fileinfo_new_accepts_known_filetype() {
    let info = FileInfo::new("f.lance".to_string(), "dense", (4, 2), None, None)
        .expect("dense is a known filetype");
    assert_eq!(info.filetype, "dense");
    assert_eq!(info.storage_format, "lance fixed-row");
    assert_eq!(info.rows, 4);
    assert_eq!(info.cols, 2);
}

// ===== Ergonomic constructors: reachable without importing the Metadata trait =====

/// Isolated scope: imports ONLY the types, not the `Metadata` trait. If the
/// construction chain below compiles and runs, the inherent impls on
/// `GeneMetadata` are doing their job.
mod inherent_only {
    use crate::lance_storage_graph::LanceStorageGraph;
    use crate::metadata::GeneMetadata;
    // StorageBackend is unrelated to the Metadata-trait ergonomics under
    // test here; base_path() needs it.
    use crate::traits::backend::StorageBackend;

    #[test]
    fn gene_metadata_new_without_trait_import() {
        let md = GeneMetadata::new("no_trait_import");
        assert_eq!(md.name_id, "no_trait_import");
        assert_eq!(md.nrows, 0);
        assert!(md.files.is_empty());
    }

    #[test]
    fn construction_chain_without_trait_import() {
        let base = std::env::temp_dir();
        let storage = LanceStorageGraph::new(
            base.join("no_trait").to_string_lossy().to_string(),
            "no_trait_import".to_string(),
        );
        let md = GeneMetadata::new("no_trait_import")
            .with_base(storage.base_path())
            .with_dimensions(10, 5);
        assert_eq!(md.nrows, 10);
        assert_eq!(md.ncols, 5);
        assert_eq!(md.base, storage.base_path().to_string_lossy().to_string());

        let info = md
            .new_fileinfo("rawinput", "dense", (10, 5), None, None)
            .expect("dense filetype");
        let md = md.add_file("rawinput", info);
        assert!(md.files.contains_key("rawinput"));
    }
}
