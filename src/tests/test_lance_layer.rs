use crate::StorageError;
use crate::lance_storage_graph::LanceStorageGraph;
use crate::metadata::FileInfo;
use crate::metadata::GeneMetadata;
use crate::tests::tmp_dir;
use crate::traits::backend::StorageBackend;
use crate::traits::metadata::Metadata;

use log::debug;
use std::path::{Path, PathBuf};

use approx::{assert_relative_eq, relative_eq};
use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::CsMat;

/// Initialise a test dataset with basic builder values
/// To instantiate: `let (aspace, gl) = builder.build_for_persistence(data)`
async fn init_test_builder(
    instance_name: &str,
) -> (
    PathBuf,
    LanceStorageGraph,
    DenseMatrix<f64>,
    CsMat<f64>,
    Vec<f64>,
) {
    let (nitems, nfeatures) = (150, 300);
    let (dense, adjacency, norms) =
        crate::tests::test_data::make_gaussian_cliques_multi(nitems, 0.3, 5, nfeatures, 42);
    let base = tmp_dir(instance_name).await;
    let storage = LanceStorageGraph::new(
        base.to_string_lossy().to_string(),
        instance_name.to_string(),
    );

    let data =
        DenseMatrix::<f64>::from_iterator(dense.iter().flatten().copied(), nitems, nfeatures, 0);

    (base, storage, data, adjacency, norms)
}

#[tokio::test(flavor = "multi_thread")]
async fn test_no_metadata() {
    crate::tests::init();
    let name = "meta_layout";
    let (_, storage, data, _, _) = init_test_builder(name).await;

    // Correct metadata path, but metadata was never seeded: the save must
    // fail with an error instead of writing outside an initialized store.
    let md_path = storage.metadata_path();
    let err = storage
        .save_dense("rawinput", &data, &md_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, StorageError::Invalid(_)),
        "expected Invalid (metadata missing), got {err:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metadata_simple() {
    crate::tests::init();
    let name_id = "meta_layout";
    let (_, storage, data, _, _) = init_test_builder(name_id).await;
    let (nitems, nfeatures) = data.shape();

    // Create metadata
    let md = GeneMetadata::seed_metadata(name_id, nitems, nfeatures, &storage.clone())
        .await
        .unwrap();
    debug!("Saving metadata first to initialize storage directory");
    let md_path: PathBuf = storage.save_metadata(&md).await.unwrap();

    // Assert metadata file was created
    assert!(
        md_path.exists(),
        "Metadata file should exist at {:?}",
        md_path
    );

    // Assert metadata file is in the expected location
    let expected_path = storage.metadata_path();
    assert_eq!(
        md_path, expected_path,
        "Metadata path should match expected location"
    );

    // Load the metadata back and verify content
    let loaded_md = storage.load_metadata().await.unwrap();

    // Verify basic metadata fields
    assert_eq!(loaded_md.name_id, name_id, "Metadata name should match");
    assert_eq!(loaded_md.nrows, nitems, "Metadata rows should match");
    assert_eq!(loaded_md.ncols, nfeatures, "Metadata cols should match");

    // Verify the files HashMap structure
    assert_eq!(loaded_md.files.len(), 0, "Should have no file entry");

    // Verify metadata is valid JSON by checking file size
    let metadata = std::fs::metadata(md_path.clone()).unwrap();
    assert!(
        metadata.len() > 0,
        "Metadata file should have non-zero size"
    );

    // Verify JSON structure by reading raw file
    let json_str = std::fs::read_to_string(md_path.clone()).unwrap();
    let json_value: serde_json::Value = serde_json::from_str(&json_str).unwrap();

    // Assert key JSON fields exist
    assert!(
        json_value.get("name_id").is_some(),
        "JSON should have 'name_id' field"
    );
    assert!(
        json_value.get("nrows").is_some(),
        "JSON should have 'nrows' field"
    );
    assert!(
        json_value.get("ncols").is_some(),
        "JSON should have 'ncols' field"
    );
    assert!(
        json_value.get("files").is_some(),
        "JSON should have 'files' field"
    );
    assert!(
        json_value.get("created_at").is_some(),
        "JSON should have 'created_at' timestamp"
    );

    // Verify JSON is pretty-printed (contains newlines)
    assert!(
        json_str.contains('\n'),
        "Metadata JSON should be pretty-printed"
    );

    debug!("✓ Metadata file created successfully");
    debug!("✓ Metadata content validated");
    debug!("✓ JSON structure verified");
    debug!("✓ Files HashMap validated with correct key-value pair");
    debug!("  Location: {:?}", md_path);
    debug!("  Size: {} bytes", metadata.len());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lance_dense_missing_metadata() {
    crate::tests::init();
    let name = "missing_metadata";
    let (base, storage, data, _, _) = init_test_builder(name).await;

    // A data file path is not a metadata path: validate_initialized must
    // report the mismatch as an error instead of asserting (issue #47).
    let wrong_path = storage.file_path("dense");
    let err = storage
        .save_dense("missing_data", &data, &wrong_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, StorageError::InvalidState(ref msg) if msg.contains(wrong_path.to_string_lossy().as_ref())),
        "expected InvalidState (metadata path mismatch), got {err:?}"
    );
    drop(base);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lance_dense_roundtrip() {
    crate::tests::init();
    let name = "dense_roundtrip";
    let (_, storage, data, _, _) = init_test_builder(name).await;
    let (nitems, nfeatures) = data.shape();

    // Save metadata FIRST to initialize the storage directory
    let md = GeneMetadata::seed_metadata(name, nitems, nfeatures, &storage.clone())
        .await
        .unwrap()
        .with_dimensions(nitems, nfeatures);

    let md_path = storage.save_metadata(&md).await.unwrap();

    assert!(md.files.is_empty());

    storage
        .save_dense("rawinput", &data, &md_path)
        .await
        .unwrap();

    let md = storage.load_metadata().await.unwrap();
    let expected_filename = format!("{}_rawinput.lance", name);
    assert!(md.files.get("rawinput").unwrap().filename == expected_filename);

    let loaded = storage.load_dense("rawinput").await.unwrap();

    assert_eq!(data.shape(), loaded.shape());
    let (rows, cols) = data.shape();

    for r in 0..rows {
        for c in 0..cols {
            let orig = *data.get((r, c));
            let load = *loaded.get((r, c));
            assert!(
                relative_eq!(orig, load, epsilon = 1e-9),
                "Mismatch at ({}, {}): original={}, loaded={}",
                r,
                c,
                orig,
                load
            );
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lance_sparse_roundtrip() {
    crate::tests::init();
    let name = "sparse_roundtrip";
    let (_, storage, data, adjacency, _) = init_test_builder(name).await;
    let (nitems, nfeatures) = data.shape();

    // Create metadata
    let md = GeneMetadata::seed_metadata(name, nitems, nfeatures, &storage)
        .await
        .unwrap();
    debug!("Saving metadata first to initialize storage directory");
    let md_path: PathBuf = storage.save_metadata(&md).await.unwrap();

    storage
        .save_sparse("adjacency", &adjacency, &md_path)
        .await
        .unwrap();

    let md = storage.load_metadata().await.unwrap();
    let expected_filename = format!("{}_adjacency.lance", name);
    assert!(md.files.get("adjacency").unwrap().filename == expected_filename);

    let loaded: CsMat<f64> = storage.load_sparse("adjacency").await.unwrap();

    assert_eq!(adjacency.rows(), loaded.rows());
    assert_eq!(adjacency.cols(), loaded.cols());
    assert_eq!(adjacency.nnz(), loaded.nnz());

    assert_eq!(adjacency, loaded);
}

/// #102 (fail early and typed): a fully disconnected graph (adjacency
/// nnz=0) is a legitimate data state that the sparse artifact path cannot
/// persist. It must be rejected at save time — before the metadata
/// read-modify-write cycle and before any artifact write — so no partial
/// directory is left behind and the consumer gets an actionable error.
#[tokio::test(flavor = "multi_thread")]
async fn save_sparse_rejects_disconnected_nnz_zero_matrix_before_writes() {
    crate::tests::init();
    let name = "sparse_nnz0_rejected";
    let (_, storage, data, _, _) = init_test_builder(name).await;
    let (nitems, nfeatures) = data.shape();

    let md = GeneMetadata::seed_metadata(name, nitems, nfeatures, &storage)
        .await
        .unwrap();
    let md_path: PathBuf = storage.save_metadata(&md).await.unwrap();

    // 8x8 fully disconnected adjacency: nnz=0 with declared shape, exactly
    // the lambda-graph state a too-small eps produces.
    let adjacency = CsMat::zero((8, 8));
    let err = storage
        .save_sparse("adjacency", &adjacency, &md_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, StorageError::Invalid(ref m) if m.contains("nnz=0") && m.contains("disconnected")),
        "expected typed disconnected-graph validation error, got {err:?}"
    );

    // fail-early contract: the registry commit must not have happened —
    // the metadata still has no adjacency entry.
    let md_after = storage.load_metadata().await.unwrap();
    assert!(
        !md_after.files.contains_key("adjacency"),
        "a rejected nnz=0 save must not register metadata (partial directory)"
    );
    // ... and no artifact file may be left behind.
    let artifact = storage
        .base_path()
        .join(format!("{name}_adjacency.lance"));
    assert!(
        !artifact.exists(),
        "a rejected nnz=0 save must not leave an artifact at {artifact:?}"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lambdas_roundtrip() {
    crate::tests::init();
    let name_id = "lambdas_roundtrip";
    let (_, storage, data, _, norms) = init_test_builder(name_id).await;
    let (nitems, nfeatures) = data.shape();

    // Create metadata
    let md = GeneMetadata::seed_metadata(name_id, nitems, nfeatures, &storage)
        .await
        .unwrap();
    debug!("Saving metadata first to initialize storage directory");
    let md_path: PathBuf = storage.save_metadata(&md).await.unwrap();

    storage
        .save_lambdas(norms.as_slice(), &md_path)
        .await
        .unwrap();

    let md = storage.load_metadata().await.unwrap();
    let expected_filename = format!("{}_lambdas.lance", name_id);
    assert!(md.files.get("lambdas").unwrap().filename == expected_filename);

    let loaded = storage.load_lambdas().await.unwrap();

    assert_eq!(norms.len(), loaded.len());
    for (a, b) in norms.iter().zip(loaded.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-10);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metadata_and_files_layout() {
    crate::tests::init();
    let name_id = "test_metadata_and_files_layout";
    let (_, storage, data, adjacency, norms) = init_test_builder(name_id).await;
    let (nitems, nfeatures) = data.shape();

    // Create metadata
    GeneMetadata::seed_metadata(name_id, nitems, nfeatures, &storage)
        .await
        .unwrap();
    debug!("Saving metadata first to initialize storage directory");

    let md_path = storage.metadata_path();
    assert!(md_path.exists());

    storage
        .save_dense("rawinput", &data, &md_path)
        .await
        .unwrap();
    storage
        .save_sparse("adjacency", &adjacency, &md_path)
        .await
        .unwrap();
    storage
        .save_vector("norms", norms.as_slice(), &md_path)
        .await
        .unwrap();

    // Reload metadata and check FileInfo entries for adjacency and norms.
    let md_loaded: GeneMetadata = storage.load_metadata().await.unwrap();

    debug!("{:?}", md_loaded);

    let raw_info = md_loaded
        .files
        .get("rawinput")
        .expect("rawinput entry missing");
    assert_eq!(raw_info.rows, nitems);
    assert_eq!(raw_info.cols, nfeatures);

    let adj_info = md_loaded
        .files
        .get("adjacency")
        .expect("adjacency entry missing");
    assert_eq!(adj_info.rows, nitems);
    assert_eq!(adj_info.cols, nitems);
    assert_eq!(adj_info.nnz, Some(adjacency.nnz()));

    let norms_info = md_loaded.files.get("norms").expect("norms entry missing");
    assert_eq!(norms_info.rows, nitems);
    assert_eq!(norms_info.cols, 1);
    assert_eq!(norms_info.nnz, None);

    // Reload adjacency and norms from storage and check content.
    let loaded_adj = storage
        .load_sparse("adjacency")
        .await
        .map_err(|e| panic!("{:?}", e))
        .unwrap();
    assert_eq!(loaded_adj.shape(), adjacency.shape());
    assert_eq!(loaded_adj.nnz(), adjacency.nnz());
    assert_eq!(loaded_adj.indptr(), adjacency.indptr());
    assert_eq!(loaded_adj.indices(), adjacency.indices());
    assert_eq!(loaded_adj.data(), adjacency.data());

    let loaded_norms = storage
        .load_vector("norms")
        .await
        .map_err(|e| panic!("{:?}", e))
        .unwrap();
    assert_eq!(loaded_norms.len(), norms.len());
    for (a, b) in norms.iter().zip(loaded_norms.iter()) {
        assert!(
            relative_eq!(a, b, epsilon = 1e-10),
            "mismatching norm: {} != {}",
            a,
            b
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metadata_persistence() {
    crate::tests::init();
    let name = "metadata";

    let (_, storage, data, _, _) = init_test_builder(name).await;
    let (nitems, nfeatures) = data.shape();

    GeneMetadata::seed_metadata(name, nitems, nfeatures, &storage)
        .await
        .unwrap();

    let loaded_md = storage.load_metadata().await.unwrap();

    assert_eq!(loaded_md.name_id, name);
    assert_eq!(loaded_md.nrows, nitems);
    assert_eq!(loaded_md.ncols, nfeatures);

    assert!(loaded_md.name_id == name);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_storage_instances() {
    crate::tests::init();
    let base = tmp_dir("test_concurrent_storage_instances").await;
    let name = "concurrent";

    let storage1 =
        LanceStorageGraph::new(base.to_string_lossy().to_string(), "instance1".to_string());
    let storage2 =
        LanceStorageGraph::new(base.to_string_lossy().to_string(), "instance2".to_string());

    let data1 = vec![1.0, 2.0, 3.0, 4.0];
    let data2 = vec![5.0, 6.0, 7.0, 8.0];

    let mat1 = DenseMatrix::new(2, 2, data1, true).unwrap();
    let (nitems1, nfeatures1) = mat1.shape();
    let mat2 = DenseMatrix::new(2, 2, data2, true).unwrap();
    let (nitems2, nfeatures2) = mat2.shape();

    let path1 = storage1.file_path("test");
    let path2 = storage2.file_path("test");

    debug!("Saving:\n{:?}\n{:?}", path1, path2);
    // Create metadata
    let md1 = GeneMetadata::seed_metadata(name, nitems1, nfeatures1, &storage1)
        .await
        .unwrap();
    let md2 = GeneMetadata::seed_metadata(name, nitems2, nfeatures2, &storage2)
        .await
        .unwrap();
    debug!("Saving metadata first to initialize storage directory");
    let md_path1: PathBuf = storage1.save_metadata(&md1).await.unwrap();
    let md_path2: PathBuf = storage2.save_metadata(&md2).await.unwrap();

    storage1
        .save_dense("matrix1", &mat1, &md_path1)
        .await
        .unwrap();
    storage2
        .save_dense("matrix2", &mat2, &md_path2)
        .await
        .unwrap();

    let loaded1 = storage1.load_dense("matrix1").await.unwrap();
    let loaded2 = storage2.load_dense("matrix2").await.unwrap();

    for r in 0..2 {
        for c in 0..2 {
            assert_relative_eq!(mat1.get((r, c)), loaded1.get((r, c)), epsilon = 1e-9);
            assert_relative_eq!(mat2.get((r, c)), loaded2.get((r, c)), epsilon = 1e-9);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lance_storage_spawn() {
    // Setup: Create a temporary directory
    let temp_dir = tmp_dir("test_concurrent_storage_instances").await;
    let base_path = temp_dir.as_path().to_str().unwrap().to_string();
    let name_id = "test_spawn_storage";

    // Step 1: Create and seed initial storage with metadata
    let storage = LanceStorageGraph::new(base_path.clone(), name_id.to_string());

    GeneMetadata::seed_metadata(
        name_id, 100, // nitems
        50,  // nfeatures
        &storage,
    )
    .await
    .expect("Failed to seed metadata");

    // Step 2: Save some sample data to make it realistic
    let test_matrix = DenseMatrix::from_2d_array(&[&[1.0, 2.0, 3.0], &[4.0, 5.0, 6.0]])
        .expect("Failed to create test matrix");

    storage
        .save_dense("rawinput", &test_matrix, &storage.metadata_path())
        .await
        .expect("Failed to save dense matrix");

    // Save lambdas
    let lambdas = vec![0.1, 0.2, 0.3];
    storage
        .save_lambdas(&lambdas, &storage.metadata_path())
        .await
        .expect("Failed to save lambdas");

    // Step 3: Now spawn from the existing directory
    let (spawned_storage, spawned_metadata) = LanceStorageGraph::spawn(base_path.clone())
        .await
        .expect("Failed to spawn LanceStorage");

    // Assertions: Verify spawned storage matches original
    assert_eq!(spawned_storage.get_base(), base_path);
    assert_eq!(spawned_storage.get_name(), name_id);
    assert_eq!(spawned_metadata.name_id, name_id);
    assert_eq!(spawned_metadata.nrows, 100);
    assert_eq!(spawned_metadata.ncols, 50);

    // Verify we can load the saved data using spawned storage
    let loaded_matrix = spawned_storage
        .load_dense("rawinput")
        .await
        .expect("Failed to load dense matrix");

    assert_eq!(loaded_matrix.shape(), (2, 3));

    let loaded_lambdas = spawned_storage
        .load_lambdas()
        .await
        .expect("Failed to load lambdas");

    assert_eq!(loaded_lambdas, lambdas);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lance_storage_spawn_missing_metadata() {
    // Setup: Create a temporary directory without metadata
    let temp_dir = tmp_dir("test_concurrent_storage_instances").await;
    let base_path = temp_dir.as_path().to_str().unwrap().to_string();

    // Attempt to spawn without metadata - should return error
    let result = LanceStorageGraph::spawn(base_path.clone()).await;

    assert!(result.is_err(), "Expected error when metadata is missing");

    let err = result.unwrap_err();
    assert!(
        matches!(err, StorageError::Invalid(_)),
        "Expected StorageError::Invalid, got: {:?}",
        err
    );

    // Verify error message contains relevant information
    if let StorageError::Invalid(msg) = err {
        assert!(
            msg.contains("Metadata does not exist"),
            "Error message should mention missing metadata, got: {}",
            msg
        );
        assert!(
            msg.contains(&base_path),
            "Error message should include the base path, got: {}",
            msg
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lance_storage_spawn_nonexistent_directory() {
    // Try to spawn from a directory that doesn't exist
    let base_path = "/tmp/nonexistent_directory_12345".to_string();

    let result = LanceStorageGraph::spawn(base_path.clone()).await;

    assert!(
        result.is_err(),
        "Expected error when directory doesn't exist"
    );

    let err = result.unwrap_err();
    assert!(
        matches!(err, StorageError::Invalid(_)),
        "Expected StorageError::Invalid, got: {:?}",
        err
    );

    // Verify error message contains relevant information
    if let StorageError::Invalid(msg) = err {
        assert!(
            msg.contains("Metadata does not exist"),
            "Error message should mention missing metadata, got: {}",
            msg
        );
        assert!(
            msg.contains(&base_path),
            "Error message should include the base path, got: {}",
            msg
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lance_storage_spawn_metadata_consistency() {
    // Setup: Create storage with specific metadata
    let temp_dir = tmp_dir("test_concurrent_storage_instances").await;
    let base_path = temp_dir.as_path().to_str().unwrap().to_string();
    let name_id = "consistency_test";

    let storage = LanceStorageGraph::new(base_path.clone(), name_id.to_string());

    let mut metadata = GeneMetadata::seed_metadata(name_id, 200, 75, &storage)
        .await
        .expect("Failed to seed metadata");

    // Add some file info
    metadata = metadata.add_file(
        "test_file",
        FileInfo::new(
            format!("{}_test_file.lance", name_id),
            "dense",
            (200, 75),
            None,
            None,
        )
        .expect("dense is a known filetype"),
    );

    storage
        .save_metadata(&metadata.clone())
        .await
        .expect("Failed to save metadata");

    // Spawn and verify all metadata fields
    let (_spawned_storage, spawned_metadata) = LanceStorageGraph::spawn(base_path.clone())
        .await
        .expect("Failed to spawn");

    assert_eq!(spawned_metadata.name_id, metadata.name_id);
    assert_eq!(spawned_metadata.nrows, metadata.nrows);
    assert_eq!(spawned_metadata.ncols, metadata.ncols);
    assert_eq!(spawned_metadata.base, metadata.base);
    assert_eq!(spawned_metadata.files.len(), metadata.files.len());
}

#[tokio::test(flavor = "multi_thread")]
async fn test_load_lambdas_reads_all_batches() {
    crate::tests::init();
    let name_id = "multi_batch_lambdas";
    let (_, storage, _, _, _) = init_test_builder(name_id).await;

    let md = GeneMetadata::seed_metadata(name_id, 1, 1, &storage.clone())
        .await
        .unwrap();
    let md_path = storage.save_metadata(&md).await.unwrap();

    // 10_000 rows exceeds the scan default batch size (8192), forcing a
    // multi-batch read on load.
    let n = 10_000;
    let lambdas: Vec<f64> = (0..n).map(|i| (i % 100) as f64 / 100.0).collect();
    storage.save_lambdas(&lambdas, &md_path).await.unwrap();

    let loaded = storage.load_lambdas().await.unwrap();
    assert_eq!(
        loaded.len(),
        n,
        "load_lambdas must read all batches, not just the first"
    );
    assert_relative_eq!(loaded[9999], 99.0 / 100.0);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_load_sparse_reads_all_batches() {
    crate::tests::init();
    let name_id = "multi_batch_sparse";
    let (_, storage, _, _, _) = init_test_builder(name_id).await;

    let md = GeneMetadata::seed_metadata(name_id, 1, 1, &storage.clone())
        .await
        .unwrap();
    let md_path = storage.save_metadata(&md).await.unwrap();

    // 10_000 triplets exceed the scan default batch size (8192).
    let n = 10_000;
    let rows: Vec<usize> = (0..n).collect();
    let cols: Vec<usize> = (0..n).map(|i| i % 1000).collect();
    let vals: Vec<f64> = (0..n).map(|i| (i % 7) as f64 + 1.0).collect();
    let matrix = sprs::TriMat::from_triplets((n, 1000), rows, cols, vals).to_csr();

    storage
        .save_sparse("laplacian", &matrix, &md_path)
        .await
        .unwrap();
    let loaded = storage.load_sparse("laplacian").await.unwrap();
    assert_eq!(
        loaded.nnz(),
        n,
        "load_sparse must read all batches, not just the first"
    );
}

#[test]
fn test_path_to_uri_rejects_relative_paths() {
    let err = <LanceStorageGraph as StorageBackend>::path_to_uri(Path::new("relative/no.lance"))
        .unwrap_err();
    assert!(
        matches!(err, StorageError::Invalid(ref msg) if msg.contains("relative")),
        "expected Invalid for relative path, got: {err}"
    );
}

#[test]
fn test_path_to_uri_absolute_missing_path_is_ok() {
    // Windows requires a drive letter for absolute paths and renders file://
    // URIs with the drive component (file:///C:/...).
    let (path, expected) = if cfg!(windows) {
        (
            Path::new(r"C:\tmp\genegraph-definitely-not-yet-written.lance"),
            "file:///C:/tmp/genegraph-definitely-not-yet-written.lance",
        )
    } else {
        (
            Path::new("/tmp/genegraph-definitely-not-yet-written.lance"),
            "file:///tmp/genegraph-definitely-not-yet-written.lance",
        )
    };
    let uri = <LanceStorageGraph as StorageBackend>::path_to_uri(path).unwrap();
    assert_eq!(uri, expected);
}

#[test]
#[cfg(unix)]
fn test_path_to_uri_unresolvable_path_returns_error() {
    // Issue #48: a symlink loop makes canonicalize() fail with ELOOP (not
    // NotFound). The failure must surface as an error instead of being
    // silently replaced by the unresolved path.
    let base = std::env::temp_dir().join(format!("gg_uri_loop_{}", uuid::Uuid::new_v4().simple()));
    std::fs::create_dir_all(&base).unwrap();
    // Self-referential symlink: canonicalize() fails with ELOOP (not NotFound)
    std::os::unix::fs::symlink("loop", base.join("loop")).unwrap();
    let path = base.join("loop");

    let result = <LanceStorageGraph as StorageBackend>::path_to_uri(&path);
    assert!(
        result.is_err(),
        "expected error for unresolvable path, got {:?}",
        result
    );

    std::fs::remove_dir_all(&base).ok();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_from_sparse_batch_dimension_mismatch_returns_error() {
    // Issue #46: schema metadata dimensions that disagree with storage
    // metadata must produce a typed error, not a panic.
    use std::collections::HashMap;
    use std::sync::Arc;

    use arrow::array::{Float64Array, UInt32Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    crate::tests::init();
    let name_id = "sparse_dim_mismatch";
    let base = tmp_dir(name_id).await;
    let storage = LanceStorageGraph::new(base.to_string_lossy().to_string(), name_id.to_string());

    // Batch schema metadata claims a 5x5 matrix...
    let mut schema_metadata = HashMap::new();
    schema_metadata.insert("rows".to_string(), "5".to_string());
    schema_metadata.insert("cols".to_string(), "5".to_string());
    let schema = Schema::new(vec![
        Field::new("row", DataType::UInt32, false),
        Field::new("col", DataType::UInt32, false),
        Field::new("value", DataType::Float64, false),
    ])
    .with_metadata(schema_metadata);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(UInt32Array::from(vec![0u32, 1u32])) as _,
            Arc::new(UInt32Array::from(vec![1u32, 2u32])) as _,
            Arc::new(Float64Array::from(vec![1.0, 2.0])) as _,
        ],
    )
    .unwrap();

    // ...but storage metadata expects 10x10.
    let result = storage.from_sparse_record_batch(batch, 10, 10);
    assert!(
        matches!(result, Err(StorageError::DimensionMismatch { .. })),
        "expected DimensionMismatch, got {:?}",
        result
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_save_index_rejects_values_exceeding_u32_max() {
    // Issue #51: usize values above u32::MAX must not be silently truncated.
    crate::tests::init();
    let name_id = "index_overflow";
    let (_, storage, _, _, _) = init_test_builder(name_id).await;

    let md = GeneMetadata::seed_metadata(name_id, 1, 1, &storage)
        .await
        .unwrap();
    let md_path = storage.save_metadata(&md).await.unwrap();

    let too_big = u32::MAX as usize + 1;
    let err = storage
        .save_index("big_index", &[0, 1, too_big], &md_path)
        .await
        .unwrap_err();

    assert!(
        matches!(err, StorageError::Overflow(ref msg) if msg.contains(&too_big.to_string())),
        "expected Overflow error mentioning {}, got {:?}",
        too_big,
        err
    );

    // The failed save must not register anything in metadata
    let md = storage.load_metadata().await.unwrap();
    assert!(
        !md.files.contains_key("big_index"),
        "failed save must not register the file in metadata"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn test_save_centroid_map_rejects_values_exceeding_u32_max() {
    // Issue #51: same checked-conversion contract as save_index.
    crate::tests::init();
    let name_id = "centroid_overflow";
    let (_, storage, _, _, _) = init_test_builder(name_id).await;

    let md = GeneMetadata::seed_metadata(name_id, 1, 1, &storage)
        .await
        .unwrap();
    let md_path = storage.save_metadata(&md).await.unwrap();

    let too_big = u32::MAX as usize + 7;
    let err = storage
        .save_centroid_map(&[0, 1, too_big], &md_path)
        .await
        .unwrap_err();

    assert!(
        matches!(err, StorageError::Overflow(ref msg) if msg.contains(&too_big.to_string())),
        "expected Overflow error mentioning {}, got {:?}",
        too_big,
        err
    );

    let md = storage.load_metadata().await.unwrap();
    assert!(
        !md.files.contains_key("centroid_map"),
        "failed save must not register the file in metadata"
    );
}

// ===== Issue #50 guard tests: round-trips that must keep working while the
// duplicated save/load pattern is consolidated into a shared helper. =====

#[tokio::test(flavor = "multi_thread")]
async fn test_index_roundtrip() {
    crate::tests::init();
    let name_id = "index_roundtrip";
    let (_, storage, data, _, _) = init_test_builder(name_id).await;
    let (nitems, _) = data.shape();

    let md = GeneMetadata::seed_metadata(name_id, nitems, 1, &storage)
        .await
        .unwrap();
    let md_path = storage.save_metadata(&md).await.unwrap();

    let index: Vec<usize> = (0..nitems).rev().collect();
    storage
        .save_index("ordering", &index, &md_path)
        .await
        .unwrap();

    let loaded = storage.load_index("ordering").await.unwrap();
    assert_eq!(loaded, index);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_centroid_map_roundtrip() {
    crate::tests::init();
    let name_id = "centroid_map_roundtrip";
    let (_, storage, data, _, _) = init_test_builder(name_id).await;
    let (nitems, _) = data.shape();

    let md = GeneMetadata::seed_metadata(name_id, nitems, 1, &storage)
        .await
        .unwrap();
    let md_path = storage.save_metadata(&md).await.unwrap();

    let map: Vec<usize> = (0..nitems).map(|i| i % 5).collect();
    storage.save_centroid_map(&map, &md_path).await.unwrap();

    let loaded = storage.load_centroid_map().await.unwrap();
    assert_eq!(loaded, map);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_item_norms_roundtrip() {
    crate::tests::init();
    let name_id = "item_norms_roundtrip";
    let (_, storage, data, _, norms) = init_test_builder(name_id).await;
    let (nitems, nfeatures) = data.shape();

    let md = GeneMetadata::seed_metadata(name_id, nitems, nfeatures, &storage)
        .await
        .unwrap();
    let md_path = storage.save_metadata(&md).await.unwrap();

    storage.save_item_norms(&norms, &md_path).await.unwrap();

    let loaded = storage.load_item_norms().await.unwrap();
    assert_eq!(loaded.len(), norms.len());
    for (a, b) in norms.iter().zip(loaded.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-10);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_subcentroids_roundtrip() {
    crate::tests::init();
    let name_id = "subcentroids_roundtrip";
    let (_, storage, data, _, _) = init_test_builder(name_id).await;
    let (nitems, nfeatures) = data.shape();

    let md = GeneMetadata::seed_metadata(name_id, nitems, nfeatures, &storage)
        .await
        .unwrap();
    let md_path = storage.save_metadata(&md).await.unwrap();

    let subcentroids = DenseMatrix::new(
        3,
        nfeatures,
        (0..3 * nfeatures).map(|i| (i % 7) as f64 * 0.25).collect(),
        true,
    )
    .unwrap();
    storage
        .save_subcentroids(&subcentroids, &md_path)
        .await
        .unwrap();

    let loaded = storage.load_subcentroids().await.unwrap();
    assert_eq!(loaded.len(), 3);
    assert_eq!(loaded[0].len(), nfeatures);
    for (r, row) in loaded.iter().enumerate() {
        for (c, value) in row.iter().enumerate() {
            assert_relative_eq!(value, subcentroids.get((r, c)), epsilon = 1e-10);
        }
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_subcentroid_lambdas_roundtrip() {
    crate::tests::init();
    let name_id = "subcentroid_lambdas_roundtrip";
    let (_, storage, data, _, _) = init_test_builder(name_id).await;
    let (nitems, nfeatures) = data.shape();

    let md = GeneMetadata::seed_metadata(name_id, nitems, nfeatures, &storage)
        .await
        .unwrap();
    let md_path = storage.save_metadata(&md).await.unwrap();

    let lambdas: Vec<f64> = (0..nitems).map(|i| i as f64 * 0.01).collect();
    storage
        .save_subcentroid_lambdas(&lambdas, &md_path)
        .await
        .unwrap();

    let loaded = storage.load_subcentroid_lambdas().await.unwrap();
    assert_eq!(loaded.len(), lambdas.len());
    for (a, b) in lambdas.iter().zip(loaded.iter()) {
        assert_relative_eq!(a, b, epsilon = 1e-10);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn test_cluster_assignments_roundtrip() {
    crate::tests::init();
    let name_id = "cluster_assignments_roundtrip";
    let (_, storage, data, _, _) = init_test_builder(name_id).await;
    let (nitems, nfeatures) = data.shape();

    let md = GeneMetadata::seed_metadata(name_id, nitems, nfeatures, &storage)
        .await
        .unwrap();
    let md_path = storage.save_metadata(&md).await.unwrap();

    let assignments: Vec<Option<usize>> = (0..nitems)
        .map(|i| if i % 4 == 3 { None } else { Some(i % 5) })
        .collect();
    storage
        .save_cluster_assignments(&assignments, &md_path)
        .await
        .unwrap();

    let loaded = storage.load_cluster_assignments().await.unwrap();
    assert_eq!(loaded, assignments);
}
