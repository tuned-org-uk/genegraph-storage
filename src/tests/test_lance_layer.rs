use crate::lance::LanceStorage;
use crate::metadata::FileInfo;
use crate::metadata::GeneMetadata;
use crate::tests::tmp_dir;
use crate::traits::StorageBackend;

use log::debug;
use std::path::PathBuf;

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
    LanceStorage,
    DenseMatrix<f64>,
    CsMat<f64>,
    Vec<f64>,
) {
    let (nitems, nfeatures) = (150, 300);
    let (dense, adjacency, norms) =
        crate::tests::test_data::make_gaussian_cliques_multi(nitems, 0.3, 5, nfeatures, 42);
    let base = tmp_dir(instance_name).await;
    let storage = LanceStorage::new(
        base.to_string_lossy().to_string(),
        instance_name.to_string(),
    );

    let data =
        DenseMatrix::<f64>::from_iterator(dense.iter().flatten().map(|x| *x), nitems, nfeatures, 0);

    (base, storage, data, adjacency, norms)
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic]
async fn test_no_metadata() {
    crate::tests::init();
    let name = "meta_layout";
    let (_, storage, data, _, _) = init_test_builder(name).await;

    let path = storage.file_path("test");
    storage.save_dense("rawinput", &data, &path).await.unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_metadata_simple() {
    crate::tests::init();
    let name_id = "meta_layout";
    let (_, storage, data, _, _) = init_test_builder(name_id).await;
    let (nitems, nfeatures) = data.shape();

    // Create metadata
    let md = GeneMetadata::seed_metadata(&name_id, nitems, nfeatures, &storage.clone())
        .await
        .unwrap();
    debug!("Saving metadata first to initialize storage directory");
    let md_path: PathBuf = storage.save_metadata(&md).await.unwrap();

    // Assert metadata file was created
    assert!(
        md_path.exists(),
        "Metadata file should exist at {:?}",
        &md_path
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

    // Verify file entries
    assert!(
        loaded_md.files.contains_key("rawinput"),
        "Metadata should contain rawinput entry"
    );

    let rawinput = loaded_md.files.get("rawinput").unwrap();
    assert_eq!(rawinput.filetype, "dense", "File type should be dense");
    assert!(rawinput.filename.contains("rawinput"));
    assert_eq!(rawinput.rows, nitems, "File info rows should match");
    assert_eq!(rawinput.cols, nfeatures, "File info cols should match");
    assert_eq!(
        rawinput.filename, md.files["rawinput"].filename,
        "File info filenames should match {} != {}",
        rawinput.filename, md.files["rawinput"].filename,
    );

    // Verify the files HashMap structure
    assert_eq!(
        loaded_md.files.len(),
        1,
        "Should have exactly one file entry"
    );

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

    // Verify the files object in JSON has the correct structure
    let files_obj = json_value.get("files").unwrap().as_object().unwrap();
    assert!(
        files_obj.contains_key("rawinput"),
        "JSON files object should have 'rawinput' key"
    );

    let rawinput = files_obj.get("rawinput").unwrap().as_object().unwrap();
    assert_eq!(
        rawinput.get("filename").unwrap().as_str().unwrap(),
        md.files["rawinput"].filename,
        "JSON filename should match"
    );
    assert_eq!(
        rawinput.get("filetype").unwrap().as_str().unwrap(),
        "dense",
        "JSON filetype should be 'dense'"
    );
    assert_eq!(
        rawinput.get("rows").unwrap().as_u64().unwrap(),
        nitems as u64,
        "JSON rows should match"
    );
    assert_eq!(
        rawinput.get("cols").unwrap().as_u64().unwrap(),
        nfeatures as u64,
        "JSON cols should match"
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
    debug!("  Rawdata file: {}", md.files["rawinput"].filename);
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic]
async fn test_lance_dense_missing_metadata() {
    crate::tests::init();
    let name = "missing_metadata";
    let (_, storage, data, _, _) = init_test_builder(name).await;

    // Use row-major ordering (true parameter)
    let path = storage.file_path("dense");

    storage
        .save_dense("missing_data", &data, &path)
        .await
        .unwrap();
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lance_dense_roundtrip() {
    crate::tests::init();
    let name = "dense_roundtrip";
    let (_, storage, data, _, _) = init_test_builder(name).await;
    let (nitems, nfeatures) = data.shape();

    // Create expected filename for rawinput
    let expected_filename = format!("{}_rawinput.lance", name);

    // Save metadata FIRST to initialize the storage directory
    let md = GeneMetadata::seed_metadata(name, nitems, nfeatures, &storage.clone())
        .await
        .unwrap()
        .with_dimensions(nitems, nfeatures)
        .add_file(
            "rawinput",
            FileInfo::new(
                expected_filename.clone(),
                "dense",
                (nitems, nfeatures),
                None,
                None,
            ),
        );

    let md_path = storage.save_metadata(&md).await.unwrap();

    storage
        .save_dense("rawinput", &data, &md_path)
        .await
        .unwrap();
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
    let loaded: CsMat<f64> = storage.load_sparse("adjacency").await.unwrap();

    assert_eq!(adjacency.rows(), loaded.rows());
    assert_eq!(adjacency.cols(), loaded.cols());
    assert_eq!(adjacency.nnz(), loaded.nnz());

    assert_eq!(adjacency, loaded);
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lambdas_roundtrip() {
    crate::tests::init();
    let name_id = "lambdas_roundtrip";
    let (_, storage, data, _, norms) = init_test_builder(name_id).await;
    let (nitems, nfeatures) = data.shape();

    // Create metadata
    let md = GeneMetadata::seed_metadata(&name_id, nitems, nfeatures, &storage)
        .await
        .unwrap();
    debug!("Saving metadata first to initialize storage directory");
    let md_path: PathBuf = storage.save_metadata(&md).await.unwrap();

    storage
        .save_lambdas(norms.as_slice(), &md_path)
        .await
        .unwrap();
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
    GeneMetadata::seed_metadata(&name_id, nitems, nfeatures, &storage)
        .await
        .unwrap();
    debug!("Saving metadata first to initialize storage directory");

    let md_path = storage.metadata_path();
    assert!(md_path.exists());

    // load and add files to metadata
    let mut md: GeneMetadata = storage.load_metadata().await.unwrap();

    let mock_info_adj = md.new_fileinfo(
        "adjacency",
        "sparse",
        (nitems, nitems),
        Some(adjacency.nnz()),
        None,
    );

    let mock_info_norms = md.new_fileinfo("norms", "vector", (nitems, 1), None, None);

    md = md.add_file("adjacency", mock_info_adj);
    md = md.add_file("norms", mock_info_norms);
    let md_path = storage.save_metadata(&md).await.unwrap();

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

    GeneMetadata::seed_metadata(&name, nitems, nfeatures, &storage)
        .await
        .unwrap();

    let loaded_md = storage.load_metadata().await.unwrap();

    assert_eq!(loaded_md.name_id, name);
    assert_eq!(loaded_md.nrows, nitems);
    assert_eq!(loaded_md.ncols, nfeatures);

    assert!(loaded_md.files.contains_key("rawinput"));
}

#[tokio::test(flavor = "multi_thread")]
async fn test_concurrent_storage_instances() {
    crate::tests::init();
    let base = tmp_dir("test_concurrent_storage_instances").await;
    let name = "concurrent";

    let storage1 = LanceStorage::new(base.to_string_lossy().to_string(), "instance1".to_string());
    let storage2 = LanceStorage::new(base.to_string_lossy().to_string(), "instance2".to_string());

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
    let md1 = GeneMetadata::seed_metadata(&name, nitems1, nfeatures1, &storage1)
        .await
        .unwrap();
    let md2 = GeneMetadata::seed_metadata(&name, nitems2, nfeatures2, &storage2)
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
    let storage = LanceStorage::new(base_path.clone(), name_id.to_string());

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
    let (spawned_storage, spawned_metadata) = LanceStorage::spawn(base_path.clone())
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
#[should_panic(expected = "Metadata does not exist in this base path")]
async fn test_lance_storage_spawn_missing_metadata() {
    // Setup: Create a temporary directory without metadata
    let temp_dir = tmp_dir("test_concurrent_storage_instances").await;
    let base_path = temp_dir.as_path().to_str().unwrap().to_string();

    // Attempt to spawn without metadata - should panic
    let _result = LanceStorage::spawn(base_path)
        .await
        .expect("Should panic before this");
}

#[tokio::test(flavor = "multi_thread")]
#[should_panic(expected = "Metadata does not exist in this base path")]
async fn test_lance_storage_spawn_nonexistent_directory() {
    // Try to spawn from a directory that doesn't exist
    let base_path = "/tmp/nonexistent_directory_12345".to_string();

    let _result = LanceStorage::spawn(base_path)
        .await
        .expect("Should panic before this");
}

#[tokio::test(flavor = "multi_thread")]
async fn test_lance_storage_spawn_metadata_consistency() {
    // Setup: Create storage with specific metadata
    let temp_dir = tmp_dir("test_concurrent_storage_instances").await;
    let base_path = temp_dir.as_path().to_str().unwrap().to_string();
    let name_id = "consistency_test";

    let storage = LanceStorage::new(base_path.clone(), name_id.to_string());

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
        ),
    );

    storage
        .save_metadata(&metadata.clone())
        .await
        .expect("Failed to save metadata");

    // Spawn and verify all metadata fields
    let (_spawned_storage, spawned_metadata) = LanceStorage::spawn(base_path.clone())
        .await
        .expect("Failed to spawn");

    assert_eq!(spawned_metadata.name_id, metadata.name_id);
    assert_eq!(spawned_metadata.nrows, metadata.nrows);
    assert_eq!(spawned_metadata.ncols, metadata.ncols);
    assert_eq!(spawned_metadata.base, metadata.base);
    assert_eq!(spawned_metadata.files.len(), metadata.files.len());
}
