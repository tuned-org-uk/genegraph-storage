//! #4: ZarrStorage — a `StorageBackend` sibling over one Zarr root.
//!
//! Failing-test-first pins:
//! - kernel metadata registry location: `{root}/.arro/metadata.json`
//! - dataset-ID codec round-trips the Python contract (`--` separator)
//! - root scan discovers nested arrays/groups (v3 `zarr.json` and legacy
//!   `.zarray`/`.zgroup` markers) with shape, dtype, chunks, fill value
//! - trait-fit decisions: dense/vector IO fits Zarr trees; sparse, graph
//!   and RecordBatch-collection IO surface `UnsupportedFiletype`
//!
//! Run with: `cargo test --release --lib -- zarr_storage`.

use std::path::{Path, PathBuf};

use smartcore::linalg::basic::arrays::{Array, Array2};
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::TriMat;

use crate::catalog::{Catalog, LocalRegistry};
use crate::metadata::GeneMetadata;
use crate::traits::backend::StorageBackend;
use crate::traits::metadata::Metadata;
use crate::traits::zarr::{NodeKind, ZarrStorageOps};
use crate::zarr_storage::{ZarrStorage, decode_dataset_id, make_dataset_id};

use super::tmp_dir;

// ---------------------------------------------------------------------------
// helpers
// ---------------------------------------------------------------------------

/// Dense f64 matrix from row slices (row-major input, column-major storage).
fn dense(rows: &[&[f64]]) -> DenseMatrix<f64> {
    let (r, c) = (rows.len(), rows[0].len());
    DenseMatrix::from_iterator(rows.iter().copied().flatten().copied(), r, c, 0)
}

/// Storage seeded with the kernel registry at `{root}/.arro/metadata.json`.
async fn seeded(root: &Path, label: &str, rows: usize, cols: usize) -> ZarrStorage {
    let storage = ZarrStorage::new(root.to_path_buf(), label.to_string()).unwrap();
    GeneMetadata::seed_metadata(label, rows, cols, &storage)
        .await
        .unwrap();
    storage
}

/// Minimal Zarr v3 group directory.
fn write_v3_group(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join("zarr.json"),
        r#"{"zarr_format":3,"node_type":"group","extensions":[]}"#,
    )
    .unwrap();
}

/// Minimal Zarr v2 array marker (`.zarray`).
fn write_v2_array(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(
        dir.join(".zarray"),
        r#"{"zarr_format":2,"shape":[4],"chunks":[4],"dtype":"<f4","compressor":null,"fill_value":null,"filters":null,"order":"C"}"#,
    )
    .unwrap();
}

/// Minimal Zarr v2 group marker (`.zgroup`).
fn write_v2_group(dir: &Path) {
    std::fs::create_dir_all(dir).unwrap();
    std::fs::write(dir.join(".zgroup"), r#"{"zarr_format":2}"#).unwrap();
}

fn by_id<'a>(
    list: &'a [crate::traits::zarr::DatasetSummary],
    id: &str,
) -> &'a crate::traits::zarr::DatasetSummary {
    list.iter()
        .find(|s| s.dataset_id == id)
        .unwrap_or_else(|| panic!("dataset '{id}' not in {list:?}"))
}

// ---------------------------------------------------------------------------
// dataset-ID codec (Python contract, base.py)
// ---------------------------------------------------------------------------

#[test]
fn make_dataset_id_matches_python_contract() {
    assert_eq!(make_dataset_id("main", "cube"), "main--cube");
    assert_eq!(make_dataset_id("main", "sub/array"), "main--sub--array");
    assert_eq!(make_dataset_id("main", "."), "main");
    assert_eq!(make_dataset_id("main", ""), "main");
}

#[test]
fn decode_dataset_id_matches_python_contract() {
    assert_eq!(
        decode_dataset_id("main--cube"),
        ("main".to_string(), "cube".to_string())
    );
    assert_eq!(
        decode_dataset_id("main--sub--array"),
        ("main".to_string(), "sub/array".to_string())
    );
    assert_eq!(
        decode_dataset_id("main"),
        ("main".to_string(), ".".to_string())
    );
}

#[test]
fn dataset_id_roundtrip_recovers_label_and_path() {
    for (label, path) in [("main", "cube"), ("main", "sub/array"), ("main", ".")] {
        let id = make_dataset_id(label, path);
        let want_path = if path == "." { "." } else { path };
        assert_eq!(
            decode_dataset_id(&id),
            (label.to_string(), want_path.to_string())
        );
    }
}

#[tokio::test]
async fn constructor_rejects_invalid_labels() {
    let dir = tmp_dir("zarr_st_bad_label").await;
    let err = ZarrStorage::new(dir.clone(), "ma--in".to_string()).unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "separator in label: {err:?}"
    );
    let err = ZarrStorage::new(dir, String::new()).unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "empty label: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// kernel-owned metadata registry location
// ---------------------------------------------------------------------------

#[tokio::test]
async fn metadata_path_is_root_dot_arro_metadata_json() {
    let dir = tmp_dir("zarr_st_md_path").await;
    let storage = ZarrStorage::new(dir.clone(), "main".to_string()).unwrap();
    assert_eq!(
        storage.metadata_path(),
        dir.join(".arro").join("metadata.json")
    );
}

#[tokio::test]
async fn seeded_registry_lives_outside_user_trees() {
    let root = tmp_dir("zarr_st_seed").await.join("main");
    seeded(&root, "main", 3, 2).await;
    let md_path = root.join(".arro").join("metadata.json");
    assert!(md_path.is_file(), "registry must exist at {md_path:?}");
    let text = std::fs::read_to_string(&md_path).unwrap();
    let value: serde_json::Value = serde_json::from_str(&text).unwrap();
    assert_eq!(value["name_id"], "main");
    assert_eq!(value["nrows"], 3);
    assert_eq!(value["ncols"], 2);
}

#[tokio::test]
async fn exists_reports_the_kernel_registry_only() {
    let root = tmp_dir("zarr_st_exists").await.join("main");
    std::fs::create_dir_all(&root).unwrap();
    assert_eq!(
        ZarrStorage::exists(root.to_string_lossy().as_ref()),
        (false, None),
        "empty root has no registry"
    );
    let storage = seeded(&root, "main", 1, 1).await;
    let (ok, path) = ZarrStorage::exists(root.to_string_lossy().as_ref());
    assert!(ok);
    assert_eq!(path, Some(storage.metadata_path()));
}

#[tokio::test]
async fn save_dense_requires_seeded_metadata() {
    let root = tmp_dir("zarr_st_unseeded").await.join("main");
    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let m = dense(&[&[1.0, 2.0]]);
    let err = storage
        .save_dense("matrix", &m, &storage.metadata_path())
        .await
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
}

// ---------------------------------------------------------------------------
// dense save/load
// ---------------------------------------------------------------------------

#[tokio::test]
async fn dense_roundtrip_writes_zarr_and_registers_entry() {
    let root = tmp_dir("zarr_st_dense_rt").await.join("main");
    let storage = seeded(&root, "main", 3, 2).await;
    let m = dense(&[&[0.1, 0.4], &[0.5, 0.2], &[0.03, 0.8]]);
    storage
        .save_dense("sub/array", &m, &storage.metadata_path())
        .await
        .unwrap();

    // Zarr v3 array node exists at the rel path.
    assert!(root.join("sub").join("array").join("zarr.json").is_file());

    // Registry entry records the artifact under its rel-path key.
    let md = storage.load_metadata().await.unwrap();
    let info = md
        .files
        .get("sub/array")
        .unwrap_or_else(|| panic!("no entry: {:?}", md.files));
    assert_eq!(info.filetype, "dense");
    assert_eq!((info.rows, info.cols), (3, 2));

    let loaded = storage.load_dense("sub/array").await.unwrap();
    assert_eq!(loaded.shape(), (3, 2));
    for (r, row) in [[0.1, 0.4], [0.5, 0.2], [0.03, 0.8]].iter().enumerate() {
        for (c, want) in row.iter().enumerate() {
            assert_eq!(*loaded.get((r, c)), *want, "cell ({r},{c})");
        }
    }
}

#[tokio::test]
async fn load_dense_missing_key_is_invalid() {
    let root = tmp_dir("zarr_st_dense_missing").await.join("main");
    let storage = seeded(&root, "main", 2, 2).await;
    let err = storage.load_dense("nope").await.unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
}

#[tokio::test]
async fn save_dense_rejects_overwrite() {
    let root = tmp_dir("zarr_st_dense_over").await.join("main");
    let storage = seeded(&root, "main", 2, 2).await;
    let m = dense(&[&[1.0, 2.0], &[3.0, 4.0]]);
    storage
        .save_dense("matrix", &m, &storage.metadata_path())
        .await
        .unwrap();
    let err = storage
        .save_dense("matrix", &m, &storage.metadata_path())
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::InvalidState(_)),
        "{err:?}"
    );
}

#[tokio::test]
async fn save_rejects_escaping_or_empty_keys() {
    let root = tmp_dir("zarr_st_key_guard").await.join("main");
    let storage = seeded(&root, "main", 1, 1).await;
    let m = dense(&[&[1.0]]);
    for bad in ["", "..", "../escape", "a/../b", "/abs"] {
        let err = storage
            .save_dense(bad, &m, &storage.metadata_path())
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::StorageError::Invalid(_)),
            "key '{bad}': {err:?}"
        );
    }
}

#[tokio::test]
async fn dense_file_io_roundtrips_zarr_paths() {
    let dir = tmp_dir("zarr_st_file_io").await;
    let storage = ZarrStorage::new(dir.clone(), "main".to_string()).unwrap();
    let path = dir.join("fresh").join("array");
    let m = dense(&[&[1.5, 2.5], &[3.5, 4.5], &[5.5, 6.5]]);
    ZarrStorage::save_dense_to_file(&m, &path).await.unwrap();
    assert!(path.join("zarr.json").is_file());

    let loaded = storage.load_dense_from_file(&path).await.unwrap();
    assert_eq!(loaded.shape(), (3, 2));
    for (r, row) in [[1.5, 2.5], [3.5, 4.5], [5.5, 6.5]].iter().enumerate() {
        for (c, want) in row.iter().enumerate() {
            assert_eq!(*loaded.get((r, c)), *want, "cell ({r},{c})");
        }
    }

    // Non-Zarr paths surface UnsupportedFormat, never guessed decoders.
    let plain = dir.join("plain");
    std::fs::create_dir_all(&plain).unwrap();
    let err = storage.load_dense_from_file(&plain).await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::UnsupportedFormat(_)),
        "{err:?}"
    );
}

// ---------------------------------------------------------------------------
// vector / lambdas / index IO
// ---------------------------------------------------------------------------

#[tokio::test]
async fn vector_and_lambdas_roundtrip_as_1d_arrays() {
    let root = tmp_dir("zarr_st_vec_rt").await.join("main");
    let storage = seeded(&root, "main", 3, 1).await;
    let md_path = storage.metadata_path();

    let norms = vec![0.5f64, 1.25, 2.0];
    storage
        .save_vector("norms", &norms, &md_path)
        .await
        .unwrap();
    assert!(root.join("norms").join("zarr.json").is_file());
    assert_eq!(storage.load_vector("norms").await.unwrap(), norms);

    let lambdas = vec![3.5f64, -1.0, 0.0];
    storage.save_lambdas(&lambdas, &md_path).await.unwrap();
    assert_eq!(storage.load_lambdas().await.unwrap(), lambdas);
}

#[tokio::test]
async fn index_roundtrips_and_overflows_typefully() {
    let root = tmp_dir("zarr_st_idx_rt").await.join("main");
    let storage = seeded(&root, "main", 3, 1).await;
    let md_path = storage.metadata_path();

    storage
        .save_index("assign", &[7usize, 0, 42], &md_path)
        .await
        .unwrap();
    assert_eq!(
        storage.load_index("assign").await.unwrap(),
        vec![7usize, 0, 42]
    );

    let huge = (i64::MAX as usize) + 1;
    let err = storage
        .save_index("huge", &[huge], &md_path)
        .await
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::Overflow(_)), "{err:?}");
}

// ---------------------------------------------------------------------------
// trait-fit: methods that do not fit Zarr trees
// ---------------------------------------------------------------------------

#[tokio::test]
async fn sparse_matrices_are_unsupported_filetype() {
    let root = tmp_dir("zarr_st_sparse").await.join("main");
    let storage = seeded(&root, "main", 2, 2).await;
    let mut tm = TriMat::new((2, 2));
    tm.add_triplet(0, 0, 1.0);
    let csr = tm.to_csr();

    let err = storage
        .save_sparse("lap", &csr, &storage.metadata_path())
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::UnsupportedFiletype(_)),
        "{err:?}"
    );

    let err = storage.load_sparse("lap").await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::UnsupportedFiletype(_)),
        "{err:?}"
    );
}

#[tokio::test]
async fn graph_collections_are_unsupported_filetype() {
    let root = tmp_dir("zarr_st_graph").await.join("main");
    let storage = seeded(&root, "main", 2, 2).await;
    let err = storage
        .save_graph(
            "g",
            &[crate::graph::GraphEdge::weighted(0, 1, 0.5)],
            &storage.metadata_path(),
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::UnsupportedFiletype(_)),
        "{err:?}"
    );
    let err = storage.load_graph("g").await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::UnsupportedFiletype(_)),
        "{err:?}"
    );
}

#[tokio::test]
async fn vector_space_collections_are_unsupported_filetype() {
    let root = tmp_dir("zarr_st_vectors").await.join("main");
    let storage = seeded(&root, "main", 2, 2).await;
    let m = dense(&[&[1.0, 2.0], &[3.0, 4.0]]);
    let batch = storage.to_dense_record_batch(&m).unwrap();

    let err = storage
        .save_vectors("v", &batch, &storage.metadata_path())
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::UnsupportedFiletype(_)),
        "{err:?}"
    );
    let err = storage.load_vectors("v").await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::UnsupportedFiletype(_)),
        "{err:?}"
    );

    let err = storage.collection_schema("v").await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::UnsupportedFiletype(_)),
        "{err:?}"
    );

    let err = storage
        .save_scalars_to_path(&root.join("x"), &[1.0])
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::UnsupportedFiletype(_)),
        "{err:?}"
    );
}

// ---------------------------------------------------------------------------
// root scan (list_datasets)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn scan_finds_nested_arrays_under_a_group_root() {
    let root = tmp_dir("zarr_st_scan_group").await.join("main");
    write_v3_group(&root);
    write_v3_group(&root.join("sub"));
    let values: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
    crate::zzarr::write_array(
        &root.join("leaf"),
        &[4],
        &[4],
        &(0..4).map(|i| i as f64).collect::<Vec<_>>(),
        true,
    )
    .unwrap();
    crate::zzarr::write_array(
        &root.join("sub").join("cube"),
        &[2, 3],
        &[2, 3],
        &values,
        true,
    )
    .unwrap();

    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let all = storage.list_datasets().await.unwrap();

    // DFS order: the group, then its arrays, then each sub-group in turn.
    assert_eq!(
        all.iter()
            .map(|s| s.dataset_id.as_str())
            .collect::<Vec<_>>(),
        vec!["main", "main--leaf", "main--sub", "main--sub--cube"]
    );

    let root_group = by_id(&all, "main");
    assert_eq!(root_group.kind, NodeKind::Group);
    assert_eq!(root_group.path, ".");
    assert_eq!(root_group.shape, Vec::<u64>::new());
    assert_eq!(root_group.extra["n_arrays"], serde_json::json!(1));
    assert_eq!(root_group.extra["n_groups"], serde_json::json!(1));

    let cube = by_id(&all, "main--sub--cube");
    assert_eq!(cube.kind, NodeKind::Array);
    assert_eq!(cube.root, "main");
    assert_eq!(cube.path, "sub/cube");
    assert_eq!(cube.shape, vec![2, 3]);
    assert_eq!(cube.dtype, "float32");
    assert_eq!(cube.chunks, Some(vec![2, 3]));
    assert_eq!(cube.fill_value, Some(serde_json::json!(0.0)));
}

#[tokio::test]
async fn scan_finds_direct_children_of_a_plain_root() {
    let root = tmp_dir("zarr_st_scan_plain").await.join("main");
    crate::zzarr::write_array(
        &root.join("matrix"),
        &[3, 2],
        &[3, 2],
        &(0..6).map(|i| i as f64).collect::<Vec<_>>(),
        true,
    )
    .unwrap();
    write_v3_group(&root.join("tree"));
    crate::zzarr::write_array(
        &root.join("tree").join("inner"),
        &[5],
        &[5],
        &[1.0f32, 2.0, 3.0, 4.0, 5.0],
        true,
    )
    .unwrap();
    std::fs::create_dir_all(root.join("notes")).unwrap();
    std::fs::write(root.join("notes").join("readme.txt"), "not a zarr node").unwrap();

    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let all = storage.list_datasets().await.unwrap();

    assert_eq!(
        all.iter()
            .map(|s| s.dataset_id.as_str())
            .collect::<Vec<_>>(),
        vec!["main--matrix", "main--tree", "main--tree--inner"]
    );
    let matrix = by_id(&all, "main--matrix");
    assert_eq!(matrix.kind, NodeKind::Array);
    assert_eq!(matrix.shape, vec![3, 2]);
    assert_eq!(matrix.dtype, "float64");
}

#[tokio::test]
async fn scan_reads_v2_markers() {
    let root = tmp_dir("zarr_st_scan_v2").await.join("main");
    write_v2_array(&root.join("legacy"));
    write_v2_group(&root.join("oldgrp"));

    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let all = storage.list_datasets().await.unwrap();

    let legacy = by_id(&all, "main--legacy");
    assert_eq!(legacy.kind, NodeKind::Array);
    assert_eq!(legacy.shape, vec![4]);
    assert_eq!(legacy.dtype, "<f4");
    assert_eq!(legacy.chunks, Some(vec![4]));
    assert_eq!(legacy.fill_value, None);

    assert_eq!(by_id(&all, "main--oldgrp").kind, NodeKind::Group);
}

#[tokio::test]
async fn scan_never_lists_the_kernel_registry_dir() {
    // Plain root.
    let root = tmp_dir("zarr_st_scan_arro_plain").await.join("main");
    crate::zzarr::write_array(&root.join("matrix"), &[2], &[2], &[1.0f64, 2.0], true).unwrap();
    let storage = seeded(&root, "main", 2, 1).await;
    let all = storage.list_datasets().await.unwrap();
    assert_eq!(
        all.iter()
            .map(|s| s.dataset_id.as_str())
            .collect::<Vec<_>>(),
        vec!["main--matrix"]
    );

    // Root that is itself a Zarr group.
    let root = tmp_dir("zarr_st_scan_arro_group").await.join("main");
    write_v3_group(&root);
    crate::zzarr::write_array(&root.join("matrix"), &[2], &[2], &[1.0f64, 2.0], true).unwrap();
    let storage = seeded(&root, "main", 2, 1).await;
    let all = storage.list_datasets().await.unwrap();
    assert_eq!(
        all.iter()
            .map(|s| s.dataset_id.as_str())
            .collect::<Vec<_>>(),
        vec!["main", "main--matrix"]
    );
}

#[tokio::test]
async fn dataset_keys_must_not_target_the_kernel_registry_dir() {
    // The registry dir is kernel-only: `.arro` is reserved for
    // `metadata.json` and the rendezvous locks. No resolution — write or
    // read — may land inside it (sibling of scan_never_lists above).
    let root = tmp_dir("zarr_st_arro_key").await.join("main");
    let storage = seeded(&root, "main", 2, 1).await;
    let m = dense(&[&[1.0, 2.0]]);

    let err = storage
        .save_dense(".arro", &m, &storage.metadata_path())
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "registry-dir key rejected: {err:?}"
    );
    let err = storage
        .save_dense(".arro/locks", &m, &storage.metadata_path())
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "locks subdir key rejected: {err:?}"
    );
    let err = storage.open("main--.arro").await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "registry-dir id rejected: {err:?}"
    );
    // The registry stays pristine: no node markers written inside `.arro`.
    assert!(root.join(".arro").join("metadata.json").is_file());
    assert!(!root.join(".arro").join("zarr.json").exists());
}

#[tokio::test]
async fn missing_root_scans_to_empty() {
    let root = tmp_dir("zarr_st_scan_missing").await.join("absent");
    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    assert!(storage.list_datasets().await.unwrap().is_empty());
}

// ---------------------------------------------------------------------------
// open + summarize (child trait)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn open_returns_a_readable_v3_array() {
    let root = tmp_dir("zarr_st_open").await.join("main");
    write_v3_group(&root);
    let values: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
    crate::zzarr::write_array(
        &root.join("sub").join("cube"),
        &[2, 3],
        &[2, 3],
        &values,
        true,
    )
    .unwrap();

    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let arr = storage.open("main--sub--cube").await.unwrap();
    assert_eq!(arr.shape(), vec![2, 3]);
    assert_eq!(arr.read_all::<f32>().unwrap(), values);

    // A group id is not an array.
    let err = storage.open("main").await.unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
    // Unknown id.
    let err = storage.open("main--nope").await.unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
    // Foreign label.
    let err = storage.open("other--sub--cube").await.unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
    // Traversal is rejected, not joined.
    let err = storage.open("main--..").await.unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
}

#[tokio::test]
async fn open_rejects_v2_arrays() {
    let root = tmp_dir("zarr_st_open_v2").await.join("main");
    write_v2_array(&root.join("legacy"));
    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let err = storage.open("main--legacy").await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::UnsupportedFormat(_)),
        "{err:?}"
    );
}

#[tokio::test]
async fn summarize_matches_the_scan_for_a_single_node() {
    let root = tmp_dir("zarr_st_summarize").await.join("main");
    write_v3_group(&root);
    write_v3_group(&root.join("sub"));
    let values: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
    crate::zzarr::write_array(
        &root.join("sub").join("cube"),
        &[2, 3],
        &[2, 3],
        &values,
        true,
    )
    .unwrap();

    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();
    let all = storage.list_datasets().await.unwrap();
    let scanned = by_id(&all, "main--sub--cube").clone();

    let summarized = storage
        .summarize("main--sub--cube", &root.join("sub").join("cube"))
        .await
        .unwrap();
    assert_eq!(summarized, scanned);
}

#[tokio::test]
async fn summarize_rejects_missing_nodes_and_groups() {
    let root = tmp_dir("zarr_st_summarize_bad").await.join("main");
    write_v3_group(&root);
    write_v3_group(&root.join("sub"));
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();

    let err = storage
        .summarize("main--sub", &root.join("sub"))
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "group: {err:?}"
    );

    let err = storage
        .summarize("main--nope", &root.join("nope"))
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "missing: {err:?}"
    );
}

// ---------------------------------------------------------------------------
// catalog surfacing
// ---------------------------------------------------------------------------

#[tokio::test]
async fn zarr_datasets_surface_through_local_registry() {
    let root = tmp_dir("zarr_st_catalog").await.join("main");
    let storage = seeded(&root, "main", 3, 2).await;
    let m = dense(&[&[0.1, 0.4], &[0.5, 0.2], &[0.03, 0.8]]);
    storage
        .save_dense("sub/array", &m, &storage.metadata_path())
        .await
        .unwrap();

    let md = storage.load_metadata().await.unwrap();
    let registry = LocalRegistry::new(md, root.clone());
    assert!(registry.table_exists("sub/array").unwrap());
    let desc = registry.describe_table("sub/array").unwrap();
    assert_eq!(desc.kind, crate::metadata::CollectionKind::VectorSpace);
    assert_eq!(desc.base_location, root.join("sub").join("array"));
    assert_eq!(desc.properties["rows"], "3");
    assert_eq!(desc.properties["cols"], "2");
}

// ---------------------------------------------------------------------------
// #5: vector write paths — append
// ---------------------------------------------------------------------------

/// f32 arange matrix at `<root>/<name>` (chunks `[chunk_rows, cols]`) — the
/// Python test_vectors_append/overwrite fixture shape.
fn write_arange_f32(dir: &Path, rows: u64, cols: u64, chunk_rows: u64) {
    let values: Vec<f32> = (0..rows * cols).map(|i| i as f32).collect();
    crate::zzarr::write_array(dir, &[rows, cols], &[chunk_rows, cols], &values, true).unwrap();
}

/// Dense (m, d) matrix filled with `value`.
fn filled(m: usize, d: usize, value: f64) -> DenseMatrix<f64> {
    DenseMatrix::from_iterator((0..m * d).map(|_| value), m, d, 0)
}

fn read_rows_f32(dir: &Path, from: u64, to: u64, cols: u64) -> Vec<f32> {
    let arr = crate::zzarr::open(dir).unwrap();
    arr.read_subset::<f32>(&[from..to, 0..cols]).unwrap()
}

fn read_rows_f64(dir: &Path, from: u64, to: u64, cols: u64) -> Vec<f64> {
    let arr = crate::zzarr::open(dir).unwrap();
    arr.read_subset::<f64>(&[from..to, 0..cols]).unwrap()
}

#[tokio::test]
async fn append_returns_start_row_and_new_shape_and_writes_rows() {
    let root = tmp_dir("zarr_app_basic").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();

    let rows: Vec<Vec<f64>> = (0..7)
        .map(|i| (0..4).map(|j| 100.0 + (i * 4 + j) as f64).collect())
        .collect();
    let refs: Vec<&[f64]> = rows.iter().map(|r| r.as_slice()).collect();
    let (start, new_n) = storage
        .append_vectors("main--matrix", &dense(&refs))
        .await
        .unwrap();

    assert_eq!(start, 50);
    assert_eq!(new_n, 57);
    let got = read_rows_f32(&root.join("matrix"), 50, 57, 4);
    for (i, row) in rows.iter().enumerate() {
        for (j, want) in row.iter().enumerate() {
            assert!(
                (got[i * 4 + j] as f64 - want).abs() < 1e-6,
                "row {i} col {j}"
            );
        }
    }
    // Existing rows are untouched.
    assert_eq!(
        read_rows_f32(&root.join("matrix"), 0, 1, 4),
        vec![0.0f32, 1.0, 2.0, 3.0]
    );
}

#[tokio::test]
async fn append_start_row_advances_across_sequential_appends() {
    let root = tmp_dir("zarr_app_seq").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();

    let r1 = storage
        .append_vectors("main--matrix", &filled(10, 4, 1.0))
        .await
        .unwrap();
    assert_eq!(r1, (50, 60));
    let r2 = storage
        .append_vectors("main--matrix", &filled(5, 4, 2.0))
        .await
        .unwrap();
    assert_eq!(r2, (60, 65));
}

#[tokio::test]
async fn append_updates_registered_shape() {
    let root = tmp_dir("zarr_app_registry").await.join("main");
    let storage = seeded(&root, "main", 50, 4).await;
    let m = filled(50, 4, 0.0);
    storage
        .save_dense("matrix", &m, &storage.metadata_path())
        .await
        .unwrap();
    storage
        .append_vectors("main--matrix", &filled(3, 4, 1.0))
        .await
        .unwrap();

    let md = storage.load_metadata().await.unwrap();
    let info = md.files.get("matrix").unwrap();
    assert_eq!((info.rows, info.cols), (53, 4));
}

#[tokio::test]
async fn append_works_on_unregistered_user_trees() {
    // Uploaded trees carry no kernel registry: append is filesystem-first.
    let root = tmp_dir("zarr_app_unreg").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();
    let (start, new_n) = storage
        .append_vectors("main--matrix", &filled(2, 4, 9.0))
        .await
        .unwrap();
    assert_eq!((start, new_n), (50, 52));
    assert!(!root.join(".arro").join("metadata.json").is_file());
}

#[tokio::test]
async fn append_dim_mismatch_surfaces_dimension_mismatch() {
    let root = tmp_dir("zarr_app_dim").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let err = storage
        .append_vectors("main--matrix", &filled(1, 7, 0.0))
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::DimensionMismatch { .. }),
        "{err:?}"
    );
    assert!(err.to_string().contains("Dimension mismatch"));
}

#[tokio::test]
async fn append_empty_batch_is_invalid() {
    let root = tmp_dir("zarr_app_empty").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let err = storage
        .append_vectors("main--matrix", &filled(0, 4, 0.0))
        .await
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
}

#[tokio::test]
async fn append_to_non_float_dtype_is_rejected() {
    // f64 vectors into an i64 array is not same_kind: dtype mismatch.
    let root = tmp_dir("zarr_app_dtype").await.join("main");
    crate::zzarr::write_array(&root.join("ints"), &[4, 2], &[4, 2], &[0i64; 8], true).unwrap();
    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let err = storage
        .append_vectors("main--ints", &filled(1, 2, 1.0))
        .await
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
    assert!(err.to_string().contains("dtype mismatch"));
}

#[tokio::test]
async fn append_missing_dataset_is_invalid() {
    let root = tmp_dir("zarr_app_missing").await.join("main");
    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let err = storage
        .append_vectors("main--nope", &filled(1, 4, 0.0))
        .await
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
}

#[tokio::test]
async fn append_rejects_non_2d_target() {
    let root = tmp_dir("zarr_app_ndim").await.join("main");
    crate::zzarr::write_array(&root.join("flat"), &[10], &[10], &[0.0f32; 10], true).unwrap();
    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let err = storage
        .append_vectors("main--flat", &filled(1, 4, 0.0))
        .await
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
    assert!(err.to_string().contains("not a 2-D array"));
}

#[tokio::test]
async fn append_concurrent_ranges_partition_without_overlap() {
    // Direct port of test_append_concurrent_start_rows_no_overlap: three
    // concurrent appends must return disjoint, contiguous start ranges.
    let root = tmp_dir("zarr_app_conc").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();

    let sizes = [4usize, 7, 3];
    let handles: Vec<_> = sizes
        .iter()
        .map(|&size| {
            let storage = storage.clone();
            tokio::spawn(async move {
                storage
                    .append_vectors("main--matrix", &filled(size, 4, size as f64))
                    .await
                    .unwrap()
            })
        })
        .collect();
    let mut results: Vec<(usize, usize)> = Vec::new();
    for h in handles {
        results.push(h.await.unwrap());
    }
    results.sort();

    assert_eq!(results[0].0, 50, "first range starts at old_n");
    for i in 0..results.len() - 1 {
        assert_eq!(results[i].1, results[i + 1].0, "gap between {results:?}");
    }
    let total: usize = sizes.iter().sum();
    assert_eq!(results[2].1, 50 + total);
    assert_eq!(
        storage
            .summarize("main--matrix", &root.join("matrix"))
            .await
            .unwrap()
            .shape,
        vec![64, 4]
    );
}

#[tokio::test]
async fn append_to_different_datasets_proceeds_independently() {
    let root = tmp_dir("zarr_app_par").await.join("main");
    write_arange_f32(&root.join("a"), 10, 4, 4);
    write_arange_f32(&root.join("other"), 20, 4, 4);
    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();

    let m1 = filled(2, 4, 1.0);
    let m2 = filled(3, 4, 2.0);
    let (r1, r2) = tokio::join!(
        storage.append_vectors("main--a", &m1),
        storage.append_vectors("main--other", &m2),
    );
    assert_eq!(r1.unwrap(), (10, 12));
    assert_eq!(r2.unwrap(), (20, 23));
}

// ---------------------------------------------------------------------------
// #5: vector write paths — overwrite
// ---------------------------------------------------------------------------

#[tokio::test]
async fn overwrite_single_row_updates_values() {
    let root = tmp_dir("zarr_ovw_single").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();

    let n = storage
        .overwrite_vectors(
            "main--matrix",
            &[crate::traits::zarr::RowUpdate {
                row_index: 10,
                vector: vec![99.0, 98.0, 97.0, 96.0],
            }],
        )
        .await
        .unwrap();
    assert_eq!(n, 1);
    assert_eq!(
        read_rows_f32(&root.join("matrix"), 10, 11, 4),
        vec![99.0f32, 98.0, 97.0, 96.0]
    );
}

#[tokio::test]
async fn overwrite_multiple_rows_keeps_shape() {
    let root = tmp_dir("zarr_ovw_multi").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();

    let n = storage
        .overwrite_vectors(
            "main--matrix",
            &[
                crate::traits::zarr::RowUpdate {
                    row_index: 0,
                    vector: vec![1.0, 2.0, 3.0, 4.0],
                },
                crate::traits::zarr::RowUpdate {
                    row_index: 25,
                    vector: vec![5.0, 6.0, 7.0, 8.0],
                },
                crate::traits::zarr::RowUpdate {
                    row_index: 49,
                    vector: vec![9.0, 10.0, 11.0, 12.0],
                },
            ],
        )
        .await
        .unwrap();
    assert_eq!(n, 3);
    assert_eq!(
        read_rows_f32(&root.join("matrix"), 25, 26, 4),
        vec![5.0f32, 6.0, 7.0, 8.0]
    );
    // Shape unchanged.
    let arr = crate::zzarr::open(&root.join("matrix")).unwrap();
    assert_eq!(arr.shape(), vec![50, 4]);
}

#[tokio::test]
async fn overwrite_out_of_bounds_rejects_whole_batch_without_partial_write() {
    let root = tmp_dir("zarr_ovw_bounds").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();

    let err = storage
        .overwrite_vectors(
            "main--matrix",
            &[
                crate::traits::zarr::RowUpdate {
                    row_index: 0,
                    vector: vec![99.0; 4],
                },
                crate::traits::zarr::RowUpdate {
                    row_index: 50,
                    vector: vec![1.0; 4],
                },
            ],
        )
        .await
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
    assert!(err.to_string().contains("out of bounds"));
    // Validate-all-then-write: row 0 must be unchanged.
    assert_eq!(
        read_rows_f32(&root.join("matrix"), 0, 1, 4),
        vec![0.0f32, 1.0, 2.0, 3.0]
    );
}

#[tokio::test]
async fn overwrite_dim_mismatch_rejects_whole_batch() {
    let root = tmp_dir("zarr_ovw_dim").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();
    let err = storage
        .overwrite_vectors(
            "main--matrix",
            &[
                crate::traits::zarr::RowUpdate {
                    row_index: 0,
                    vector: vec![1.0; 4],
                },
                crate::traits::zarr::RowUpdate {
                    row_index: 1,
                    vector: vec![1.0, 2.0, 3.0],
                },
            ],
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::DimensionMismatch { .. }),
        "{err:?}"
    );
    // Row 0 unchanged.
    assert_eq!(
        read_rows_f32(&root.join("matrix"), 0, 1, 4),
        vec![0.0f32, 1.0, 2.0, 3.0]
    );
}

#[tokio::test]
async fn overwrite_empty_updates_is_invalid() {
    let root = tmp_dir("zarr_ovw_empty").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let err = storage
        .overwrite_vectors("main--matrix", &[])
        .await
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
}

#[tokio::test]
async fn overwrite_duplicate_row_index_last_wins() {
    let root = tmp_dir("zarr_ovw_dup").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();

    let n = storage
        .overwrite_vectors(
            "main--matrix",
            &[
                crate::traits::zarr::RowUpdate {
                    row_index: 7,
                    vector: vec![1.0; 4],
                },
                crate::traits::zarr::RowUpdate {
                    row_index: 7,
                    vector: vec![2.0; 4],
                },
            ],
        )
        .await
        .unwrap();
    assert_eq!(n, 2);
    assert_eq!(
        read_rows_f32(&root.join("matrix"), 7, 8, 4),
        vec![2.0f32; 4]
    );
}

#[tokio::test]
async fn overwrite_missing_dataset_is_invalid() {
    let root = tmp_dir("zarr_ovw_missing").await.join("main");
    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let err = storage
        .overwrite_vectors(
            "main--nope",
            &[crate::traits::zarr::RowUpdate {
                row_index: 0,
                vector: vec![1.0; 4],
            }],
        )
        .await
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
}

// ---------------------------------------------------------------------------
// #5 review follow-ups (post-close findings)
// ---------------------------------------------------------------------------

#[tokio::test]
async fn append_succeeds_when_registry_refresh_fails() {
    // Review finding 1: the registry refresh is best-effort in BOTH
    // directions — a save failure after a successful tail write must not
    // surface as an append failure, or a retrying client appends twice.
    let root = tmp_dir("zarr_app_reg_fail").await.join("main");
    let storage = seeded(&root, "main", 3, 2).await;
    let m = filled(3, 2, 1.0);
    storage
        .save_dense("m", &m, &storage.metadata_path())
        .await
        .unwrap();

    // Make the registry unwritable: the tail write is already on disk.
    let arro = root.join(".arro");
    let mut perms = std::fs::metadata(&arro).unwrap().permissions();
    use std::os::unix::fs::PermissionsExt;
    let original = perms.mode();
    perms.set_mode(0o555);
    std::fs::set_permissions(&arro, perms).unwrap();

    let result = storage.append_vectors("main--m", &filled(2, 2, 9.0)).await;
    // Restore before the tempdir drops (cleanup needs write permission).
    let mut perms = std::fs::metadata(&arro).unwrap().permissions();
    perms.set_mode(original);
    std::fs::set_permissions(&arro, perms).unwrap();

    let (start, new_n) = result.unwrap();
    assert_eq!(
        (start, new_n),
        (3, 5),
        "append must succeed despite the registry failure"
    );
    assert_eq!(read_rows_f64(&root.join("m"), 3, 5, 2), vec![9.0f64; 4]);
}

#[tokio::test]
async fn save_dense_waits_for_the_dataset_mailbox() {
    // Review finding 4: creation (save_dense/save_vector/save_index) must
    // serialize through the same per-dataset mailbox as append/overwrite;
    // otherwise the InvalidState overwrite rejection is not guaranteed
    // under concurrency.
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::time::{Duration, Instant};

    let root = tmp_dir("zarr_save_mailbox").await.join("main");
    let storage = seeded(&root, "main", 1, 1).await;
    let path = root.join("held");

    let entered = Arc::new(AtomicBool::new(false));
    let flag = entered.clone();
    let holder = path.clone();
    let handle = std::thread::spawn(move || {
        crate::commit::with_dataset_write_lock(&holder, || {
            flag.store(true, Ordering::SeqCst);
            std::thread::sleep(Duration::from_millis(400));
            Ok(())
        })
    });
    while !entered.load(Ordering::SeqCst) {
        std::thread::sleep(Duration::from_millis(5));
    }

    let start = Instant::now();
    storage
        .save_dense("held", &filled(2, 2, 1.0), &storage.metadata_path())
        .await
        .unwrap();
    let elapsed = start.elapsed();
    handle.join().unwrap().unwrap();
    assert!(
        elapsed >= Duration::from_millis(300),
        "save_dense must wait for the dataset mailbox; waited {elapsed:?}"
    );
}

#[tokio::test]
async fn save_rejects_separator_in_keys() {
    // Review finding 5: a segment containing `--` would encode into a
    // dataset ID that decodes to a different path — reject it so the codec
    // stays a bijection (labels are already guarded).
    let root = tmp_dir("zarr_key_sep").await.join("main");
    let storage = seeded(&root, "main", 1, 1).await;
    let err = storage
        .save_dense("my--array", &filled(1, 1, 1.0), &storage.metadata_path())
        .await
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
    let err = storage.load_vector("a--b").await.unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
}

#[tokio::test]
async fn scan_rejects_malformed_chunk_shape() {
    // Review minor: chunk-shape entries that are not u64 must surface
    // UnsupportedFormat, never be silently dropped.
    let root = tmp_dir("zarr_scan_badchunks").await.join("main");
    std::fs::create_dir_all(root.join("bad")).unwrap();
    std::fs::write(
        root.join("bad").join("zarr.json"),
        r#"{"zarr_format":3,"node_type":"array","shape":[4],"data_type":"float32","chunk_grid":{"configuration":{"chunk_shape":["4"]}},"fill_value":0}"#,
    )
    .unwrap();
    let storage = ZarrStorage::new(root, "main".to_string()).unwrap();
    let err = storage.list_datasets().await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::UnsupportedFormat(_)),
        "{err:?}"
    );
}

// ---------------------------------------------------------------------------
// #5 concurrency: cross-process rendezvous locks (genefold-vd patterns)
// ---------------------------------------------------------------------------

/// Rendezvous lock-file path of `dataset_id` under `root`.
fn lock_file_of(root: &Path, dataset_id: &str) -> PathBuf {
    root.join(".arro")
        .join("locks")
        .join(format!("{dataset_id}.lock"))
}

/// Holds the rendezvous flock as a foreign writer would (own file
/// description, so it excludes the kernel's flock until dropped).
#[cfg(unix)]
fn hold_flock(path: &Path) -> std::fs::File {
    use std::os::unix::io::AsRawFd;

    std::fs::create_dir_all(path.parent().unwrap()).unwrap();
    let file = std::fs::OpenOptions::new()
        .create(true)
        .truncate(false)
        .write(true)
        .open(path)
        .unwrap();
    let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX | libc::LOCK_NB) };
    assert_eq!(rc, 0, "test must hold the rendezvous flock at {path:?}");
    file
}

#[tokio::test]
async fn append_leaves_a_rendezvous_lock_file_under_arro_locks() {
    // Lock files are rendezvous points: created on demand, left in place,
    // and owned by the kernel namespace (never written into user trees).
    let root = tmp_dir("zarr_lock_created").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();

    storage
        .append_vectors("main--matrix", &filled(2, 4, 1.0))
        .await
        .unwrap();

    let lock = lock_file_of(&root, "main--matrix");
    assert!(
        lock.is_file(),
        "rendezvous lock must exist at {lock:?} after the append"
    );
    // The user tree stays pristine: no lock artifacts inside the array.
    assert!(root.join("matrix").join("zarr.json").is_file());
    assert!(!root.join("matrix").join(".lock").exists());
}

#[cfg(unix)]
#[tokio::test]
async fn append_fails_fast_when_a_foreign_writer_holds_the_dataset_lock() {
    let root = tmp_dir("zarr_lock_ext_append").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    write_arange_f32(&root.join("other"), 20, 4, 4);
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();

    let held = hold_flock(&lock_file_of(&root, "main--matrix"));

    let err = storage
        .append_vectors("main--matrix", &filled(2, 4, 1.0))
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::LockWouldBlock { .. }),
        "{err:?}"
    );
    assert!(err.to_string().contains("main--matrix.lock"), "{err:?}");
    // No partial write: the shape is untouched.
    assert_eq!(
        crate::zzarr::open(&root.join("matrix")).unwrap().shape(),
        vec![50, 4]
    );
    // Per-dataset rendezvous: a different dataset proceeds.
    storage
        .append_vectors("main--other", &filled(2, 4, 2.0))
        .await
        .unwrap();
    // Released: the same append goes through.
    drop(held);
    storage
        .append_vectors("main--matrix", &filled(2, 4, 3.0))
        .await
        .unwrap();
}

#[cfg(unix)]
#[tokio::test]
async fn overwrite_fails_fast_when_a_foreign_writer_holds_the_dataset_lock() {
    let root = tmp_dir("zarr_lock_ext_overwrite").await.join("main");
    write_arange_f32(&root.join("matrix"), 50, 4, 10);
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();

    let held = hold_flock(&lock_file_of(&root, "main--matrix"));

    let err = storage
        .overwrite_vectors(
            "main--matrix",
            &[crate::traits::zarr::RowUpdate {
                row_index: 0,
                vector: vec![9.0; 4],
            }],
        )
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::LockWouldBlock { .. }),
        "{err:?}"
    );
    assert_eq!(
        read_rows_f32(&root.join("matrix"), 0, 1, 4),
        vec![0.0f32, 1.0, 2.0, 3.0],
        "no partial overwrite under contention"
    );
    drop(held);
}

#[cfg(unix)]
#[tokio::test]
async fn creation_fails_fast_when_a_foreign_writer_holds_the_dataset_lock() {
    // Creation takes the same rendezvous as appends: save_dense and
    // append_vectors exclude each other across processes.
    let root = tmp_dir("zarr_lock_ext_create").await.join("main");
    let storage = seeded(&root, "main", 2, 2).await;

    let held = hold_flock(&lock_file_of(&root, "main--held"));
    let err = storage
        .save_dense("held", &filled(2, 2, 1.0), &storage.metadata_path())
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::LockWouldBlock { .. }),
        "{err:?}"
    );
    assert!(!root.join("held").exists(), "no partial creation");
    drop(held);
    storage
        .save_dense("held", &filled(2, 2, 1.0), &storage.metadata_path())
        .await
        .unwrap();
}

#[tokio::test]
async fn in_process_appends_queue_behind_a_foreign_lock_on_another_dataset() {
    // In-process writers queue on the mailbox; only a foreign holder of
    // THE SAME dataset's rendezvous fails fast. Three concurrent appends
    // must all succeed (the acceptance-suite partition contract).
    let root = tmp_dir("zarr_lock_queue").await.join("main");
    write_arange_f32(&root.join("matrix"), 10, 4, 4);
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();

    let sizes = [3usize, 2, 5];
    let handles: Vec<_> = sizes
        .iter()
        .map(|&size| {
            let storage = storage.clone();
            tokio::spawn(async move {
                storage
                    .append_vectors("main--matrix", &filled(size, 4, size as f64))
                    .await
                    .unwrap()
            })
        })
        .collect();
    let mut results: Vec<(usize, usize)> = Vec::new();
    for h in handles {
        results.push(h.await.unwrap());
    }
    results.sort();
    assert_eq!(results[0].0, 10);
    for i in 0..results.len() - 1 {
        assert_eq!(results[i].1, results[i + 1].0, "gap between {results:?}");
    }
    assert_eq!(results[2].1, 20);
}

#[tokio::test]
async fn scan_ignores_the_arro_locks_dir() {
    // The locks dir sits inside the kernel registry namespace; discovery
    // never surfaces it, with or without lock files present.
    let root = tmp_dir("zarr_lock_scan").await.join("main");
    let storage = seeded(&root, "main", 10, 4).await;
    storage
        .save_dense("matrix", &filled(10, 4, 0.0), &storage.metadata_path())
        .await
        .unwrap();
    storage
        .append_vectors("main--matrix", &filled(2, 4, 1.0))
        .await
        .unwrap();
    assert!(lock_file_of(&root, "main--matrix").is_file());

    let all = storage.list_datasets().await.unwrap();
    assert_eq!(
        all.iter()
            .map(|s| s.dataset_id.as_str())
            .collect::<Vec<_>>(),
        vec!["main--matrix"]
    );
}

#[tokio::test]
async fn registry_shape_refresh_reads_the_disk_shape_not_a_caller_value() {
    // Corrupt the registered shape; the refresh must restore the disk
    // truth by reading the array's own metadata inside the cycle.
    let root = tmp_dir("zarr_refresh_disk").await.join("main");
    let storage = seeded(&root, "main", 10, 4).await;
    storage
        .save_dense("matrix", &filled(10, 4, 0.0), &storage.metadata_path())
        .await
        .unwrap();
    {
        let mut md = storage.load_metadata().await.unwrap();
        let info = md.files.get_mut("matrix").unwrap();
        info.rows = 3;
        info.cols = 9;
        storage.save_metadata(&md).await.unwrap();
    }

    storage.update_registered_shape("matrix").await.unwrap();

    let md = storage.load_metadata().await.unwrap();
    assert_eq!(
        (md.files["matrix"].rows, md.files["matrix"].cols),
        (10, 4),
        "refresh must record the zarr.json shape"
    );
}

#[tokio::test]
async fn append_registry_refresh_converges_to_the_disk_shape() {
    // Deterministic replay of the stale-registry race: the refresh must
    // read the shape from disk inside the commit-actor cycle, so a
    // refresh that lands after a later write still records the disk
    // truth — not the append's own (start + m) value.
    let root = tmp_dir("zarr_refresh_conv").await.join("main");
    let storage = seeded(&root, "main", 10, 4).await;
    storage
        .save_dense("matrix", &filled(10, 4, 0.0), &storage.metadata_path())
        .await
        .unwrap();

    // Hold the registry commit actor; signal once the guard is inside.
    use std::sync::Arc;
    use std::sync::atomic::{AtomicBool, Ordering};

    let (release_tx, release_rx) = tokio::sync::oneshot::channel::<()>();
    let actor_held = Arc::new(AtomicBool::new(false));
    let flag = actor_held.clone();
    let md_path = storage.metadata_path();
    let holder = tokio::spawn(async move {
        crate::commit::with_commit_actor(&md_path, || async move {
            flag.store(true, Ordering::SeqCst);
            let _ = release_rx.await;
            Ok::<(), crate::StorageError>(())
        })
        .await
        .unwrap();
    });
    while !actor_held.load(Ordering::SeqCst) {
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    // Append +2 rows; the data write completes, the refresh parks on the
    // held commit actor.
    let app = tokio::spawn({
        let storage = storage.clone();
        async move {
            storage
                .append_vectors("main--matrix", &filled(2, 4, 7.0))
                .await
        }
    });
    // Wait until the append's tail landed (rows 10..12 hold 7.0).
    loop {
        let tail = read_rows_f64(&root.join("matrix"), 10, 12, 4);
        if tail == vec![7.0f64; 8] {
            break;
        }
        tokio::time::sleep(std::time::Duration::from_millis(5)).await;
    }

    // Grow the array directly (+5 rows) while the refresh is parked.
    {
        let dir = root.join("matrix");
        tokio::task::spawn_blocking(move || {
            let mut arr = crate::zzarr::open(&dir).unwrap();
            arr.append(&[2.0f64; 20]).unwrap();
        })
        .await
        .unwrap();
    }

    // Release the actor; the parked refresh must now record 17 (disk).
    release_tx.send(()).unwrap();
    let (start, new_n) = tokio::time::timeout(std::time::Duration::from_secs(10), app)
        .await
        .expect("append must not deadlock")
        .unwrap()
        .unwrap();
    assert_eq!((start, new_n), (10, 12));
    holder.await.unwrap();

    let md = storage.load_metadata().await.unwrap();
    assert_eq!(
        md.files["matrix"].rows, 17,
        "refresh must read the disk shape, not the append's own value"
    );
    assert_eq!(md.files["matrix"].cols, 4);
}

#[tokio::test]
async fn summarize_by_id_resolves_and_guards() {
    // Serving-layer registry need (#8): resolve a dataset ID to its node
    // summary in one call — label check, traversal guard, existence, and
    // the O(1) node summary.
    let root = tmp_dir("zarr_summ_by_id").await.join("main");
    write_v3_group(&root);
    let values: Vec<f32> = (0..6).map(|i| i as f32 * 0.5).collect();
    crate::zzarr::write_array(
        &root.join("sub").join("cube"),
        &[2, 3],
        &[2, 3],
        &values,
        true,
    )
    .unwrap();
    let storage = ZarrStorage::new(root.clone(), "main".to_string()).unwrap();

    let summary = storage.summarize_by_id("main--sub--cube").await.unwrap();
    assert_eq!(summary.dataset_id, "main--sub--cube");
    assert_eq!(summary.path, "sub/cube");
    assert_eq!(summary.shape, vec![2, 3]);
    assert_eq!(summary.dtype, "float32");

    // Foreign label, traversal and missing nodes are rejected.
    let err = storage
        .summarize_by_id("other--sub--cube")
        .await
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
    let err = storage.summarize_by_id("main--..").await.unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
    let err = storage.summarize_by_id("main--nope").await.unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)), "{err:?}");
}
