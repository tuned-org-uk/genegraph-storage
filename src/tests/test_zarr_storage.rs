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

use std::path::Path;

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
