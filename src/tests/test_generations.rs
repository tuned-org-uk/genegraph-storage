//! #93 / RFC #81-P5: transactional generations.
//!
//! Artifact generations are immutable: every append mints `{logical}__g{N}`
//! artifact paths and commits by atomically publishing the generation's
//! metadata JSON (the single commit pointer). A generation without a
//! metadata file was never committed and is invisible to readers.

use std::fs;
use std::path::Path;

use crate::generations::{
    GenerationInfo, generation_name, list_artifact_generations, list_generations, logical_name,
    parse_generation, write_json_atomic,
};
use crate::lance_storage_graph::LanceStorageGraph;
use crate::traits::backend::StorageBackend;

use super::tmp_dir;

/// `{logical}__g{N}` is reserved: logical names round-trip, generations parse.
#[test]
fn test_generation_naming_roundtrip() {
    assert_eq!(generation_name("ds_ab12", 0), "ds_ab12__g0");
    assert_eq!(generation_name("ds_ab12", 17), "ds_ab12__g17");
    assert_eq!(logical_name("ds_ab12__g17"), "ds_ab12");
    assert_eq!(logical_name("ds_ab12__g0"), "ds_ab12");
    assert_eq!(logical_name("ds_ab12"), "ds_ab12", "no suffix = logical");
    assert_eq!(parse_generation("ds_ab12__g17"), Some(17));
    assert_eq!(parse_generation("ds_ab12"), None);
    assert_eq!(parse_generation("ds_ab12__g1x"), None, "non-digit suffix");
    // names that merely contain the separator mid-name are left alone
    assert_eq!(logical_name("ds__g1x"), "ds__g1x");
    // double suffix strips the last generation only
    assert_eq!(logical_name("ds__g1__g2"), "ds__g1");
    assert_eq!(parse_generation("ds__g1__g2"), Some(2));
}

/// Atomic publish: the file always holds a complete document, overwrites
/// work, and no `.tmp` residue remains.
#[tokio::test(flavor = "multi_thread")]
async fn test_write_json_atomic_overwrites_completely() {
    let dir = tmp_dir("test_write_json_atomic_overwrites_completely").await;
    let path = dir.join("ds__g1_metadata.json");

    write_json_atomic(&path, r#"{"v": 1}"#).expect("first publish must succeed");
    assert_eq!(fs::read_to_string(&path).unwrap(), r#"{"v": 1}"#);

    // overwrite with a longer document: no truncation window residue
    write_json_atomic(&path, r#"{"v": 2, "files": {"rawinput": {}}}"#)
        .expect("republish must succeed");
    assert_eq!(
        fs::read_to_string(&path).unwrap(),
        r#"{"v": 2, "files": {"rawinput": {}}}"#
    );

    let residue: Vec<_> = fs::read_dir(&dir)
        .unwrap()
        .filter_map(|e| e.ok())
        .filter(|e| e.path().extension().is_some_and(|x| x == "tmp"))
        .collect();
    assert!(residue.is_empty(), "tmp files must not leak: {residue:?}");

    let _ = fs::remove_dir_all(&dir);
}

/// Only generations with a metadata file are committed/listed; artifact
/// generations without one (orphans from a pre-commit crash) are visible
/// to the sweep API but never to resolution.
#[tokio::test(flavor = "multi_thread")]
async fn test_list_generations_ignores_orphans() {
    let dir = tmp_dir("test_list_generations_ignores_orphans").await;

    // committed gen 1 and gen 3, orphaned artifacts of gen 2
    for (generation, rows) in [(1u64, 10usize), (3, 30)] {
        write_json_atomic(
            &dir.join(format!("ds__g{generation}_metadata.json")),
            &format!(r#"{{"nrows": {rows}}}"#),
        )
        .unwrap();
    }
    fs::create_dir_all(dir.join("ds__g2_rawinput.lance")).unwrap();

    let committed = list_generations(Path::new(&dir), "ds").await.unwrap();
    assert_eq!(
        committed,
        vec![
            GenerationInfo {
                generation: 1,
                metadata_path: dir.join("ds__g1_metadata.json")
            },
            GenerationInfo {
                generation: 3,
                metadata_path: dir.join("ds__g3_metadata.json")
            },
        ],
        "ascending, committed only"
    );

    let artifacts = list_artifact_generations(Path::new(&dir), "ds")
        .await
        .unwrap();
    assert_eq!(artifacts, vec![1, 2, 3], "orphans visible to the sweep");

    let _ = fs::remove_dir_all(&dir);
}

/// `delete_generation` removes the generation's artifacts and metadata;
/// prefix matches must not leak into sibling datasets (`ds` vs `ds2`).
#[tokio::test(flavor = "multi_thread")]
async fn test_delete_generation_is_prefix_exact() {
    let dir = tmp_dir("test_delete_generation_is_prefix_exact").await;

    for name in ["ds__g1_rawinput.lance", "ds2__g1_rawinput.lance"] {
        fs::create_dir_all(dir.join(name)).unwrap();
    }
    write_json_atomic(&dir.join("ds__g1_metadata.json"), "{}").unwrap();
    write_json_atomic(&dir.join("ds2__g1_metadata.json"), "{}").unwrap();

    crate::generations::delete_generation(Path::new(&dir), "ds", 1)
        .await
        .expect("delete must succeed");

    assert!(!dir.join("ds__g1_rawinput.lance").exists());
    assert!(!dir.join("ds__g1_metadata.json").exists());
    assert!(
        dir.join("ds2__g1_rawinput.lance").exists(),
        "sibling intact"
    );
    assert!(dir.join("ds2__g1_metadata.json").exists(), "sibling intact");

    // orphan sweep: deleting a never-committed generation is a no-op-safe path
    fs::create_dir_all(dir.join("ds__g9_rawinput.lance")).unwrap();
    crate::generations::delete_generation(Path::new(&dir), "ds", 9)
        .await
        .expect("orphan sweep must succeed");
    assert!(!dir.join("ds__g9_rawinput.lance").exists());

    let _ = fs::remove_dir_all(&dir);
}

/// A scoped handle routes artifact IO at the generation's paths while
/// keeping the logical identity accessible.
#[tokio::test(flavor = "multi_thread")]
async fn test_scoped_generation_routes_artifact_paths() {
    let dir = tmp_dir("test_scoped_generation_routes_artifact_paths").await;
    let storage = LanceStorageGraph::new(dir.to_string_lossy().to_string(), "ds".to_string())
        .scoped_generation(3);

    assert_eq!(storage.get_name(), "ds__g3");
    assert_eq!(
        storage.file_path("rawinput"),
        dir.join("ds__g3_rawinput.lance")
    );
    assert_eq!(
        storage.metadata_path(),
        dir.join("ds__g3_metadata.json"),
        "per-generation metadata = per-generation commit pointer"
    );
    assert_eq!(crate::generations::logical_name(&storage.get_name()), "ds");

    let _ = fs::remove_dir_all(&dir);
}

/// Generation 0 is the initial build: scoped_generation(0) equals a plain
/// instance named `{logical}__g0` and the metadata path stays distinct.
#[tokio::test(flavor = "multi_thread")]
async fn test_scoped_generation_zero_is_the_build_generation() {
    let dir = tmp_dir("test_scoped_generation_zero_is_the_build_generation").await;
    let storage = LanceStorageGraph::new(dir.to_string_lossy().to_string(), "ds".to_string())
        .scoped_generation(0);

    assert_eq!(
        storage.file_path("lambdas"),
        dir.join("ds__g0_lambdas.lance")
    );
    assert_eq!(storage.metadata_path(), dir.join("ds__g0_metadata.json"));

    let _ = fs::remove_dir_all(&dir);
}
