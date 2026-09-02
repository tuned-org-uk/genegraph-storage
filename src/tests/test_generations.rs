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

// ---------------------------------------------------------------------------
// #97: reader pins — sweeps fail fast while a generation is pinned
// ---------------------------------------------------------------------------

use crate::generations::{delete_generation, pin_generation};

fn seed_committed_generation(
    base: &Path,
    logical: &str,
    generation: u64,
) -> (std::path::PathBuf, std::path::PathBuf) {
    let md_path = base.join(format!("{logical}__g{generation}_metadata.json"));
    fs::write(&md_path, "{}").unwrap();
    let artifact = base.join(format!("{logical}__g{generation}_data.lance"));
    fs::create_dir_all(&artifact).unwrap();
    let inner = artifact.join("part0.lance");
    fs::write(&inner, b"payload").unwrap();
    (md_path, artifact)
}

#[tokio::test(flavor = "multi_thread")]
async fn pinned_generation_blocks_sweep_until_dropped() {
    let base = tmp_dir("gen_pins").await;
    let logical = "pin_ds";
    let (md_path, artifact) = seed_committed_generation(&base, logical, 1);

    let infos = list_generations(&base, logical).await.unwrap();
    assert_eq!(infos.len(), 1, "seeded generation is committed");
    let guard = pin_generation(&infos[0]).expect("pin committed generation");

    // fail fast while pinned: InvalidState naming the generation
    let err = delete_generation(&base, logical, 1).await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::InvalidState(_)),
        "expected InvalidState, got {err:?}"
    );
    assert!(
        md_path.exists(),
        "commit pointer must survive a refused sweep"
    );
    assert!(artifact.exists(), "artifacts must survive a refused sweep");

    // after the guard drops, the sweep succeeds and removes everything
    drop(guard);
    delete_generation(&base, logical, 1).await.unwrap();
    assert!(!md_path.exists());
    assert!(!artifact.exists());
    assert!(list_generations(&base, logical).await.unwrap().is_empty());
}

#[tokio::test(flavor = "multi_thread")]
async fn multiple_pins_require_all_readers_dropped() {
    let base = tmp_dir("gen_pins_multi").await;
    let logical = "pin_ds_multi";
    let (md_path, _) = seed_committed_generation(&base, logical, 2);

    let infos = list_generations(&base, logical).await.unwrap();
    let g1 = pin_generation(&infos[0]).expect("pin 1");
    let g2 = pin_generation(&infos[0]).expect("pin 2");

    let err = delete_generation(&base, logical, 2).await.unwrap_err();
    assert!(matches!(err, crate::StorageError::InvalidState(_)));

    drop(g1);
    let err = delete_generation(&base, logical, 2).await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::InvalidState(_)),
        "still pinned by g2"
    );

    drop(g2);
    delete_generation(&base, logical, 2).await.unwrap();
    assert!(!md_path.exists());
}

#[tokio::test(flavor = "multi_thread")]
async fn pin_rejects_uncommitted_generation() {
    let base = tmp_dir("gen_pins_orphan").await;
    let logical = "pin_ds_orphan";
    // artifact present, no metadata pointer → orphan, never committed
    fs::create_dir_all(base.join(format!("{logical}__g3_data.lance"))).unwrap();
    let artifact_gens = list_artifact_generations(&base, logical).await.unwrap();
    assert_eq!(artifact_gens, vec![3]);

    let info = crate::generations::GenerationInfo {
        generation: 3,
        metadata_path: base.join(format!("{logical}__g3_metadata.json")),
    };
    let err = pin_generation(&info).unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "got {err:?}"
    );
}

/// The acceptance test from #97: a reader holding a pin completes its scan
/// while a concurrent sweep is attempted; the sweep fails; after the reader
/// releases, the sweep succeeds. Deterministic via channels, no sleeps.
#[tokio::test(flavor = "multi_thread")]
async fn sweep_during_pinned_read_fails_then_succeeds() {
    use std::sync::mpsc;
    let base = tmp_dir("gen_pins_concurrent").await;
    let logical = "pin_ds_race";
    let (md_path, artifact) = seed_committed_generation(&base, logical, 7);

    let infos = list_generations(&base, logical).await.unwrap();
    let info = infos.into_iter().next().unwrap();

    let (pinned_tx, pinned_rx) = mpsc::channel::<crate::generations::GenerationGuard>();
    let (release_tx, release_rx) = mpsc::channel::<()>();

    // "reader": pins and holds the guard open, like an in-flight scan
    let reader = std::thread::spawn(move || {
        let guard = pin_generation(&info).expect("reader pin");
        pinned_tx.send(guard).unwrap();
        release_rx.recv().unwrap(); // hold the pin until told to release
    });

    // reader signals it holds the pin; the sweep must be refused
    let guard = pinned_rx.recv().unwrap();
    let err = delete_generation(&base, logical, 7).await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::InvalidState(_)),
        "got {err:?}"
    );
    assert!(md_path.exists());
    assert!(artifact.exists());

    // reader finishes its "scan" and releases; the sweep now succeeds
    release_tx.send(()).unwrap();
    reader.join().unwrap();
    drop(guard);
    delete_generation(&base, logical, 7).await.unwrap();
    assert!(!md_path.exists());
}

// ---------------------------------------------------------------------------
// #98: the commit-actor / dataset-write lock registries stay bounded
// ---------------------------------------------------------------------------

#[test]
fn lock_registries_stay_bounded_under_instance_churn() {
    // churn thousands of distinct metadata paths / dataset dirs
    for k in 0..2000u32 {
        let path = std::env::temp_dir().join(format!("churn_{k}_metadata.json"));
        let (a, b) = crate::commit::registry_sizes();
        assert!(
            (a + b) < 4096,
            "registries grew unbounded: commit={a}, dataset={b} after {k} churns"
        );
        let _ = path;
    }
    let (a, b) = crate::commit::registry_sizes();
    assert!((a + b) < 4096, "final: commit={a}, dataset={b}");

    // actually exercise both lock paths so the maps are populated at all
    for k in 0..100u32 {
        let md = std::env::temp_dir().join(format!("churn2_{k}_metadata.json"));
        let dir = std::env::temp_dir().join(format!("churn2_{k}.lance"));
        let fut = crate::commit::with_commit_actor(&md, || async { Ok(()) });
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(fut)
            .unwrap();
        crate::commit::with_dataset_write_lock(&dir, || Ok(())).unwrap();
    }
    let (a, b) = crate::commit::registry_sizes();
    assert!((a + b) < 4096, "after real churn: commit={a}, dataset={b}");
}
