//! #100: downstream canary for the public commit-serialization surface.
//!
//! This file links `genegraph-storage` as an *external* crate — the same
//! view a consumer (genefold-vd) has. If `with_commit_actor`,
//! `with_file_lock`, `with_metadata_file_lock`, or
//! `lock_file_for_metadata` lose their `pub` visibility, this file stops
//! compiling.
//!
//! #105 extends the guarded surface with the fail-fast try forms
//! (`try_with_file_lock`, `try_with_metadata_file_lock`) and the
//! distinctly matchable `StorageError::LockWouldBlock`.
//!
//! Run with `cargo test --release --test api_public`.

use std::path::PathBuf;

use genegraph_storage::commit::{
    lock_file_for_metadata, try_with_file_lock, try_with_metadata_file_lock, with_commit_actor,
    with_file_lock, with_metadata_file_lock,
};
use genegraph_storage::generations::write_json_atomic;
use genegraph_storage::StorageError;

fn scratch_dir(tag: &str) -> PathBuf {
    let dir = std::env::temp_dir().join(format!(
        "api_public_{tag}_{}",
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_nanos()
    ));
    std::fs::create_dir_all(&dir).unwrap();
    dir
}

/// The downstream recipe executes a real read-modify-write while BOTH
/// locks are held: the flock is taken around the awaited commit-actor
/// cycle, and a foreign process-shaped flock holder is excluded until the
/// cycle completes.
#[tokio::test]
async fn downstream_metadata_cycle_runs_under_both_locks() {
    let dir = scratch_dir("composed");
    let metadata_path = dir.join("ds__g1_metadata.json");
    write_json_atomic(&metadata_path, r#"{"count":0}"#).unwrap();
    assert_eq!(
        lock_file_for_metadata(&metadata_path)
            .file_name()
            .unwrap(),
        "ds__g1_metadata.lock"
    );

    // A foreign holder takes the lock file first, as an independent CLI
    // process would, and parks.
    let lock_path = lock_file_for_metadata(&metadata_path);
    let (held_tx, held_rx) = std::sync::mpsc::channel::<()>();
    let (proceed_tx, proceed_rx) = std::sync::mpsc::channel::<()>();
    let lock_for_foreign = lock_path.clone();
    let foreign = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                with_file_lock(&lock_for_foreign, move || {
                    held_tx.send(()).unwrap();
                    proceed_rx.recv().unwrap();
                    Ok(())
                })
                .await
                .unwrap()
            })
    });
    held_rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();

    // The composed cycle waits for the foreign holder, then runs its RMW
    // (read → increment → atomic publish) under flock + commit actor.
    let md = metadata_path.clone();
    let md_ref = md.clone();
    let cycle_task = tokio::spawn(async move {
        with_metadata_file_lock(&md_ref, move || async move {
            let mut doc: serde_json::Value =
                serde_json::from_str(&std::fs::read_to_string(&md).unwrap()).unwrap();
            let count = doc["count"].as_u64().unwrap();
            doc["count"] = serde_json::json!(count + 1);
            write_json_atomic(&md, &doc.to_string()).unwrap();
            Ok(())
        })
        .await
        .unwrap();
    });

    // While the foreign holder parks, the composed cycle must not run its
    // RMW (the counter file is untouched).
    std::thread::sleep(std::time::Duration::from_millis(200));
    assert!(
        !cycle_task.is_finished(),
        "composed cycle must wait for the foreign lock holder"
    );

    // Release; the cycle runs and the RMW lands.
    proceed_tx.send(()).unwrap();
    foreign.join().unwrap();
    cycle_task.await.unwrap();
    assert_eq!(
        std::fs::read_to_string(&metadata_path).unwrap(),
        r#"{"count":1}"#,
        "the RMW must execute exactly once, under both locks"
    );

    let _ = std::fs::remove_dir_all(&dir);
}

/// The in-process commit actor remains reachable on its own for consumers
/// that only need in-process serialization.
#[tokio::test]
async fn in_process_actor_cycle_is_publicly_callable() {
    let dir = scratch_dir("actor");
    let metadata_path = dir.join("ds__g1_metadata.json");
    let result: genegraph_storage::StorageResult<()> =
        with_commit_actor(&metadata_path, || async { Ok(()) }).await;
    result.unwrap();
    let _ = std::fs::remove_dir_all(&dir);
}

/// #105: the fail-fast try forms are publicly callable from downstream and
/// surface contention as the distinctly matchable `LockWouldBlock` naming
/// the lock file — the shape a multi-process CLI maps onto its own exit
/// codes without waiting.
#[tokio::test]
async fn downstream_try_lock_cycle_maps_contention_to_lock_would_block() {
    let dir = scratch_dir("try");
    let metadata_path = dir.join("ds__g1_metadata.json");
    write_json_atomic(&metadata_path, r#"{"count":0}"#).unwrap();
    let lock_path = lock_file_for_metadata(&metadata_path);

    // A foreign holder parks on the lock file, as an independent CLI
    // process would.
    let (held_tx, held_rx) = std::sync::mpsc::channel::<()>();
    let (proceed_tx, proceed_rx) = std::sync::mpsc::channel::<()>();
    let lock_for_foreign = lock_path.clone();
    let foreign = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                with_file_lock(&lock_for_foreign, move || {
                    held_tx.send(()).unwrap();
                    proceed_rx.recv().unwrap();
                    Ok(())
                })
                .await
                .unwrap()
            })
    });
    held_rx.recv_timeout(std::time::Duration::from_secs(5)).unwrap();

    // Both try forms fail fast, naming the lock file.
    let raw_err = try_with_file_lock(&lock_path, || Ok::<(), StorageError>(()))
        .await
        .unwrap_err();
    let composed_err: StorageError =
        try_with_metadata_file_lock(&metadata_path, || async { Ok(()) })
            .await
            .unwrap_err();
    for err in [raw_err, composed_err] {
        match err {
            StorageError::LockWouldBlock { path } => {
                assert_eq!(path, lock_path, "the error must name the lock file");
            }
            other => panic!("expected LockWouldBlock, got {other:?}"),
        }
    }

    // Release; the try forms succeed downstream.
    proceed_tx.send(()).unwrap();
    foreign.join().unwrap();
    let md = metadata_path.clone();
    try_with_metadata_file_lock(&metadata_path, move || async move {
        let mut doc: serde_json::Value =
            serde_json::from_str(&std::fs::read_to_string(&md).unwrap()).unwrap();
        doc["count"] = serde_json::json!(doc["count"].as_u64().unwrap() + 1);
        write_json_atomic(&md, &doc.to_string()).unwrap();
        Ok(())
    })
    .await
    .unwrap();
    assert_eq!(
        std::fs::read_to_string(&metadata_path).unwrap(),
        r#"{"count":1}"#,
        "the uncontended try cycle must publish its RMW"
    );

    let _ = std::fs::remove_dir_all(&dir);
}
