//! #100: downstream canary for the public commit-serialization surface.
//!
//! This file links `genegraph-storage` as an *external* crate — the same
//! view a consumer (genefold-vd) has. If `with_commit_actor`,
//! `with_file_lock`, or `lock_file_for_metadata` lose their `pub`
//! visibility, this file stops compiling.
//!
//! Run with `cargo test --release --test api_public`.

use std::path::PathBuf;

use genegraph_storage::commit::{lock_file_for_metadata, with_commit_actor, with_file_lock};
use genegraph_storage::generations::write_json_atomic;

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

/// The full downstream recipe for a consumer's own metadata
/// read-modify-write cycle: cross-process advisory lock (independent CLI
/// processes) around an in-process commit-actor cycle, publishing through
/// the sanctioned atomic write.
#[tokio::test]
async fn downstream_metadata_cycle_is_publicly_composable() {
    let dir = scratch_dir("cycle");
    let metadata_path = dir.join("ds__g1_metadata.json");
    write_json_atomic(&metadata_path, r#"{"v":0}"#).unwrap();
    let lock_path = lock_file_for_metadata(&metadata_path);
    assert_eq!(lock_path.file_name().unwrap(), "ds__g1_metadata.lock");

    with_file_lock(&lock_path, || {
        // The closure is sync (it runs on the blocking pool): a consumer's
        // serde_json load → mutate → write_json_atomic cycle happens here.
        Ok(())
    })
    .await
    .unwrap();

    // The in-process commit actor is reachable with the same shape the
    // internal save_* registry paths use.
    let result: genegraph_storage::StorageResult<()> =
        with_commit_actor(&metadata_path, || async { Ok(()) }).await;
    result.unwrap();

    let _ = std::fs::remove_dir_all(&dir);
}
