//! Commit serialization for metadata read-modify-write cycles.
//!
//! Base concurrency-safety model borrowed from duva's actor design
//! (<https://github.com/Migorithm/duva>): every storage instance has a single
//! logical *commit actor*. All metadata mutations (the `load_metadata →
//! mutate → save_metadata` cycle performed by every `save_*` call) are
//! serialized through it, so concurrent writers cannot interleave their
//! cycles and lose each other's registry entries (lost update) — the same
//! reason duva routes writes through one actor mailbox instead of shared
//! mutable state.
//!
//! Durability of the commit itself remains
//! [`crate::generations::write_json_atomic`] (tmp + fsync + rename): readers
//! never observe a half-written commit pointer, and a read after a completed
//! commit observes its effects (read-your-own-writes). Cross-process
//! arbitration stays with the transactional-generations work (#93/#81-P5).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};

use tokio::sync::Mutex;

use crate::StorageResult;

/// One commit-actor mailbox per metadata path (process-wide).
fn lock_for(metadata_path: &Path) -> Arc<Mutex<()>> {
    static LOCKS: OnceLock<StdMutex<HashMap<String, Arc<Mutex<()>>>>> = OnceLock::new();
    let locks = LOCKS.get_or_init(|| StdMutex::new(HashMap::new()));
    let key = metadata_path.to_string_lossy().to_string();
    Arc::clone(
        locks
            .lock()
            .expect("commit lock registry poisoned")
            .entry(key)
            .or_insert_with(|| Arc::new(Mutex::new(()))),
    )
}

/// Runs `commit` (a full metadata read-modify-write cycle) under the
/// instance's commit actor: at most one cycle runs at a time for the same
/// metadata path.
pub(crate) async fn with_commit_actor<T, F, Fut>(
    metadata_path: &Path,
    commit: F,
) -> StorageResult<T>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = StorageResult<T>>,
{
    let mailbox = lock_for(metadata_path);
    let _guard = mailbox.lock().await;
    commit().await
}
