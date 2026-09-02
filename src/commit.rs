//! Commit serialization for metadata read-modify-write cycles and dataset
//! writes.
//!
//! Base concurrency-safety model borrowed from duva's actor design
//! ([`<https://github.com/Migorithm/duva>`]): every storage instance has a single
//! logical *commit actor*. All metadata mutations (the `load_metadata →
//! mutate → save_metadata` cycle performed by every `save_*` call) are
//! serialized through it, so concurrent writers cannot interleave their
//! cycles and lose each other's registry entries (lost update) — the same
//! reason duva routes writes through one actor mailbox instead of shared
//! mutable state.
//!
//! The same mailbox shape extends to Lance dataset writes
//! ([`with_dataset_write_lock`]): manifest-version allocation and the
//! commit-point publish of one dataset directory are serialized, so two
//! concurrent overwrites cannot mint the same `N.manifest` (#95).
//!
//! Durability of the commit itself is the tmp + fsync + rename discipline
//! ([`crate::generations::write_json_atomic`] for metadata, the same
//! sequence inside the lancefmt writer for data/txn/manifest files):
//! readers never observe a half-written commit pointer, and a read after a
//! completed commit observes its effects (read-your-own-writes).
//! Cross-process arbitration stays with the transactional-generations work
//! (#93/#81-P5).

use std::collections::HashMap;
use std::path::Path;
use std::sync::{Arc, Mutex as StdMutex, OnceLock};

use tokio::sync::Mutex;

use crate::{StorageError, StorageResult};

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

/// One dataset-write mailbox per dataset directory (process-wide, sync —
/// `write_dataset` runs on the blocking pool).
///
/// Serializes manifest-version allocation (`latest_manifest_version` →
/// `+1`) against concurrent overwrites of the same dataset (#95): without
/// it two writers could mint the same `N.manifest` and one overwrite would
/// be silently lost.
fn dataset_lock_for(dataset_dir: &Path) -> Arc<StdMutex<()>> {
    static LOCKS: OnceLock<StdMutex<HashMap<String, Arc<StdMutex<()>>>>> = OnceLock::new();
    let locks = LOCKS.get_or_init(|| StdMutex::new(HashMap::new()));
    let key = dataset_dir.to_string_lossy().to_string();
    Arc::clone(
        locks
            .lock()
            .expect("dataset lock registry poisoned")
            .entry(key)
            .or_insert_with(|| Arc::new(StdMutex::new(()))),
    )
}

/// Runs `write` under the dataset's write mailbox. `write` must be a
/// non-async closure: the whole dataset write — version allocation through
/// commit-point publish — happens under the lock.
pub(crate) fn with_dataset_write_lock<T>(
    dataset_dir: &Path,
    write: impl FnOnce() -> StorageResult<T>,
) -> StorageResult<T> {
    let mailbox = dataset_lock_for(dataset_dir);
    let _guard = mailbox
        .lock()
        .map_err(|_| StorageError::InvalidState("dataset write lock poisoned".into()))?;
    write()
}
