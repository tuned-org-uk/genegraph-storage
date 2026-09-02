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
//! Both registries hold **weak** references (#98): a mailbox stays alive
//! only while some caller holds its `Arc` (i.e. while a commit cycle or
//! dataset write is in flight), and dead entries are swept on insert.
//! Instances that churn (create/drop thousands of collections) keep the
//! registries bounded.
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
use std::sync::{Arc, Mutex as StdMutex, OnceLock, Weak};

use tokio::sync::Mutex;

use crate::{StorageError, StorageResult};

/// Commit-actor mailboxes, keyed by metadata path (weak-valued, #98).
static COMMIT_LOCKS: OnceLock<StdMutex<HashMap<String, Weak<Mutex<()>>>>> = OnceLock::new();
/// Dataset-write mailboxes, keyed by dataset dir (weak-valued, #98).
static DATASET_LOCKS: OnceLock<StdMutex<HashMap<String, Weak<StdMutex<()>>>>> = OnceLock::new();

/// Shared weak-registry lookup (#98): reuse the live mailbox for `key` if
/// one exists, otherwise sweep dead entries and insert `fresh`.
pub(crate) fn weak_lookup<T>(
    registry: &'static OnceLock<StdMutex<HashMap<String, Weak<T>>>>,
    key: String,
    fresh: Arc<T>,
) -> Arc<T> {
    let mut map = registry
        .get_or_init(|| StdMutex::new(HashMap::new()))
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    if let Some(weak) = map.get(&key)
        && let Some(strong) = weak.upgrade()
    {
        return strong;
    }
    map.retain(|_, weak| weak.strong_count() > 0);
    let arc = Arc::clone(&fresh);
    map.insert(key, Arc::downgrade(&fresh));
    arc
}

/// One commit-actor mailbox per metadata path.
fn lock_for(metadata_path: &Path) -> Arc<Mutex<()>> {
    let key = metadata_path.to_string_lossy().to_string();
    weak_lookup(&COMMIT_LOCKS, key, Arc::new(Mutex::new(())))
}

/// One dataset-write mailbox per dataset directory.
fn dataset_lock_for(dataset_dir: &Path) -> Arc<StdMutex<()>> {
    let key = dataset_dir.to_string_lossy().to_string();
    weak_lookup(&DATASET_LOCKS, key, Arc::new(StdMutex::new(())))
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

/// Test hook: (commit-registry size, dataset-registry size).
#[cfg(test)]
pub(crate) fn registry_sizes() -> (usize, usize) {
    let commit = COMMIT_LOCKS
        .get()
        .map(|m| m.lock().unwrap().len())
        .unwrap_or(0);
    let dataset = DATASET_LOCKS
        .get()
        .map(|m| m.lock().unwrap().len())
        .unwrap_or(0);
    (commit, dataset)
}
