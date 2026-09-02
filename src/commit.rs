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
//!
//! # Downstream contract (#100)
//!
//! The commit actor is an **in-process** mailbox: it serializes concurrent
//! tasks inside one process, never two independent processes. Two levels of
//! arbitration are exposed:
//!
//! 1. **In-process** — [`with_commit_actor`] wraps a full metadata
//!    read-modify-write cycle (load → mutate → publish). Every cycle over a
//!    given metadata file runs under the same per-path mailbox, so cycles
//!    from your code and cycles from the `save_*` registry paths cannot
//!    interleave and lose updates:
//!
//!    ```no_run
//!    use std::path::Path;
//!    use genegraph_storage::commit::with_commit_actor;
//!
//!    futures::executor::block_on(async {
//!        let metadata_path = Path::new("base/ds__g1_metadata.json");
//!        let cycle: genegraph_storage::StorageResult<()> = with_commit_actor(metadata_path, || async {
//!            // load → mutate → publish (via generations::write_json_atomic)
//!            Ok(())
//!        })
//!        .await;
//!        let _ = cycle;
//!    });
//!    ```
//!
//! 2. **Cross-process** — an advisory `flock` on a lock file, held for the
//!    documented hold scope (the whole read-modify-write cycle: lock →
//!    load → mutate → publish → release). The blessed convention is one
//!    lock file next to the metadata file, named by
//!    [`lock_file_for_metadata`] (`{metadata-stem}.lock`). Two forms
//!    exist:
//!
//!    - [`with_metadata_file_lock`] — the **composed** recipe: the file
//!      lock is held across the whole awaited commit-actor cycle, so
//!      cross-process exclusion *and* in-process actor serialization are
//!      both active for the duration of the cycle. This is what
//!      multi-process consumers whose metadata file is also touched by
//!      async `save_*` paths must use:
//!
//!      ```no_run
//!      use std::path::Path;
//!      use genegraph_storage::commit::with_metadata_file_lock;
//!
//!      futures::executor::block_on(async {
//!          let metadata_path = Path::new("base/ds__g1_metadata.json");
//!          let cycle = with_metadata_file_lock(metadata_path, || async {
//!              // load → mutate → publish (via generations::write_json_atomic)
//!              Ok(())
//!          })
//!          .await;
//!          let _ = cycle;
//!      });
//!      ```
//!
//!    - [`with_file_lock`] — the raw lock for consumers whose whole cycle
//!      is **synchronous**. Its closure runs on the blocking pool and
//!      cannot await the commit actor; it therefore does *not* serialize
//!      against in-process async `save_*` cycles. If that matters, use
//!      [`with_metadata_file_lock`].
//!
//! The lock file is a rendezvous point for cooperating writers, not a
//! commit artifact: it carries no data and is left in place after release.
//! Arbitration is only as strong as the convention — every writer of the
//! same metadata file must take the same lock file before mutating it.
//!
//! Both lock forms wait on the blocking pool; see the operational caution
//! on [`with_metadata_file_lock`] for the assumptions this puts on commit
//! cycles and what to prefer when waits could be prolonged or numerous.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
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
/// metadata path's commit actor: at most one cycle runs at a time for the
/// same path **within this process**.
///
/// Public for downstream consumers (#100): any code that performs its own
/// load → mutate → publish cycle over a metadata file outside the `save_*`
/// registry paths routes the whole cycle through this function with the
/// same `metadata_path` the storage instance uses, so cycles from both
/// sides are serialized against each other. Cross-process arbitration is a
/// separate concern — wrap the cycle in [`with_file_lock`] (see the module
/// docs for the recipe).
pub async fn with_commit_actor<T, F, Fut>(
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

/// Cross-process commit arbitration (#100): an advisory `flock` on
/// `lock_path`, held for the closure's scope — lock → read-modify-write →
/// publish → release. Independent processes (e.g. separate CLI
/// invocations) that take the same lock file cannot interleave their
/// metadata cycles.
///
/// The blessed lock-file location for a metadata file is
/// [`lock_file_for_metadata`] (`{metadata-stem}.lock` next to the file);
/// the lock file is created on demand (missing parent directories
/// included) and left in place after release — it is a rendezvous point,
/// not a commit artifact.
///
/// The closure is synchronous and runs on the blocking pool: the flock can
/// block arbitrarily long on a competing holder, so it must never run on
/// an async executor thread. Because the closure is sync, it **cannot**
/// await [`with_commit_actor`] — a cycle run here is serialized across
/// processes but not against in-process async `save_*` cycles; use
/// [`with_metadata_file_lock`] when both are required. Off unix this fails
/// with [`StorageError::UnsupportedFormat`] rather than silently skipping
/// arbitration.
///
/// The blocking-pool caveats documented on [`with_metadata_file_lock`]
/// apply here identically: unbounded waits, non-abortable waiters,
/// blocking-thread capacity.
pub async fn with_file_lock<T, F>(lock_path: &Path, f: F) -> StorageResult<T>
where
    T: Send + 'static,
    F: FnOnce() -> StorageResult<T> + Send + 'static,
{
    let lock_path = lock_path.to_path_buf();
    tokio::task::spawn_blocking(move || {
        let _lock = FileLock::acquire(&lock_path)?;
        f()
    })
    .await
    .map_err(|e| StorageError::Io(format!("file lock task failed: {e}")))?
}

/// The blessed lock-file path for a metadata file (#100):
/// `{metadata-stem}.lock` next to it — `ds__g1_metadata.json` locks through
/// `ds__g1_metadata.lock`. All cooperating writers of the same metadata
/// file must resolve the lock through this function so the convention
/// holds.
pub fn lock_file_for_metadata(metadata_path: &Path) -> PathBuf {
    match metadata_path.file_stem() {
        Some(stem) => metadata_path.with_file_name(format!("{}.lock", stem.to_string_lossy())),
        None => metadata_path.with_file_name(format!(
            "{}.lock",
            metadata_path.as_os_str().to_string_lossy()
        )),
    }
}

/// RAII advisory lock (unix `flock(2)`, exclusive). The lock is held by the
/// open file description, so two `acquire` calls — in this process or
/// another — exclude each other until the guard drops (explicit `LOCK_UN`,
/// and again on close).
#[cfg(unix)]
struct FileLock(std::fs::File);

#[cfg(unix)]
impl FileLock {
    fn acquire(lock_path: &Path) -> StorageResult<Self> {
        use std::os::unix::io::AsRawFd;

        if let Some(parent) = lock_path.parent()
            && !parent.as_os_str().is_empty()
        {
            std::fs::create_dir_all(parent)
                .map_err(|e| StorageError::Io(format!("create lock parent {parent:?}: {e}")))?;
        }
        let file = std::fs::OpenOptions::new()
            .create(true)
            .truncate(false)
            .write(true)
            .open(lock_path)
            .map_err(|e| StorageError::Io(format!("open lock {lock_path:?}: {e}")))?;
        // SAFETY: fd is valid; flock(2) has no preconditions beyond it.
        let rc = unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) };
        if rc != 0 {
            return Err(StorageError::Io(format!(
                "flock {lock_path:?}: {}",
                std::io::Error::last_os_error()
            )));
        }
        Ok(FileLock(file))
    }
}

#[cfg(unix)]
impl Drop for FileLock {
    fn drop(&mut self) {
        use std::os::unix::io::AsRawFd;

        // SAFETY: fd is still owned by self; the lock is released here and
        // again (idempotently) when the file closes.
        unsafe { libc::flock(self.0.as_raw_fd(), libc::LOCK_UN) };
    }
}

/// Off unix there is no blessed arbitration primitive; fail typed instead
/// of silently skipping cross-process serialization (#100).
#[cfg(not(unix))]
struct FileLock;

#[cfg(not(unix))]
impl FileLock {
    fn acquire(_lock_path: &Path) -> StorageResult<Self> {
        Err(StorageError::UnsupportedFormat(
            "cross-process file locking (with_file_lock) requires a POSIX platform".into(),
        ))
    }
}

/// The blessed recipe for a multi-process consumer's metadata
/// read-modify-write cycle (#100, review composition fix): the advisory
/// file lock ([`lock_file_for_metadata`]) is held across the **whole**
/// awaited commit-actor cycle — cross-process exclusion and in-process
/// actor serialization are both active for the duration of `cycle`.
///
/// The closure is async: a full load → mutate → publish cycle (including
/// any `save_*`-style actor work) runs inside, while independent processes
/// taking the same lock file serialize behind it. Lock acquisition runs on
/// the blocking pool (the flock can park on a competing holder); the lock
/// is released when the cycle's future completes.
///
/// For consumers whose RMW is entirely synchronous, [`with_file_lock`]
/// wraps the same lock file — but a sync closure cannot await the commit
/// actor; only this helper composes the two locks.
///
/// # Operational caution (blocking-pool waits)
///
/// The flock wait is unbounded and runs through `spawn_blocking`, which
/// Tokio documents for blocking work that is bounded and eventually
/// completes: a waiter that has already parked cannot be reliably
/// aborted, and many long-lived blocked waiters can exhaust the runtime's
/// blocking-thread capacity. This design is reasonable for metadata
/// commits when:
///
/// - commit cycles are short — the hold scope is a JSON load → mutate →
///   publish, not compute;
/// - contention is normally brief;
/// - callers do not hold the lock across lengthy compute, network I/O,
///   user interaction, or indefinite waits;
/// - shutdown behavior with a stuck lock holder is understood: blocked
///   `spawn_blocking` tasks are not aborted by task cancellation;
///   [`tokio::runtime::Runtime::shutdown_timeout`] abandons them after a
///   grace period while [`tokio::runtime::Runtime::shutdown_background`]
///   waits them out — and process exit always closes the descriptor,
///   releasing the flock.
///
/// If lock waits could be prolonged or numerous, prefer a dedicated
/// lock-management thread, an explicit timeout/cancellation strategy
/// (bounded wait before acquisition), or a storage system with
/// transactional coordination over unbounded `flock` waits here.
pub async fn with_metadata_file_lock<T, F, Fut>(
    metadata_path: &Path,
    cycle: F,
) -> StorageResult<T>
where
    F: FnOnce() -> Fut,
    Fut: std::future::Future<Output = StorageResult<T>>,
{
    let lock_path = lock_file_for_metadata(metadata_path);
    let lock_path2 = lock_path.clone();
    let _lock = tokio::task::spawn_blocking(move || FileLock::acquire(&lock_path2))
        .await
        .map_err(|e| StorageError::Io(format!("file lock task failed: {e}")))??;
    with_commit_actor(metadata_path, cycle).await
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
