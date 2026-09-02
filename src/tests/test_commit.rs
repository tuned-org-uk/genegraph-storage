//! #100: commit serialization exposed to downstream consumers.
//!
//! Consumers that run their **own** metadata read-modify-write cycles (outside
//! the `save_*` registry paths) must reach the same serialization the internal
//! registry paths use, plus a blessed cross-process arbitration convention.

use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use crate::commit::{lock_file_for_metadata, with_commit_actor, with_file_lock};
use crate::generations::write_json_atomic;

use super::tmp_dir;

/// Two concurrent metadata read-modify-write cycles on the same path (the
/// shape every downstream consumer's own registry write takes) are
/// serialized by the commit actor: the second cycle observes the first's
/// write, so no update is lost.
#[tokio::test(flavor = "multi_thread")]
async fn commit_actor_serializes_concurrent_read_modify_write_cycles() {
    let dir = tmp_dir("commit_actor_no_lost_update").await;
    let md = dir.join("ds__g1_metadata.json");
    write_json_atomic(&md, "0").unwrap();

    // Each cycle reads a shared counter, then publishes counter+1. Each
    // cycle's observation is recorded in its own atomic: under
    // serialization the second cycle must see the first's write; interleaved,
    // both would see 0.
    let seen_a = std::sync::Arc::new(AtomicU8::new(0));
    let seen_b = std::sync::Arc::new(AtomicU8::new(0));
    let md_a = md.clone();
    let md_a_ref = md_a.clone();
    let seen_a_for_a = seen_a.clone();
    let a = with_commit_actor(&md_a_ref, move || async move {
        let n: u8 = std::fs::read_to_string(&md_a).unwrap().trim().parse().unwrap();
        seen_a_for_a.store(n + 1, Ordering::SeqCst);
        write_json_atomic(&md_a, &(n + 1).to_string()).unwrap();
        Ok(())
    });
    let md_b = md.clone();
    let md_b_ref = md_b.clone();
    let seen_b_for_b = seen_b.clone();
    let b = with_commit_actor(&md_b_ref, move || async move {
        let n: u8 = std::fs::read_to_string(&md_b).unwrap().trim().parse().unwrap();
        seen_b_for_b.store(n + 11, Ordering::SeqCst);
        write_json_atomic(&md_b, &(n + 1).to_string()).unwrap();
        Ok(())
    });
    let (ra, rb) = tokio::join!(a, b);
    ra.unwrap();
    rb.unwrap();

    // Valid orders: A first (a saw 0 -> 1, b saw A's write -> 12) or B
    // first (b saw 0 -> 11, a saw B's write -> 2). An interleaved (lost
    // update) pair would surface as both cycles observing 0: (1, 11).
    let a_seen = seen_a.load(Ordering::SeqCst);
    let b_seen = seen_b.load(Ordering::SeqCst);
    assert!(
        (a_seen, b_seen) == (1, 12) || (a_seen, b_seen) == (2, 11),
        "cycles must observe strictly ordered states, got a={a_seen}, b={b_seen}"
    );
    let final_count: u8 = std::fs::read_to_string(&md).unwrap().trim().parse().unwrap();
    assert_eq!(final_count, 2, "concurrent RMW cycles must not lose updates");
}

/// The blessed cross-process convention: `{metadata-stem}.lock` next to the
/// metadata file.
#[test]
fn lock_file_convention_is_stem_dot_lock() {
    assert_eq!(
        lock_file_for_metadata(Path::new("/base/ds__g1_metadata.json")),
        PathBuf::from("/base/ds__g1_metadata.lock")
    );
    // no extension: the full name is the stem
    assert_eq!(
        lock_file_for_metadata(Path::new("/base/registry")),
        PathBuf::from("/base/registry.lock")
    );
}

use std::path::Path;

/// #100: the advisory file lock is a real cross-process arbitration
/// primitive — a second holder (here: a second thread, its own flock open
/// file description, exactly what another process would see) is excluded
/// until the first releases, and the lock file is created on demand.
#[test]
fn file_lock_excludes_second_holder_until_released() {
    let base = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(tmp_dir("file_lock_exclusion"));
    let lock = base.join("ds__g1_metadata.lock");

    let (arrived_tx, arrived_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let lock_a = lock.clone();
    let a = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                with_file_lock(&lock_a, move || {
                    arrived_tx.send(()).unwrap();
                    // hold the lock until the test releases it
                    release_rx.recv().unwrap();
                    Ok("first")
                })
                .await
                .unwrap()
            })
    });
    arrived_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("first holder must acquire the lock");

    // The second holder must stay excluded while the first holds the lock.
    let (entered_tx, entered_rx) = mpsc::channel::<()>();
    let lock_b = lock.clone();
    let b = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                with_file_lock(&lock_b, move || {
                    entered_tx.send(()).unwrap();
                    Ok(())
                })
                .await
                .unwrap()
            })
    });
    assert!(
        entered_rx.recv_timeout(Duration::from_millis(300)).is_err(),
        "second holder must block while the first holds the lock"
    );

    // Release; the waiter proceeds.
    release_tx.send(()).unwrap();
    assert_eq!(a.join().unwrap(), "first", "first holder completes");
    entered_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("second holder must acquire after release");
    b.join().unwrap();

    // The lock file is created on demand and left in place (it is the
    // rendezvous point, not a commit artifact).
    assert!(lock.is_file(), "lock file must exist at {lock:?}");

    let _ = std::fs::remove_dir_all(&base);
}

/// The lock file's parent directory is created on demand, so a consumer can
/// take the lock before creating the dataset directory itself.
#[test]
fn file_lock_creates_missing_parent_dirs() {
    let base = tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(tmp_dir("file_lock_fresh_parent"));
    let lock = base.join("fresh").join("nested").join("ds.lock");
    tokio::runtime::Builder::new_current_thread()
        .build()
        .unwrap()
        .block_on(async { with_file_lock(&lock, || Ok(())).await })
        .unwrap();
    assert!(lock.is_file());
    let _ = std::fs::remove_dir_all(&base);
}
