//! #100: commit serialization exposed to downstream consumers.
//!
//! Consumers that run their **own** metadata read-modify-write cycles (outside
//! the `save_*` registry paths) must reach the same serialization the internal
//! registry paths use, plus a blessed cross-process arbitration convention.

use std::path::PathBuf;
use std::sync::mpsc;
use std::sync::atomic::{AtomicU8, Ordering};
use std::time::Duration;

use crate::commit::{
    lock_file_for_metadata, try_with_file_lock, try_with_metadata_file_lock, with_commit_actor,
    with_file_lock, with_metadata_file_lock,
};
use crate::generations::write_json_atomic;
use crate::StorageError;

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

/// The composed helper (review feedback on #100) holds the advisory file
/// lock across the whole awaited actor cycle: while a cycle is parked
/// mid-RMW, another process-shaped lock holder stays excluded, and
/// concurrent cycles on the same metadata file serialize (both increments
/// land).
#[tokio::test(flavor = "multi_thread")]
async fn metadata_file_lock_holds_flock_across_the_actor_cycle() {
    let base = tmp_dir("composed_lock_cycle").await;
    let md = base.join("ds__g1_metadata.json");
    write_json_atomic(&md, "0").unwrap();

    // Park the composed cycle mid-RMW; a foreign flock holder must stay
    // excluded until the cycle completes.
    let (entered_tx, entered_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let md_a = md.clone();
    let md_a_ref = md_a.clone();
    let cycle_a = tokio::spawn(async move {
        with_metadata_file_lock(&md_a_ref, move || async move {
            entered_tx.send(()).unwrap();
            release_rx.recv().unwrap();
            let n: u8 = std::fs::read_to_string(&md_a).unwrap().trim().parse().unwrap();
            write_json_atomic(&md_a, &(n + 1).to_string()).unwrap();
            Ok(())
        })
        .await
        .unwrap();
    });
    entered_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("composed cycle must start");

    let (entered_b_tx, entered_b_rx) = mpsc::channel::<()>();
    let lock_b = lock_file_for_metadata(&md);
    let holder_b = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                with_file_lock(&lock_b, move || {
                    entered_b_tx.send(()).unwrap();
                    Ok(())
                })
                .await
                .unwrap()
            })
    });
    assert!(
        entered_b_rx.recv_timeout(Duration::from_millis(300)).is_err(),
        "the flock must be held across the awaited cycle"
    );

    // Release; the cycle completes its write and the waiter enters.
    release_tx.send(()).unwrap();
    cycle_a.await.unwrap();
    entered_b_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("waiter must acquire after the cycle releases");
    holder_b.join().unwrap();

    let _ = std::fs::remove_dir_all(&base);
}

/// Two concurrent composed cycles on the same metadata file serialize end
/// to end — flock first, then actor — so both increments land.
#[tokio::test(flavor = "multi_thread")]
async fn metadata_file_lock_serializes_concurrent_cycles() {
    let base = tmp_dir("composed_lock_concurrent").await;
    let md = base.join("ds__g1_metadata.json");
    write_json_atomic(&md, "0").unwrap();

    let run = |md: PathBuf| async move {
        let md_inner = md.clone();
        with_metadata_file_lock(&md, move || async move {
            let n: u8 = std::fs::read_to_string(&md_inner)
                .unwrap()
                .trim()
                .parse()
                .unwrap();
            write_json_atomic(&md_inner, &(n + 1).to_string()).unwrap();
            Ok(())
        })
        .await
        .unwrap();
    };
    let (ra, rb) = tokio::join!(run(md.clone()), run(md.clone()));
    let _ = (ra, rb);

    let final_count: u8 = std::fs::read_to_string(&md).unwrap().trim().parse().unwrap();
    assert_eq!(final_count, 2, "concurrent composed cycles must not lose updates");

    let _ = std::fs::remove_dir_all(&base);
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

/// #105: the try variant runs the closure when uncontended, releases the
/// lock on completion (a subsequent try acquires again), and the lock file
/// stays in place as the rendezvous point.
#[tokio::test(flavor = "multi_thread")]
async fn try_file_lock_runs_closure_uncontended_and_releases() {
    let base = tmp_dir("try_file_lock_uncontended").await;
    let lock = base.join("ds__g1_metadata.lock");

    let out = try_with_file_lock(&lock, || Ok::<_, StorageError>("ran")).await.unwrap();
    assert_eq!(out, "ran", "closure output must pass through");

    try_with_file_lock(&lock, || Ok::<_, StorageError>(())).await.unwrap();
    assert!(lock.is_file(), "lock file is the rendezvous point, left in place");

    let _ = std::fs::remove_dir_all(&base);
}

/// #105: the try variant fails fast on contention — while a holder parks
/// on the lock file, a second taker returns immediately with a
/// distinctly matchable `LockWouldBlock` naming the lock file, instead of
/// parking on the blocking pool.
#[tokio::test(flavor = "multi_thread")]
async fn try_file_lock_fails_fast_naming_lock_file_while_held() {
    let base = tmp_dir("try_file_lock_contended").await;
    let lock = base.join("ds__g1_metadata.lock");

    let (held_tx, held_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let lock_a = lock.clone();
    let holder = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                with_file_lock(&lock_a, move || {
                    held_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                })
                .await
                .unwrap()
            })
    });
    held_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("holder must acquire the lock");

    let started = std::time::Instant::now();
    let err = try_with_file_lock(&lock, || Ok::<_, StorageError>(())).await.unwrap_err();
    assert!(
        started.elapsed() < Duration::from_secs(2),
        "try variant must fail fast, not wait for the holder"
    );
    match err {
        StorageError::LockWouldBlock { path } => {
            assert_eq!(path, lock, "the error must name the lock file");
        }
        other => panic!("expected LockWouldBlock, got {other:?}"),
    }

    // After the holder releases, the try succeeds — the guard was RAII.
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    try_with_file_lock(&lock, || Ok::<_, StorageError>(())).await.unwrap();

    let _ = std::fs::remove_dir_all(&base);
}

/// #105: the composed try helper resolves the lock through the blessed
/// convention and fails fast, naming the derived lock file, when another
/// holder has it; uncontended it runs the whole actor cycle.
#[tokio::test(flavor = "multi_thread")]
async fn try_metadata_file_lock_fails_fast_then_runs_cycle_uncontended() {
    let base = tmp_dir("try_composed_lock").await;
    let md = base.join("ds__g1_metadata.json");
    write_json_atomic(&md, "0").unwrap();
    let lock = lock_file_for_metadata(&md);

    // A foreign holder parks on the derived lock file.
    let (held_tx, held_rx) = mpsc::channel::<()>();
    let (release_tx, release_rx) = mpsc::channel::<()>();
    let lock_a = lock.clone();
    let holder = std::thread::spawn(move || {
        tokio::runtime::Builder::new_current_thread()
            .build()
            .unwrap()
            .block_on(async {
                with_file_lock(&lock_a, move || {
                    held_tx.send(()).unwrap();
                    release_rx.recv().unwrap();
                    Ok(())
                })
                .await
                .unwrap()
            })
    });
    held_rx
        .recv_timeout(Duration::from_secs(5))
        .expect("holder must acquire the derived lock file");

    let err = try_with_metadata_file_lock(&md, || async { Ok(()) }).await.unwrap_err();
    match err {
        StorageError::LockWouldBlock { path } => {
            assert_eq!(path, lock, "must name the derived lock file");
        }
        other => panic!("expected LockWouldBlock, got {other:?}"),
    }

    // Uncontended, the full RMW cycle runs under the actor and lands.
    release_tx.send(()).unwrap();
    holder.join().unwrap();
    let md_cycle = md.clone();
    try_with_metadata_file_lock(&md, move || async move {
        let n: u8 = std::fs::read_to_string(&md_cycle).unwrap().trim().parse().unwrap();
        write_json_atomic(&md_cycle, &(n + 1).to_string()).unwrap();
        Ok(())
    })
    .await
    .unwrap();
    let count: u8 = std::fs::read_to_string(&md).unwrap().trim().parse().unwrap();
    assert_eq!(count, 1, "the uncontended try cycle must publish its RMW");

    let _ = std::fs::remove_dir_all(&base);
}
