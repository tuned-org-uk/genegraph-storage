//! # Transactional generations (genegraph-storage #93, RFC #81 phase P5).
//!
//! Artifact generations are **immutable**: every publish (build, append,
//! recompute) mints a fresh generation `{logical}__g{N}` — its artifacts are
//! written once and never overwritten — and commits by atomically publishing
//! the generation's metadata JSON (the single commit pointer, mirroring the
//! snapshot/consensus-index pattern of segmented logs).
//!
//! Invariants:
//! - A generation without a metadata file **was never committed**: readers
//!   and discovery ignore it; the sweep API sees it for garbage collection.
//! - The commit itself is a single atomic filesystem operation
//!   ([`write_json_atomic`]): readers observe either the previous file or
//!   the complete new one, never a partial write.
//! - Reader isolation (#97): pin acquisition (validation + registration) and
//!   the sweep's check-to-retirement transition are serialized by a
//!   per-generation state lock — a sweep never removes artifacts under a
//!   registered reader, and a pin never registers for a retired generation.
//! - `__g{digits}` at the end of an instance name is reserved.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex as StdMutex, OnceLock, Weak};

use crate::{StorageError, StorageResult};

/// Separator between a logical dataset name and its generation number.
pub const GENERATION_SEP: &str = "__g";

/// Logical dataset name: strips a trailing `__g{digits}` generation suffix.
pub fn logical_name(name: &str) -> &str {
    if let Some(pos) = name.rfind(GENERATION_SEP) {
        let suffix = &name[pos + GENERATION_SEP.len()..];
        if !suffix.is_empty() && suffix.bytes().all(|b| b.is_ascii_digit()) {
            return &name[..pos];
        }
    }
    name
}

/// Generation-qualified instance name: `{logical}__g{gen}`.
pub fn generation_name(logical: &str, generation: u64) -> String {
    format!("{logical}{GENERATION_SEP}{generation}")
}

/// Parse the generation number out of a gen-qualified instance name.
/// Returns `None` for plain logical names or malformed suffixes.
///
/// Note (#95): digit suffixes above `u64::MAX` parse to `None` here; the
/// scanners combine this with a `unwrap_or(u64::MAX)` fallback so such names
/// sort last consistently instead of crashing discovery.
pub fn parse_generation(name: &str) -> Option<u64> {
    let pos = name.rfind(GENERATION_SEP)?;
    let suffix = &name[pos + GENERATION_SEP.len()..];
    if suffix.is_empty() || !suffix.bytes().all(|b| b.is_ascii_digit()) {
        return None;
    }
    suffix.parse().ok()
}

/// A committed generation: its number and the metadata file that pins it.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct GenerationInfo {
    pub generation: u64,
    pub metadata_path: PathBuf,
}

/// Committed generations of `logical` under `base`, ascending by generation
/// number. A generation is committed iff its `{logical}__g{N}_metadata.json`
/// exists; pre-commit crash residue is invisible here.
pub async fn list_generations(base: &Path, logical: &str) -> StorageResult<Vec<GenerationInfo>> {
    let mut out = Vec::new();
    for (generation, metadata_path) in scan_generation_paths(base, logical).await? {
        if metadata_path.is_file() {
            out.push(GenerationInfo {
                generation,
                metadata_path,
            });
        }
    }
    out.sort_by_key(|g| g.generation);
    Ok(out)
}

/// Generation numbers with **any** artifact present under `base`, committed
/// or not (orphans from a pre-commit crash included), ascending.
pub async fn list_artifact_generations(base: &Path, logical: &str) -> StorageResult<Vec<u64>> {
    let mut gens = std::collections::BTreeSet::new();
    let mut rd = tokio::fs::read_dir(base)
        .await
        .map_err(|e| StorageError::Io(e.to_string()))?;
    let artifact_prefix = format!("{logical}{GENERATION_SEP}");
    while let Some(entry) = rd
        .next_entry()
        .await
        .map_err(|e| StorageError::Io(e.to_string()))?
    {
        let name = entry.file_name();
        let name = name.to_string_lossy();
        let Some(rest) = name.strip_prefix(&artifact_prefix) else {
            continue;
        };
        let Some(num) = rest.split('_').next() else {
            continue;
        };
        if !num.is_empty() && num.bytes().all(|b| b.is_ascii_digit()) {
            gens.insert(num.parse::<u64>().unwrap_or(u64::MAX));
        }
    }
    Ok(gens.into_iter().collect())
}

/// Delete every file of a generation — artifacts and, if present, the
/// metadata commit pointer. Prefix matching is exact on
/// `{logical}__g{gen}_`, so sibling datasets (`ds` vs `ds2`) are untouched.
/// Safe to call on orphaned generations (no metadata) and on missing ones.
///
/// Reader isolation (#97): the pin check through the *complete removal* runs
/// inside one per-generation state lock — the retirement transition is
/// atomic with respect to [`pin_generation`], which validates and registers
/// under that same lock. A pin attempt that races a sweep either precedes
/// the sweep's lock acquisition (the sweep then sees the pin and is refused
/// with [`StorageError::InvalidState`]) or waits for retirement to finish
/// (and then fails validation, because the commit pointer is gone). A sweep
/// can therefore never delete underneath a registered reader. Callers retry
/// a refused sweep once their readers have released.
pub async fn delete_generation(base: &Path, logical: &str, generation: u64) -> StorageResult<()> {
    let prefix = format!("{logical}{GENERATION_SEP}{generation}_");
    let metadata_path = base.join(format!("{prefix}metadata.json"));
    // Resolve (or mint) the generation's state lock up front, then run the
    // whole check-to-retirement critical section synchronously under it on
    // the blocking pool: std locks must not be held across awaits.
    let state = crate::commit::weak_lookup(
        &GENERATION_STATES,
        metadata_path.to_string_lossy().to_string(),
        Arc::new(StdMutex::new(GenerationState::default())),
    );
    sweep_gate(SWEEP_GATE_PRE_LOCK);
    let base = base.to_path_buf();
    let logical = logical.to_string();
    tokio::task::spawn_blocking(move || -> StorageResult<()> {
        let state = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if state.pins > 0 {
            return Err(StorageError::InvalidState(format!(
                "generation {generation} of '{logical}' is pinned by an in-flight reader; \
                 sweep refused until all readers release"
            )));
        }
        // Past the check and still under the lock: no pin can register for
        // this generation until the removals complete (test hook forces a
        // pin attempt exactly in this window).
        sweep_gate(SWEEP_GATE_POST_CHECK);
        let rd = std::fs::read_dir(&base)
            .map_err(|e| StorageError::Io(format!("read {base:?}: {e}")))?;
        let mut removed = Vec::new();
        for entry in rd {
            let entry = entry.map_err(|e| StorageError::Io(e.to_string()))?;
            if !entry.file_name().to_string_lossy().starts_with(&prefix) {
                continue;
            }
            removed.push(entry.path());
        }
        for path in removed {
            if path.is_dir() {
                std::fs::remove_dir_all(&path)
                    .map_err(|e| StorageError::Io(format!("remove {path:?}: {e}")))?;
            } else {
                std::fs::remove_file(&path)
                    .map_err(|e| StorageError::Io(format!("remove {path:?}: {e}")))?;
            }
        }
        Ok(())
    })
    .await
    .map_err(|e| StorageError::Io(format!("generation sweep task failed: {e}")))?
}

// ---------------------------------------------------------------------------
// Reader pins (#97)
// ---------------------------------------------------------------------------

/// Per-generation pin state; the lock guarding the state serializes pin
/// registration/validation against the sweep's retirement transition.
#[derive(Debug, Default)]
struct GenerationState {
    pins: usize,
}

/// Weak-valued registry of per-generation state locks (the #98 discipline):
/// entries live only while a pin or an in-flight sweep holds the `Arc`.
static GENERATION_STATES: OnceLock<StdMutex<HashMap<String, Weak<StdMutex<GenerationState>>>>> =
    OnceLock::new();

/// RAII pin on a committed generation (#97).
///
/// Acquired through [`pin_generation`]; while alive, it keeps one reference
/// on the generation's state so a concurrent [`delete_generation`] refuses
/// to touch it. Dropping the guard releases the reference.
#[derive(Debug)]
pub struct GenerationGuard {
    state: Arc<StdMutex<GenerationState>>,
}

impl Drop for GenerationGuard {
    fn drop(&mut self) {
        let mut state = self
            .state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        state.pins -= 1;
    }
}

/// Pins a committed generation for the duration of a read (#97).
///
/// Registration is refcounted per generation: several readers may pin the
/// same generation concurrently and all succeed; a sweep is refused while
/// any pin is alive. Validation (commit pointer exists) and registration
/// happen atomically with respect to [`delete_generation`]'s retirement
/// transition — both hold the generation's state lock, so a racing sweep
/// either observes this pin (and is refused) or has already retired the
/// generation (and this call fails validation).
///
/// Fails with [`StorageError::Invalid`] if the generation's metadata file
/// (the commit pointer) does not exist — orphaned generations cannot be
/// pinned. Pin and sweep must address the generation through the same
/// `base` path: state is keyed by the metadata file path.
pub fn pin_generation(info: &GenerationInfo) -> StorageResult<GenerationGuard> {
    let state = crate::commit::weak_lookup(
        &GENERATION_STATES,
        info.metadata_path.to_string_lossy().to_string(),
        Arc::new(StdMutex::new(GenerationState::default())),
    );
    {
        let mut guard = state
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        // Validation under the state lock: a concurrent sweep either
        // finished its retirement (metadata gone -> rejected here) or waits
        // for this pin to register (and is then refused).
        if !info.metadata_path.is_file() {
            return Err(StorageError::Invalid(format!(
                "generation {} is not committed (no metadata at {:?}); \
                 only committed generations can be pinned",
                info.generation, info.metadata_path
            )));
        }
        guard.pins += 1;
    }
    Ok(GenerationGuard { state })
}

/// Atomic JSON publish: write to a unique `{path}.tmp`, fsync, rename over
/// `path`.
///
/// The rename is the single commit point (POSIX-atomic within a directory):
/// concurrent readers observe either the previous file or the complete new
/// one, never a truncated write. This is the ONLY sanctioned way to publish
/// a metadata file. The tmp name is uuid-unique (#95) so concurrent
/// publishers of the same path cannot corrupt each other's staging file;
/// the last rename wins, which makes last-writer-wins explicit instead of
/// interleaved.
pub fn write_json_atomic(path: &Path, contents: &str) -> StorageResult<()> {
    use std::io::Write;

    let tmp = path.with_extension(format!("json.tmp.{}", uuid::Uuid::new_v4().simple()));
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent).map_err(|e| StorageError::Io(e.to_string()))?;
    }
    {
        let mut f = std::fs::File::create(&tmp).map_err(|e| StorageError::Io(e.to_string()))?;
        f.write_all(contents.as_bytes())
            .map_err(|e| StorageError::Io(e.to_string()))?;
        f.sync_all().map_err(|e| StorageError::Io(e.to_string()))?;
    }
    std::fs::rename(&tmp, path).map_err(|e| StorageError::Io(e.to_string()))?;
    Ok(())
}

/// `(generation, metadata_path)` pairs implied by `{logical}__g{N}_metadata.json`
/// names under `base` (the file itself may or may not exist).
async fn scan_generation_paths(base: &Path, logical: &str) -> StorageResult<Vec<(u64, PathBuf)>> {
    let mut out = Vec::new();
    let mut rd = tokio::fs::read_dir(base)
        .await
        .map_err(|e| StorageError::Io(e.to_string()))?;
    let prefix = format!("{logical}{GENERATION_SEP}");
    while let Some(entry) = rd
        .next_entry()
        .await
        .map_err(|e| StorageError::Io(e.to_string()))?
    {
        let name = entry.file_name();
        let name = name.to_string_lossy().into_owned();
        let Some(rest) = name.strip_prefix(&prefix) else {
            continue;
        };
        let Some(meta) = rest.strip_suffix("_metadata.json") else {
            continue;
        };
        if !meta.is_empty() && meta.bytes().all(|b| b.is_ascii_digit()) {
            out.push((meta.parse().unwrap_or(u64::MAX), entry.path()));
        }
    }
    Ok(out)
}

// ---------------------------------------------------------------------------
// Test-only sweep gates (PR #99 review): force deterministic interleavings
// of pin attempts against the sweep's critical section.
// ---------------------------------------------------------------------------

pub(crate) const SWEEP_GATE_PRE_LOCK: usize = 0;
pub(crate) const SWEEP_GATE_POST_CHECK: usize = 1;

// Non-test builds compile the gate calls away entirely.
#[cfg(not(test))]
fn sweep_gate(_stage: usize) {}

#[cfg(test)]
fn sweep_gates() -> &'static StdMutex<[Option<SweepGate>; 2]> {
    static GATES: OnceLock<StdMutex<[Option<SweepGate>; 2]>> = OnceLock::new();
    GATES.get_or_init(|| StdMutex::new([None, None]))
}

/// A gate the sweep must pass through: it signals `arrived` (so tests know
/// the sweep is parked exactly at the stage) and blocks until `release`.
#[cfg(test)]
struct SweepGate {
    arrived: std::sync::mpsc::Sender<()>,
    release: std::sync::mpsc::Receiver<()>,
}

/// Arms a gate: the next sweep to reach `stage` signals the returned
/// receiver and parks until the returned sender fires. Fires once.
#[cfg(test)]
pub(crate) fn arm_sweep_gate(
    stage: usize,
) -> (std::sync::mpsc::Receiver<()>, std::sync::mpsc::Sender<()>) {
    let (arrived_tx, arrived_rx) = std::sync::mpsc::channel();
    let (release_tx, release_rx) = std::sync::mpsc::channel();
    let mut gates = sweep_gates()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    gates[stage] = Some(SweepGate {
        arrived: arrived_tx,
        release: release_rx,
    });
    (arrived_rx, release_tx)
}

/// Blocks the sweep at `stage` if a gate is armed there (consumes it).
#[cfg(test)]
fn sweep_gate(stage: usize) {
    let gate = sweep_gates()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get_mut(stage)
        .unwrap()
        .take();
    if let Some(gate) = gate {
        // park: the test now knows exactly where the sweep is
        let _ = gate.arrived.send(());
        let _ = gate.release.recv();
    }
}
