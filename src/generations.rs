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
//! - `__g{digits}` at the end of an instance name is reserved.

use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::{Mutex as StdMutex, OnceLock};

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
/// Reader isolation (#97): if any [`GenerationGuard`] pin is alive for this
/// generation, the sweep is refused with
/// [`StorageError::InvalidState`] — it never removes artifacts under an
/// in-flight reader. Callers retry once their readers have released.
pub async fn delete_generation(base: &Path, logical: &str, generation: u64) -> StorageResult<()> {
    let prefix = format!("{logical}{GENERATION_SEP}{generation}_");
    let metadata_path = base.join(format!("{prefix}metadata.json"));
    if generation_pin_count(&metadata_path) > 0 {
        return Err(StorageError::InvalidState(format!(
            "generation {generation} of '{logical}' is pinned by an in-flight reader; \
             sweep refused until all readers release"
        )));
    }
    let mut rd = tokio::fs::read_dir(base)
        .await
        .map_err(|e| StorageError::Io(e.to_string()))?;
    while let Some(entry) = rd
        .next_entry()
        .await
        .map_err(|e| StorageError::Io(e.to_string()))?
    {
        if !entry.file_name().to_string_lossy().starts_with(&prefix) {
            continue;
        }
        let path = entry.path();
        if path.is_dir() {
            tokio::fs::remove_dir_all(&path)
                .await
                .map_err(|e| StorageError::Io(e.to_string()))?;
        } else {
            tokio::fs::remove_file(&path)
                .await
                .map_err(|e| StorageError::Io(e.to_string()))?;
        }
    }
    Ok(())
}

// ---------------------------------------------------------------------------
// Reader pins (#97)
// ---------------------------------------------------------------------------

/// Process-wide pin registry: metadata-path → live pin count. Counts hit
/// zero only transiently (the dropping guard removes the entry), so the
/// map stays bounded by the number of pinned generations.
fn pin_registry() -> &'static StdMutex<HashMap<String, usize>> {
    static PINS: OnceLock<StdMutex<HashMap<String, usize>>> = OnceLock::new();
    PINS.get_or_init(|| StdMutex::new(HashMap::new()))
}

/// RAII pin on a committed generation (#97).
///
/// Acquired through [`pin_generation`]; while alive, it keeps one reference
/// on the generation so a concurrent [`delete_generation`] refuses to touch
/// it. Dropping the guard releases the reference; the last drop for a
/// generation unregisters it.
#[derive(Debug)]
pub struct GenerationGuard {
    key: String,
}

impl Drop for GenerationGuard {
    fn drop(&mut self) {
        let mut pins = pin_registry()
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner());
        if let Some(count) = pins.get_mut(&self.key) {
            *count -= 1;
            if *count == 0 {
                pins.remove(&self.key);
            }
        }
    }
}

/// Pins a committed generation for the duration of a read (#97).
///
/// Registration is process-wide and refcounted: several readers may pin the
/// same generation concurrently and all succeed; a sweep is refused while
/// any pin is alive. Fails with [`StorageError::Invalid`] if the
/// generation's metadata file (the commit pointer) does not exist —
/// orphaned generations cannot be pinned.
///
/// Pin and sweep must address the generation through the same `base` path:
/// the registry is keyed by the metadata file path.
pub fn pin_generation(info: &GenerationInfo) -> StorageResult<GenerationGuard> {
    if !info.metadata_path.is_file() {
        return Err(StorageError::Invalid(format!(
            "generation {} is not committed (no metadata at {:?}); \
             only committed generations can be pinned",
            info.generation, info.metadata_path
        )));
    }
    let key = info.metadata_path.to_string_lossy().to_string();
    let mut pins = pin_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner());
    *pins.entry(key.clone()).or_insert(0) += 1;
    Ok(GenerationGuard { key })
}

/// Live pin count for a generation, keyed by its metadata file path.
fn generation_pin_count(metadata_path: &Path) -> usize {
    let key = metadata_path.to_string_lossy().to_string();
    pin_registry()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .get(&key)
        .copied()
        .unwrap_or(0)
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
