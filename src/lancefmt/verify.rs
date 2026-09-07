//! #115: dataset verification inspection primitive (#104 Phase 2).
//!
//! A consumer `verify` cross-checks the stamped dataset schema — read via
//! [`read_schema`](super::read_schema), no column buffers decoded — against
//! the facts its registry descriptor asserts. Both sides share one grammar:
//! the writer stamps a fixed fact set per artifact class, and
//! [`DatasetFacts`] derives the expectation from the registry entry
//! following exactly that set:
//!
//! | artifact class        | dataset schema stamps                                   |
//! |-----------------------|---------------------------------------------------------|
//! | dense (legacy)        | none (registry shape is a computed fact)                |
//! | sparse                | `rows`, `cols`, `nnz`                                   |
//! | vectors / scalars     | `kind` = `vector-space`                                 |
//! | graph                 | `kind` = `graph`, `num_nodes`, `weight_type`, `node_id_width`, `weighted` |
//!
//! Count mismatches surface as [`StorageError::DimensionMismatch`]; type
//! and semantic mismatches as [`StorageError::Invalid`]. Every error names
//! the mismatched fact.

use std::collections::BTreeMap;
use std::path::Path;

use arrow::datatypes::Schema as ArrowSchema;

use crate::catalog::TableDescriptor;
use crate::metadata::{CollectionKind, FileInfo};
use crate::{StorageError, StorageResult};

/// Stamped dataset facts: the expectation set of a verification, or the
/// extracted fact set of an existing dataset. `None` = do not assert.
#[derive(Debug, Clone, Default, PartialEq, Eq)]
pub struct DatasetFacts {
    /// Dataset-level collection kind (`vector-space` / `graph`).
    pub kind: Option<String>,
    /// Legacy dense/sparse shape stamps.
    pub rows: Option<u64>,
    pub cols: Option<u64>,
    pub nnz: Option<u64>,
    /// Graph collection facts.
    pub num_nodes: Option<u64>,
    pub weight_type: Option<String>,
    pub node_id_width: Option<String>,
    pub weighted: Option<String>,
}

impl DatasetFacts {
    /// Extracts the facts a dataset stamps (inspection only: the schema
    /// footer, no column buffers).
    pub fn from_schema(schema: &ArrowSchema) -> Self {
        let md = schema.metadata();
        let num = |key: &str| md.get(key).and_then(|v| v.parse::<u64>().ok());
        Self {
            kind: md.get("kind").cloned(),
            rows: num("rows"),
            cols: num("cols"),
            nnz: num("nnz"),
            num_nodes: num("num_nodes"),
            weight_type: md.get("weight_type").cloned(),
            node_id_width: md.get("node_id_width").cloned(),
            weighted: md.get("weighted").cloned(),
        }
    }

    /// Expectations for a registry entry, asserting exactly what the
    /// writer of that artifact class stamps: sparse artifacts carry the
    /// shape stamps; RFC-#81 collections carry `kind` (graphs also carry
    /// the graph facts); dense datasets stamp nothing.
    pub fn from_file_info(info: &FileInfo) -> Self {
        Self::from_registry_parts(
            &info.filetype,
            info.kind,
            info.rows,
            info.cols,
            info.nnz,
            &info.properties,
        )
    }

    /// Expectations for a catalog descriptor (the [`TableDescriptor`]
    /// projection of a registry entry). Fact derivation follows the same
    /// per-filetype stamp contract as [`Self::from_file_info`]; the
    /// descriptor's typed `kind` is asserted only for artifact classes
    /// whose writer stamps it.
    pub fn from_descriptor(descriptor: &TableDescriptor) -> Self {
        let num = |key: &str| {
            descriptor
                .properties
                .get(key)
                .and_then(|v| v.parse::<u64>().ok())
        };
        let rows = num("rows").map(|v| v as usize);
        let cols = num("cols").map(|v| v as usize);
        let nnz = num("nnz").map(|v| v as usize);
        Self::from_registry_parts(
            &descriptor.properties["filetype"],
            Some(descriptor.kind),
            rows.unwrap_or(0),
            cols.unwrap_or(0),
            nnz,
            &descriptor.properties,
        )
    }

    fn from_registry_parts(
        filetype: &str,
        explicit_kind: Option<CollectionKind>,
        rows: usize,
        cols: usize,
        nnz: Option<usize>,
        properties: &BTreeMap<String, String>,
    ) -> Self {
        // `kind` is asserted only for artifact classes whose writer stamps
        // it: legacy datasets carry no dataset-level kind, so the
        // registry's filetype-inferred kind over-asserts there.
        let kind = match filetype {
            "vectors" | "vector" => Some(
                explicit_kind
                    .unwrap_or(CollectionKind::VectorSpace)
                    .as_str()
                    .to_string(),
            ),
            "graph" => Some(
                explicit_kind
                    .unwrap_or(CollectionKind::Graph)
                    .as_str()
                    .to_string(),
            ),
            _ => None,
        };
        // Shape stamps exist only on sparse artifacts (`to_sparse_record_batch`):
        // dense datasets stamp nothing, so their registry shape is a
        // computed fact, not a verifiable stamp.
        let shape = match filetype {
            "sparse" => (Some(rows as u64), Some(cols as u64)),
            _ => (None, None),
        };
        let num = |key: &str| properties.get(key).and_then(|v| v.parse::<u64>().ok());
        Self {
            kind,
            rows: shape.0,
            cols: shape.1,
            nnz: if filetype == "sparse" {
                nnz.map(|n| n as u64)
            } else {
                None
            },
            num_nodes: num("num_nodes"),
            weight_type: properties.get("weight_type").cloned(),
            node_id_width: properties.get("node_id_width").cloned(),
            weighted: properties.get("weighted").cloned(),
        }
    }
}

/// Verifies the stamped schema of a lance dataset at `dir` against
/// `expected` (#115): reads the schema footer only — no column buffers are
/// decoded — and returns a typed error naming the first mismatched fact.
pub fn verify_dataset_schema(dir: &Path, expected: &DatasetFacts) -> StorageResult<()> {
    let schema = super::read_schema(dir)?;
    let stamped = DatasetFacts::from_schema(&schema);
    verify_facts(dir, expected, &stamped)
}

fn verify_facts(dir: &Path, expected: &DatasetFacts, stamped: &DatasetFacts) -> StorageResult<()> {
    // Count facts: DimensionMismatch naming the fact on both sides.
    for (fact, want, have) in [
        ("rows", expected.rows, stamped.rows),
        ("cols", expected.cols, stamped.cols),
        ("nnz", expected.nnz, stamped.nnz),
        ("num_nodes", expected.num_nodes, stamped.num_nodes),
    ] {
        let Some(want) = want else { continue };
        match have {
            None => return Err(missing_stamp(dir, fact, want)),
            Some(have) if have != want => {
                return Err(StorageError::DimensionMismatch {
                    expected: format!("{fact}={want}"),
                    found: format!("{fact}={have}"),
                });
            }
            Some(_) => {}
        }
    }
    // Semantic stamps: Invalid naming the fact and both values.
    for (fact, want, have) in [
        ("kind", &expected.kind, &stamped.kind),
        ("weight_type", &expected.weight_type, &stamped.weight_type),
        (
            "node_id_width",
            &expected.node_id_width,
            &stamped.node_id_width,
        ),
        ("weighted", &expected.weighted, &stamped.weighted),
    ] {
        let Some(want) = want else { continue };
        match have {
            None => return Err(missing_stamp(dir, fact, want)),
            Some(have) if have != want => {
                return Err(StorageError::Invalid(format!(
                    "dataset {dir:?} stamps {fact} {have:?}, registry expects {want:?}"
                )));
            }
            Some(_) => {}
        }
    }
    Ok(())
}

fn missing_stamp(dir: &Path, fact: &str, want: impl std::fmt::Display) -> StorageError {
    StorageError::Invalid(format!(
        "dataset {dir:?} carries no {fact:?} stamp; registry expects {want}"
    ))
}
