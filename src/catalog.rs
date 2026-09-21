//! M-C1: catalog contract for table discovery and registration (#75).
//!
//! The [`TableDescriptor`] shape and the [`Catalog`] operations mirror the
//! **Lance Namespace** client spec whose Apache Polaris implementation maps
//! onto Polaris' **Generic Table API** (`name`, `format`, `base-location`,
//! `properties`); see the Polaris + Lance integration announcement
//! (2026-01-06). Per decision D1 in #75 this is a standards-hygiene
//! contract only: no catalog server client is provided here.
//!
//! [`LocalRegistry`] implements the contract over the existing JSON metadata
//! registry (a `GeneMetadata` instance), mirroring Lance Namespace's
//! *Directory* semantics: every `*.lance` dataset registered in the
//! metadata's files map is a table.

use std::collections::BTreeMap;
use std::path::PathBuf;

use crate::StorageError;
use crate::StorageResult;
use crate::graph::GraphSource;
pub use crate::metadata::CollectionKind;
use crate::metadata::{FileInfo, GeneMetadata};

/// Computed descriptor properties that mirror `FileInfo` fields; user
/// properties cannot shadow them (enforced both in `register_table` and on
/// the direct `save_*_with` write paths).
pub(crate) const RESERVED_PROPERTIES: [&str; 6] = [
    "filetype",
    "storage_format",
    "rows",
    "cols",
    "nnz",
    "size_bytes",
];

/// Generic Table API-compatible table descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableDescriptor {
    /// Unique table name within the registry (the metadata files-map key).
    pub name: String,
    /// Table format; always `lance` for this crate.
    pub format: String,
    /// Location of the table root (a `*.lance` dataset directory).
    pub base_location: PathBuf,
    /// First-class collection kind (RFC #81-P1): `vector-space`, `graph` or
    /// `table`.
    pub kind: CollectionKind,
    /// Free-form key/value properties (filetype, shape, ...), merged with
    /// the collection's user properties (RFC #81-P1).
    pub properties: BTreeMap<String, String>,
}

/// Staleness of the graph linked to a vector space (#140), computed at
/// describe time from current registry facts. Never persisted: there is
/// no latch to clear, the signal follows the data on every call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GraphStaleness {
    /// The vector space has no `graph` property: nothing to assess.
    Unlinked,
    /// Lineage (`source`/`source_rows`) is present and matches this
    /// vector space: the graph was rebuilt against the current rows.
    Fresh,
    /// Detectable divergence: the lineage names another collection, the
    /// lineage row count differs from the current row count, or (without
    /// lineage) the stamped `num_nodes` differs from the row count.
    Stale,
    /// No lineage and `num_nodes` equals the row count. The counts agree;
    /// the content may still differ. The library does not guess.
    Unknown,
}

/// Vector-space descriptor plus its linked graph collection (RFC #81-P4).
///
/// A vector space references a graph collection by name through its `graph`
/// user property; [`Catalog::describe_vector_space`] resolves both in one
/// call.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VectorSpaceDescriptor {
    /// The vector-space collection itself.
    pub vectors: TableDescriptor,
    /// The linked graph collection, if the `graph` property is set.
    pub graph: Option<TableDescriptor>,
    /// Whether the linked graph still describes the current vectors
    /// (#140). `Unlinked` when no graph is linked.
    pub graph_staleness: GraphStaleness,
}

/// Registry of Lance tables backing a storage instance.
pub trait Catalog {
    /// All tables known to this registry.
    fn list_tables(&self) -> StorageResult<Vec<TableDescriptor>>;
    /// Whether a table with `name` is registered.
    fn table_exists(&self, name: &str) -> StorageResult<bool>;
    /// Descriptor for `name`, or `StorageError::Invalid` if absent.
    fn describe_table(&self, name: &str) -> StorageResult<TableDescriptor>;
    /// Adds `table` to the registry (replaces an existing entry of the same
    /// name, mirroring Generic Table create-or-replace semantics).
    fn register_table(&mut self, table: TableDescriptor) -> StorageResult<()>;
    /// Removes `name` from the registry without deleting the dataset.
    fn deregister_table(&mut self, name: &str) -> StorageResult<()>;

    /// Describes a vector-space collection together with the graph
    /// collection it references through the `graph` user property
    /// (RFC #81-P4). Fails with [`StorageError::Invalid`] if `name` is not a
    /// vector-space collection or the linked graph is not registered.
    ///
    /// The descriptor reports whether the linked graph still describes the
    /// current vectors ([`GraphStaleness`], #140), computed from current
    /// registry facts only.
    fn describe_vector_space(&self, name: &str) -> StorageResult<VectorSpaceDescriptor> {
        let vectors = self.describe_table(name)?;
        if vectors.kind != CollectionKind::VectorSpace {
            return Err(StorageError::Invalid(format!(
                "collection '{name}' has kind '{}', not a vector space",
                vectors.kind.as_str()
            )));
        }
        let graph = match vectors.properties.get("graph") {
            Some(graph_name) => Some(self.describe_table(graph_name)?),
            None => None,
        };
        let graph_staleness = match &graph {
            None => GraphStaleness::Unlinked,
            Some(g) => {
                let rows = vectors
                    .properties
                    .get("rows")
                    .and_then(|v| v.parse::<usize>().ok());
                match (g.properties.get("source"), g.properties.get("source_rows")) {
                    // Declared lineage is authoritative when present.
                    (Some(source), Some(source_rows)) => {
                        if source == &vectors.name && source_rows.parse::<usize>().ok() == rows {
                            GraphStaleness::Fresh
                        } else {
                            GraphStaleness::Stale
                        }
                    }
                    // Without lineage only a num_nodes/rows divergence is
                    // detectable; an agreement proves nothing about content.
                    _ => {
                        match (
                            g.properties
                                .get("num_nodes")
                                .and_then(|v| v.parse::<usize>().ok()),
                            rows,
                        ) {
                            (Some(num_nodes), Some(rows)) if num_nodes != rows => {
                                GraphStaleness::Stale
                            }
                            _ => GraphStaleness::Unknown,
                        }
                    }
                }
            }
        };
        Ok(VectorSpaceDescriptor {
            vectors,
            graph,
            graph_staleness,
        })
    }

    /// Resolves a graph source against the registry (#140): the source
    /// must be a registered vector-space collection. Feed the result to
    /// [`crate::graph::GraphWriteOptions::source`]; the write paths stamp
    /// it as the reserved `source`/`source_rows` metadata, and
    /// [`Self::describe_vector_space`] turns the stamp into the
    /// staleness signal. Lineage reasoning lives here, in the catalog —
    /// storage write paths only persist declared facts.
    fn graph_source(&self, name: &str) -> StorageResult<GraphSource> {
        let table = self.describe_table(name)?;
        if table.kind != CollectionKind::VectorSpace {
            return Err(StorageError::Invalid(format!(
                "graph source '{name}' has kind '{}', not a vector space",
                table.kind.as_str()
            )));
        }
        let rows = table
            .properties
            .get("rows")
            .and_then(|v| v.parse::<usize>().ok())
            .ok_or_else(|| {
                StorageError::Invalid(format!(
                    "graph source '{name}' carries no usable 'rows' fact"
                ))
            })?;
        Ok(GraphSource {
            name: table.name,
            rows,
        })
    }
}

/// [`Catalog`] implementation over the existing JSON metadata registry.
///
/// The registry wraps an owned [`GeneMetadata`]; mutate through the
/// [`Catalog`] methods, then persist the instance with
/// `StorageBackend::save_metadata` (or take it back with
/// [`LocalRegistry::into_metadata`]).
#[derive(Debug, Clone)]
pub struct LocalRegistry {
    metadata: GeneMetadata,
    base: PathBuf,
}

impl LocalRegistry {
    /// Wraps `metadata`, resolving table locations under `base`.
    pub fn new(metadata: GeneMetadata, base: PathBuf) -> Self {
        Self { metadata, base }
    }

    /// The wrapped metadata (for persistence).
    pub fn metadata(&self) -> &GeneMetadata {
        &self.metadata
    }

    /// Consumes the registry, returning the mutated metadata.
    pub fn into_metadata(self) -> GeneMetadata {
        self.metadata
    }

    fn descriptor(&self, key: &str, info: &FileInfo) -> TableDescriptor {
        let kind = info
            .kind
            .or_else(|| CollectionKind::for_filetype(&info.filetype))
            .unwrap_or(CollectionKind::Table);
        let mut properties = BTreeMap::new();
        properties.insert("filetype".to_string(), info.filetype.clone());
        properties.insert("storage_format".to_string(), info.storage_format.clone());
        properties.insert("rows".to_string(), info.rows.to_string());
        properties.insert("cols".to_string(), info.cols.to_string());
        if let Some(nnz) = info.nnz {
            properties.insert("nnz".to_string(), nnz.to_string());
        }
        if let Some(size) = info.size_bytes {
            properties.insert("size_bytes".to_string(), size.to_string());
        }
        // User properties overlay the computed ones; `kind` is always the
        // authoritative typed value.
        for (k, v) in &info.properties {
            properties.insert(k.clone(), v.clone());
        }
        properties.insert("kind".to_string(), kind.as_str().to_string());
        TableDescriptor {
            name: key.to_string(),
            format: "lance".to_string(),
            base_location: self.base.join(&info.filename),
            kind,
            properties,
        }
    }
}

impl Catalog for LocalRegistry {
    fn list_tables(&self) -> StorageResult<Vec<TableDescriptor>> {
        let mut tables: Vec<TableDescriptor> = self
            .metadata
            .files
            .iter()
            .map(|(key, info)| self.descriptor(key, info))
            .collect();
        tables.sort_by(|a, b| a.name.cmp(&b.name));
        Ok(tables)
    }

    fn table_exists(&self, name: &str) -> StorageResult<bool> {
        Ok(self.metadata.files.contains_key(name))
    }

    fn describe_table(&self, name: &str) -> StorageResult<TableDescriptor> {
        let info =
            self.metadata.files.get(name).ok_or_else(|| {
                StorageError::Invalid(format!("table '{name}' is not registered"))
            })?;
        Ok(self.descriptor(name, info))
    }

    fn register_table(&mut self, table: TableDescriptor) -> StorageResult<()> {
        if table.format != "lance" {
            return Err(StorageError::UnsupportedFormat(format!(
                "unsupported table format '{}'",
                table.format
            )));
        }
        let prop = |key: &str| table.properties.get(key).cloned();
        let parse = |key: &str| -> StorageResult<Option<usize>> {
            prop(key)
                .map(|v| {
                    v.parse::<usize>().map_err(|_| {
                        StorageError::Invalid(format!(
                            "property '{key}' = '{v}' is not a valid integer"
                        ))
                    })
                })
                .transpose()
        };
        // RFC #81-P1: an explicit `kind` property wins; otherwise the typed
        // `kind` field is authoritative. Unknown kinds and disagreements
        // between the two are rejected (review PR #96) — silently picking
        // one side would let a property contradict the descriptor.
        let kind = match prop("kind") {
            Some(k) => {
                let prop_kind = CollectionKind::parse(&k)?;
                if prop_kind != table.kind {
                    return Err(StorageError::Invalid(format!(
                        "kind mismatch for table '{}': typed field is '{}', \
                         'kind' property is '{}'",
                        table.name,
                        table.kind.as_str(),
                        prop_kind.as_str()
                    )));
                }
                Some(prop_kind)
            }
            None => Some(table.kind),
        };
        // User properties (everything not computed above) are persisted on
        // the FileInfo so they survive the descriptor -> registry -> describe
        // round-trip.
        let user_properties: BTreeMap<String, String> = table
            .properties
            .iter()
            .filter(|(k, _)| !RESERVED_PROPERTIES.contains(&k.as_str()) && k.as_str() != "kind")
            .map(|(k, v)| (k.clone(), v.clone()))
            .collect();
        let mut info = FileInfo::new(
            table
                .base_location
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| {
                    crate::generations::artifact_file_name(&self.metadata.name_id, &table.name)
                }),
            prop("filetype").as_deref().unwrap_or("vector"),
            (parse("rows")?.unwrap_or(0), parse("cols")?.unwrap_or(0)),
            parse("nnz")?,
            None,
        )?;
        info.kind = kind;
        info.properties = user_properties;
        self.metadata.files.insert(table.name, info);
        Ok(())
    }

    fn deregister_table(&mut self, name: &str) -> StorageResult<()> {
        if self.metadata.files.remove(name).is_none() {
            return Err(StorageError::Invalid(format!(
                "table '{name}' is not registered"
            )));
        }
        Ok(())
    }
}
