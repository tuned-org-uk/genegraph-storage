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
use crate::metadata::{FileInfo, GeneMetadata};

/// Generic Table API-compatible table descriptor.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TableDescriptor {
    /// Unique table name within the registry (the metadata files-map key).
    pub name: String,
    /// Table format; always `lance` for this crate.
    pub format: String,
    /// Location of the table root (a `*.lance` dataset directory).
    pub base_location: PathBuf,
    /// Free-form key/value properties (filetype, shape, ...).
    pub properties: BTreeMap<String, String>,
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
        TableDescriptor {
            name: key.to_string(),
            format: "lance".to_string(),
            base_location: self.base.join(&info.filename),
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
        let info = FileInfo::new(
            table
                .base_location
                .file_name()
                .map(|n| n.to_string_lossy().to_string())
                .unwrap_or_else(|| format!("{}_{}.lance", self.metadata.name_id, table.name)),
            prop("filetype").as_deref().unwrap_or("vector"),
            (parse("rows")?.unwrap_or(0), parse("cols")?.unwrap_or(0)),
            parse("nnz")?,
            None,
        )?;
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
