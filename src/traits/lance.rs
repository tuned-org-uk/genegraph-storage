use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use log::{debug, info};

use crate::graph::RESERVED_METADATA_KEYS;
use crate::metadata::FileInfo;
use crate::traits::backend::StorageBackend;
use crate::{StorageError, StorageResult};

/// Resolves a `file://` URI (as produced by `path_to_uri`) to a local path.
fn uri_to_path(uri: &str) -> StorageResult<std::path::PathBuf> {
    let url = url::Url::parse(uri)
        .map_err(|e| StorageError::Invalid(format!("bad dataset URI `{uri}`: {e}")))?;
    url.to_file_path().map_err(|_| {
        StorageError::Invalid(format!("dataset URI is not a local file path: `{uri}`"))
    })
}

/// Stamps collection metadata (RFC #81-P1) into the batch schema: the
/// dataset-level `kind`, writer-computed pairs, and the user properties.
/// Existing schema metadata is preserved; user properties may not shadow
/// reserved keys and the `kind` value always wins.
pub(crate) fn with_collection_metadata(
    batch: &RecordBatch,
    kind: &str,
    fixed: &[(&'static str, String)],
    user_properties: &BTreeMap<String, String>,
) -> StorageResult<RecordBatch> {
    for key in user_properties.keys() {
        if RESERVED_METADATA_KEYS.contains(&key.as_str()) {
            return Err(StorageError::Invalid(format!(
                "user property '{key}' is reserved by the collection metadata"
            )));
        }
    }
    let mut metadata = batch.schema().metadata().clone();
    for (k, v) in fixed {
        metadata.insert(k.to_string(), v.clone());
    }
    for (k, v) in user_properties {
        metadata.insert(k.clone(), v.clone());
    }
    metadata.insert("kind".to_string(), kind.to_string());

    let schema = Arc::new(batch.schema().as_ref().clone().with_metadata(metadata));
    RecordBatch::try_new(schema, batch.columns().to_vec())
        .map_err(|e| StorageError::Lance(e.to_string()))
}

/// Validates a vector-space schema (RFC #81-P2): at least one
/// `FixedSizeList<Float64|Float32>` column, no nullable top-level fields,
/// and any additional (id/property) columns limited to the lancefmt scalar
/// subset. Returns the dimension of the first vector column.
pub(crate) fn validate_vector_space_schema(schema: &Schema) -> StorageResult<(i32, DataType)> {
    let fields = schema.fields();
    if fields.is_empty() {
        return Err(StorageError::Invalid(
            "vector-space schema has no columns".into(),
        ));
    }
    let mut vector_dim: Option<(i32, DataType)> = None;
    for f in fields {
        if f.is_nullable() {
            return Err(StorageError::Invalid(format!(
                "vector-space schemas require non-nullable columns; '{}' is nullable",
                f.name()
            )));
        }
        match f.data_type() {
            DataType::FixedSizeList(child, dim)
                if matches!(child.data_type(), DataType::Float64 | DataType::Float32) =>
            {
                if vector_dim.is_none() {
                    vector_dim = Some((*dim, child.data_type().clone()));
                }
            }
            DataType::Float32
            | DataType::Float64
            | DataType::UInt8
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Int64 => {}
            other => {
                return Err(StorageError::Invalid(format!(
                    "vector-space id/property column '{}' has unsupported type {other:?}; \
                     expected FixedSizeList<Float64|Float32> or a scalar column",
                    f.name()
                )));
            }
        }
    }
    vector_dim.ok_or_else(|| {
        StorageError::Invalid(
            "vector-space schema requires at least one FixedSizeList<Float64|Float32> column"
                .into(),
        )
    })
}

pub trait LanceStorage {
    /// Async helper: write a RecordBatch to a Lance dataset.
    ///
    /// Runs the in-house v2.1 implementation (`lancefmt`) on the blocking
    /// pool (#75 M5; the `official-lance` transition flag was removed after
    /// one release cycle).
    async fn write_lance_batch_async(&self, uri: String, batch: RecordBatch) -> StorageResult<()> {
        info!("Writing Lance dataset (in-house v2.1) to {}", uri);
        let path = uri_to_path(&uri)?;
        tokio::task::spawn_blocking(move || crate::lancefmt::write_dataset(&batch, &path))
            .await
            .map_err(|e| StorageError::Io(format!("lancefmt writer task failed: {e}")))?
    }

    /// Async helper: read and concatenate all RecordBatches from a Lance dataset.
    async fn read_lance_all_batches_async(&self, uri: String) -> StorageResult<RecordBatch> {
        info!("Reading Lance dataset (in-house v2.1) from {}", uri);
        let path = uri_to_path(&uri)?;
        let combined = tokio::task::spawn_blocking(move || crate::lancefmt::scan_all(&path))
            .await
            .map_err(|e| StorageError::Io(format!("lancefmt reader task failed: {e}")))??;
        debug!(
            "Combined Lance batch for {:?} has {} rows",
            uri,
            combined.num_rows()
        );
        Ok(combined)
    }

    /// Writes a single-column primitive vector as a Lance dataset named
    /// `<name>_<key>.lance` and registers it in the metadata files map under
    /// `key` with filetype "vector" and shape (`<len>`, 1).
    ///
    /// Shared implementation for the scalar `save_*` methods (lambdas,
    /// vectors, indices, norms, centroid maps, cluster assignments).
    async fn save_primitive_column<T: ArrowPrimitiveType>(
        &self,
        key: &str,
        field_name: &str,
        values: Vec<T::Native>,
        md_path: &Path,
    ) -> StorageResult<()>
    where
        Self: StorageBackend,
    {
        self.validate_initialized(md_path)?;
        let path = self.file_path(key);
        let len = values.len();
        info!("Saving {} values for {} (field {})", len, key, field_name);

        let schema = Schema::new(vec![Field::new(field_name, T::DATA_TYPE, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(PrimitiveArray::<T>::from_iter_values(values)) as ArrayRef],
        )
        .map_err(|e| StorageError::Lance(e.to_string()))?;

        crate::commit::with_commit_actor(&self.metadata_path(), || async {
            let mut metadata = self.load_metadata().await?;
            metadata = metadata.add_file(
                key,
                FileInfo::new(
                    format!("{}_{}.lance", self.get_name(), key),
                    "vector",
                    (len, 1),
                    None,
                    None,
                )?,
            );
            self.save_metadata(&metadata).await
        })
        .await?;

        let uri = Self::path_to_uri(&path)?;
        self.write_lance_batch_async(uri, batch).await?;
        info!("Vector {} saved successfully", key);
        Ok(())
    }
}
