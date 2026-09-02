use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::datatypes::{ArrowPrimitiveType, Field, Schema};
use arrow::record_batch::RecordBatch;
use log::{debug, info};

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
        self.save_metadata(&metadata).await?;

        let uri = Self::path_to_uri(&path)?;
        self.write_lance_batch_async(uri, batch).await?;
        info!("Vector {} saved successfully", key);
        Ok(())
    }
}
