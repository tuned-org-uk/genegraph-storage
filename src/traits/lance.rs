use std::path::Path;
use std::sync::Arc;

use arrow::array::{ArrayRef, PrimitiveArray};
use arrow::datatypes::{ArrowPrimitiveType, Field, Schema};
use arrow::record_batch::RecordBatch;
use arrow::record_batch::RecordBatchIterator;
use log::{debug, info};

use futures::StreamExt;
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};

use crate::metadata::FileInfo;
use crate::traits::backend::StorageBackend;
use crate::traits::metadata::Metadata;
use crate::{StorageError, StorageResult};

pub trait LanceStorage {
    /// Async helper: write a RecordBatch to a Lance dataset.
    async fn write_lance_batch_async(&self, uri: String, batch: RecordBatch) -> StorageResult<()> {
        info!("Writing Lance dataset to {}", uri);

        let schema = batch.schema();
        let batches = vec![batch];
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

        let params = WriteParams {
            mode: WriteMode::Create,
            ..WriteParams::default()
        };

        Dataset::write(reader, &uri, Some(params))
            .await
            .map_err(|e| StorageError::Lance(e.to_string()))?;

        info!("Successfully wrote Lance dataset to {}", uri);
        Ok(())
    }

    /// Async helper: read and concatenate all RecordBatches from a Lance dataset.
    async fn read_lance_all_batches_async(&self, uri: String) -> StorageResult<RecordBatch> {
        info!("Reading Lance dataset from {}", uri);

        let dataset = Dataset::open(&uri)
            .await
            .map_err(|e| StorageError::Lance(e.to_string()))?;
        let scanner = dataset.scan();
        let mut stream = scanner
            .try_into_stream()
            .await
            .map_err(|e| StorageError::Lance(e.to_string()))?;

        let mut batches = Vec::new();
        while let Some(batch_result) = stream.next().await {
            let batch = batch_result.map_err(|e| StorageError::Lance(e.to_string()))?;
            batches.push(batch);
        }

        if batches.is_empty() {
            return Err(StorageError::Invalid("Empty Lance dataset".into()));
        }

        let schema = batches[0].schema();
        let combined = arrow::compute::concat_batches(&schema, &batches)
            .map_err(|e| StorageError::Lance(format!("Failed to concatenate batches: {}", e)))?;

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
