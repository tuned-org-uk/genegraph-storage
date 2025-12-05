use crate::{StorageError, StorageResult};
use arrow_array::RecordBatch;
use log::{debug, info};

use arrow_array::RecordBatchIterator;
use futures::StreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};

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

    /// Async helper: read the first RecordBatch from a Lance dataset.
    async fn read_lance_first_batch_async(&self, uri: String) -> StorageResult<RecordBatch> {
        info!("Reading first batch from Lance dataset {}", uri);

        let dataset = Dataset::open(&uri)
            .await
            .map_err(|e| StorageError::Lance(e.to_string()))?;
        let scanner = dataset.scan();
        let mut stream = scanner
            .try_into_stream()
            .await
            .map_err(|e| StorageError::Lance(e.to_string()))?;

        let batch = stream
            .next()
            .await
            .ok_or_else(|| StorageError::Lance("empty Lance dataset".to_string()))?
            .map_err(|e| StorageError::Lance(e.to_string()))?;

        debug!(
            "Read first RecordBatch for path {:?} with {} rows",
            uri,
            batch.num_rows()
        );
        Ok(batch)
    }
}
