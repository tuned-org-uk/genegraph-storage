use crate::{StorageError, StorageResult};
// VERIFIED(lance 6.0.0): RecordBatchIterator is in arrow::record_batch — unchanged.
use arrow::record_batch::RecordBatchIterator;
use arrow_array::RecordBatch;
use log::{debug, info};

// VERIFIED(lance 6.0.0): futures::StreamExt drives the RecordBatchStream — unchanged.
use futures::StreamExt;
// VERIFIED(lance 6.0.0): lance::Dataset is the top-level entry point — unchanged.
use lance::Dataset;
// VERIFIED(lance 6.0.0): WriteMode and WriteParams remain in lance::dataset — unchanged.
// WriteParams::default() is still valid; WriteMode::Create is still the correct variant.
use lance::dataset::{WriteMode, WriteParams};

pub trait LanceStorage {
    /// Async helper: write a RecordBatch to a Lance dataset.
    ///
    /// # Lance 6 API notes
    /// - `Dataset::write(reader, uri, Some(params))` — signature unchanged.
    /// - `WriteParams { mode: WriteMode::Create, ..Default::default() }` — stable.
    /// - `RecordBatchIterator::new(iter, schema)` — stable; iterator items must be
    ///   `Result<RecordBatch, ArrowError>` which `.map(Ok)` satisfies.
    async fn write_lance_batch_async(&self, uri: String, batch: RecordBatch) -> StorageResult<()> {
        info!("Writing Lance dataset to {}", uri);

        let schema = batch.schema();
        let batches = vec![batch];
        let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), schema);

        let params = WriteParams {
            mode: WriteMode::Create,
            ..WriteParams::default()
        };

        // VERIFIED(lance 6.0.0): Dataset::write returns
        // Result<Dataset, lance::Error>; .map_err chain is correct.
        Dataset::write(reader, &uri, Some(params))
            .await
            .map_err(|e| StorageError::Lance(e.to_string()))?;

        info!("Successfully wrote Lance dataset to {}", uri);
        Ok(())
    }

    /// Async helper: read and concatenate all RecordBatches from a Lance dataset.
    ///
    /// # Lance 6 API notes
    /// - `Dataset::open(&uri)` — returns `Result<Dataset, lance::Error>`; unchanged.
    /// - `dataset.scan()` — returns `Scanner`; builder pattern unchanged.
    /// - `scanner.try_into_stream()` — returns
    ///   `Result<RecordBatchStream, lance::Error>`; this `.await`-based form is
    ///   stable in lance 6. Stream items are `Result<RecordBatch, lance::Error>`.
    /// - `.map_err(|e| StorageError::Lance(e.to_string()))` — correct for both
    ///   the stream-open error and per-batch errors.
    async fn read_lance_all_batches_async(&self, uri: String) -> StorageResult<RecordBatch> {
        info!("Reading Lance dataset from {}", uri);

        let dataset = Dataset::open(&uri)
            .await
            .map_err(|e| StorageError::Lance(e.to_string()))?;

        // VERIFIED(lance 6.0.0): scanner.try_into_stream().await returns
        // Result<impl Stream<Item = Result<RecordBatch, lance::Error>>, lance::Error>.
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
        // VERIFIED(lance 6.0.0 / arrow ^58): arrow::compute::concat_batches
        // signature unchanged; returns Result<RecordBatch, ArrowError>.
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
    ///
    /// # Lance 6 API notes
    /// - Same scanner/stream API as `read_lance_all_batches_async`.
    /// - `stream.next().await` returns `Option<Result<RecordBatch, lance::Error>>`.
    /// - `.ok_or_else` + `.map_err` pattern is correct and unchanged.
    async fn read_lance_first_batch_async(&self, uri: String) -> StorageResult<RecordBatch> {
        info!("Reading first batch from Lance dataset {}", uri);

        let dataset = Dataset::open(&uri)
            .await
            .map_err(|e| StorageError::Lance(e.to_string()))?;

        // VERIFIED(lance 6.0.0): same try_into_stream pattern as above.
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
