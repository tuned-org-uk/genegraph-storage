//! Lance storage backend for graph embeddings.
//!
//! Async-first implementation that matches the async `StorageBackend` trait:
//! - All I/O is async, no internal `block_on` or runtime creation.
//! - Callers (CLI, tests, services) are responsible for providing a Tokio runtime.

use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Float64Array, Int64Array, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow_array::{Array as ArrowArray, FixedSizeListArray, RecordBatch, RecordBatchIterator};
use futures::StreamExt;
use lance::dataset::{Dataset, WriteMode, WriteParams};
use log::{debug, info, trace};
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::{CsMat, TriMat};

use crate::metadata::FileInfo;
use crate::metadata::GeneMetadata;
use crate::traits::StorageBackend;
use crate::{StorageError, StorageResult};

/// Lance-based storage backend for ArrowSpace graph embeddings.
///
/// Stores dense and sparse matrices as Lance datasets using a columnar format
/// (`row`, `col`, `value` for sparse; `col_*` for dense) schema for efficient
/// random and columnar access.
#[derive(Debug, Clone)]
pub struct LanceStorage {
    pub(crate) _base: String,
    pub(crate) _name: String,
}

impl LanceStorage {
    /// Creates a new Lance storage backend.
    ///
    /// This is used for on-the-fly creation. For proper setup use `Genefold<...>::seed`.
    ///
    /// # Arguments
    ///
    /// * `_base` - Base directory path for all storage files
    /// * `_name` - Name prefix for this storage instance
    pub fn new(_base: String, _name: String) -> Self {
        info!("Creating LanceStorage at base={}, name={}", _base, _name);
        Self { _base, _name }
    }

    /// Validates that the storage directory is properly initialized with metadata.
    ///
    /// # Returns
    ///
    /// Returns `Ok(())` if metadata file exists, otherwise returns an error.
    fn validate_initialized(&self, md_path: &Path) -> StorageResult<()> {
        assert_eq!(self.metadata_path(), *md_path);
        if !md_path.exists() {
            return Err(StorageError::Invalid(format!(
                "Storage not initialized: metadata file missing at {:?}. \
Call save_metadata() or save_eigenmaps_all()/save_energymaps_all() first.",
                md_path
            )));
        }
        Ok(())
    }

    /// Converts a dense matrix to a RecordBatch in vector format (Lance-optimized).
    /// Each row of the matrix becomes a single FixedSizeList entry.
    ///
    /// Arguments:
    /// * matrix - Dense matrix to convert (N rows × F cols)
    ///
    /// Returns:
    /// RecordBatch with schema: { vector: FixedSizeList<Float64>[F] }
    fn to_dense_record_batch(
        &self,
        matrix: &DenseMatrix<f64>,
    ) -> Result<RecordBatch, StorageError> {
        let (rows, cols) = (matrix.shape().0, matrix.shape().1);

        debug!(
            "Converting dense matrix to RecordBatch (vector format): {}x{}",
            rows, cols
        );

        if rows == 0 || cols == 0 {
            return Err(StorageError::Invalid(
                "Cannot convert empty matrix to RecordBatch".to_string(),
            ));
        }

        // Flatten matrix row-by-row into a single Vec<f64>
        let mut values: Vec<f64> = Vec::with_capacity(rows * cols);
        for r in 0..rows {
            for c in 0..cols {
                values.push(*matrix.get((r, c)));
            }
        }

        // Create FixedSizeList field: each entry is a vector of length cols
        let value_field = Field::new("item", DataType::Float64, false);
        let list_field = Field::new(
            "vector",
            DataType::FixedSizeList(Arc::new(value_field), cols as i32),
            false,
        );

        let schema = Schema::new(vec![list_field]);

        // Build the FixedSizeList array
        let values_array = Float64Array::from(values);
        let list_array = FixedSizeListArray::new(
            Arc::new(Field::new("item", DataType::Float64, false)),
            cols as i32,
            Arc::new(values_array),
            None, // No nulls
        );

        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list_array)])
            .map_err(|e| StorageError::Lance(e.to_string()))?;

        trace!(
            "RecordBatch created with {} rows (vectors of length {})",
            batch.num_rows(),
            cols
        );

        Ok(batch)
    }

    /// Reconstructs a dense matrix from a RecordBatch in vector format.
    ///
    /// Arguments:
    /// * batch - RecordBatch containing FixedSizeList<Float64> vectors
    ///
    /// Returns:
    /// DenseMatrix in column-major format (smartcore convention)
    #[allow(clippy::wrong_self_convention)]
    fn from_dense_record_batch(
        &self,
        batch: &RecordBatch,
    ) -> Result<DenseMatrix<f64>, StorageError> {
        use std::mem;

        debug!("Reconstructing dense matrix from RecordBatch (vector format)");
        debug!("Batch has {} columns", batch.num_columns());

        if batch.num_columns() != 1 {
            return Err(StorageError::Invalid(format!(
                "Expected Lance row-major format with 1 FixedSizeList<Float64> column, but found {} columns. \
                  This parquet file appears to be in wide format (feature-per-column). \
                  Convert it first using: \
                  `python -c \"import pyarrow.parquet as pq; import pyarrow.compute as pc; \
                  tbl = pq.read_table('input.parquet'); \
                  import pyarrow as pa; \
                  vectors = pa.array([row.as_py() for row in tbl.to_pylist()], type=pa.list_(pa.float64(), len(tbl.column_names))); \
                  new_tbl = pa.table({{'vector': vectors}}); \
                  pq.write_table(new_tbl, 'output.parquet')\"` \
                  or use a Lance-native writer in your data pipeline.",
                batch.num_columns()
            )));
        }

        debug!("Extracting FixedSizeList column");
        let column = batch.column(0);
        let list_array = column
            .as_any()
            .downcast_ref::<FixedSizeListArray>()
            .ok_or_else(|| {
                StorageError::Invalid(format!(
                    "Column 0 is not FixedSizeList (found type: {:?}). \
                      Expected Lance row-major format with a single FixedSizeList<Float64> column.",
                    column.data_type()
                ))
            })?;

        let rows = list_array.len();
        let cols = list_array.value_length() as usize;

        debug!("Matrix dimensions: {}x{}", rows, cols);

        // Guard against excessive allocations
        let total = rows
            .checked_mul(cols)
            .ok_or_else(|| StorageError::Invalid("Matrix size overflow (rows*cols)".to_string()))?;
        let bytes = total
            .checked_mul(mem::size_of::<f64>())
            .ok_or_else(|| StorageError::Invalid("Byte size overflow".to_string()))?;

        const MAX_BYTES: usize = 4usize * 1024 * 1024 * 1024; // 4 GiB
        if bytes > MAX_BYTES {
            return Err(StorageError::Invalid(format!(
                "Dense load would allocate {} bytes for {}x{} matrix; exceeds 4GiB cap. \
                  Enable --reduce-dim or shard your input data.",
                bytes, rows, cols
            )));
        }

        // Extract Float64 values
        let values_array = list_array
            .values()
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                StorageError::Invalid("FixedSizeList values are not Float64Array".to_string())
            })?;

        debug!("Converting row-major to column-major");
        let mut data = vec![0.0f64; total];
        for r in 0..rows {
            for c in 0..cols {
                let row_major_idx = r * cols + c;
                let col_major_idx = c * rows + r;
                data[col_major_idx] = values_array.value(row_major_idx);
            }
        }

        debug!("Creating DenseMatrix");
        DenseMatrix::new(rows, cols, data, true).map_err(|e| StorageError::Invalid(e.to_string()))
    }

    /// Spawn a LanceStorage from an existing seeded directory (with metadata.json)
    pub async fn spawn(base_path: String) -> Result<(LanceStorage, GeneMetadata), StorageError> {
        // Reuse the generic `exists` helper from the StorageBackend trait
        let (exists, md_path) = <LanceStorage as StorageBackend>::exists(&base_path);
        assert!(
            exists && md_path.is_some(),
            "Metadata does not exist in this base path"
        );

        // Load metadata from the discovered metadata.json
        let metadata = GeneMetadata::read(md_path.unwrap()).await?;

        // Construct the LanceStorage using the metadata-provided nameid
        let storage = LanceStorage::new(base_path.clone(), metadata.name_id.clone());

        Ok((storage, metadata))
    }

    /// Converts a sparse CSR matrix to a RecordBatch in columnar format.
    ///
    /// Only non-zero entries are stored.
    fn to_sparse_record_batch(&self, m: &CsMat<f64>) -> StorageResult<RecordBatch> {
        debug!(
            "Converting sparse matrix to RecordBatch: {} x {}, nnz={}",
            m.rows(),
            m.cols(),
            m.nnz()
        );

        let mut row_idx = Vec::with_capacity(m.nnz());
        let mut col_idx = Vec::with_capacity(m.nnz());
        let mut vals = Vec::with_capacity(m.nnz());

        for (v, (r, c)) in m.iter() {
            row_idx.push(r as u32);
            col_idx.push(c as u32);
            vals.push(*v);
        }

        // Store actual dimensions in schema metadata
        let mut schema_metadata = std::collections::HashMap::new();
        schema_metadata.insert("rows".to_string(), m.rows().to_string());
        schema_metadata.insert("cols".to_string(), m.cols().to_string());
        schema_metadata.insert("nnz".to_string(), m.nnz().to_string());

        let schema = Schema::new(vec![
            Field::new("row", DataType::UInt32, false),
            Field::new("col", DataType::UInt32, false),
            Field::new("value", DataType::Float64, false),
        ])
        .with_metadata(schema_metadata);

        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![
                Arc::new(UInt32Array::from(row_idx)) as _,
                Arc::new(UInt32Array::from(col_idx)) as _,
                Arc::new(Float64Array::from(vals)) as _,
            ],
        )
        .map_err(|e| StorageError::Lance(e.to_string()))?;

        trace!(
            "Sparse RecordBatch created with {} entries",
            batch.num_rows()
        );
        Ok(batch)
    }

    /// Reconstructs a sparse CSR matrix from a RecordBatch in columnar format.
    ///
    /// * `batch` - RecordBatch containing (`row`, `col`, `value`) triplets
    /// * `expected_rows` / `expected_cols` - dimensions taken from metadata
    #[allow(clippy::wrong_self_convention)]
    fn from_sparse_record_batch(
        &self,
        batch: RecordBatch,
        expected_rows: usize,
        expected_cols: usize,
    ) -> StorageResult<CsMat<f64>> {
        use arrow::array::UInt32Array;

        debug!("Reconstructing sparse matrix from RecordBatch");

        let row_arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| StorageError::Invalid("row column type mismatch".into()))?;
        let col_arr = batch
            .column(1)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| StorageError::Invalid("col column type mismatch".into()))?;
        let val_arr = batch
            .column(2)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| StorageError::Invalid("value column type mismatch".into()))?;

        let n = row_arr.len();
        if n == 0 {
            debug!(
                "Empty RecordBatch, returning {}x{} sparse matrix",
                expected_rows, expected_cols
            );
            return Ok(CsMat::zero((expected_rows, expected_cols)));
        }

        // Try to read dimensions from schema metadata (for validation)
        let schema = batch.schema();
        let schema_metadata = schema.metadata();
        if let (Some(rows_str), Some(cols_str)) =
            (schema_metadata.get("rows"), schema_metadata.get("cols"))
        {
            let schema_rows = rows_str.parse::<usize>().ok();
            let schema_cols = cols_str.parse::<usize>().ok();
            if schema_rows != Some(expected_rows) || schema_cols != Some(expected_cols) {
                panic!(
                    "Schema metadata dimensions ({:?}x{:?}) don't match storage metadata ({}x{})",
                    schema_rows, schema_cols, expected_rows, expected_cols
                );
            } else {
                debug!(
                    "Schema metadata matches storage metadata: {}x{}",
                    expected_rows, expected_cols
                );
            }
        }

        let rows = expected_rows;
        let cols = expected_cols;
        debug!(
            "Reconstructing {}x{} sparse matrix from {} entries",
            rows, cols, n
        );

        let mut trimat = TriMat::new((rows, cols));
        for i in 0..n {
            let r = row_arr.value(i) as usize;
            let c = col_arr.value(i) as usize;
            let v = val_arr.value(i);

            if r >= rows || c >= cols {
                return Err(StorageError::Invalid(format!(
                    "Index out of bounds: ({}, {}) in {}x{} matrix",
                    r, c, rows, cols
                )));
            }
            trimat.add_triplet(r, c, v);
        }

        let result = trimat.to_csr();
        if result.rows() != rows || result.cols() != cols {
            return Err(StorageError::Invalid(format!(
                "Dimension mismatch after reconstruction: expected {}x{}, got {}x{}",
                rows,
                cols,
                result.rows(),
                result.cols()
            )));
        }

        Ok(result)
    }

    /// Async helper: write a RecordBatch to a Lance dataset.
    async fn write_lance_batch_async(&self, path: &Path, batch: RecordBatch) -> StorageResult<()> {
        let uri = Self::path_to_uri(path);
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
    async fn read_lance_all_batches_async(&self, path: &Path) -> StorageResult<RecordBatch> {
        let uri = Self::path_to_uri(path);
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
            path,
            combined.num_rows()
        );
        Ok(combined)
    }

    /// Async helper: read the first RecordBatch from a Lance dataset.
    async fn read_lance_first_batch_async(&self, path: &Path) -> StorageResult<RecordBatch> {
        let uri = Self::path_to_uri(path);
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
            path,
            batch.num_rows()
        );
        Ok(batch)
    }
}

impl StorageBackend for LanceStorage {
    fn get_base(&self) -> String {
        self._base.clone()
    }

    fn get_name(&self) -> String {
        self._name.clone()
    }

    fn base_path(&self) -> PathBuf {
        PathBuf::from(&self._base)
    }

    fn metadata_path(&self) -> PathBuf {
        self.base_path()
            .join(format!("{}_metadata.json", self._name))
    }

    /// Converts the base path for the store to a `file://` URI for Lance.
    fn basepath_to_uri(&self) -> String {
        Self::path_to_uri(PathBuf::from(self._base.clone()).as_path())
    }

    /// Converts a full file path to a `file://` URI for Lance.
    fn path_to_uri(path: &Path) -> String {
        path.canonicalize()
            .unwrap_or_else(|_| {
                if path.is_absolute() {
                    path.to_path_buf()
                } else if path.is_relative() {
                    std::env::current_dir()
                        .unwrap_or_else(|_| PathBuf::from("/"))
                        .join(path)
                } else {
                    PathBuf::from(env!("CARGO_MANIFEST_DIR")).join(path)
                }
            })
            .to_string_lossy()
            .to_string()
    }

    // Add these methods to impl LanceStorage {} block

    /// Save dense matrix using Lance-optimized vector format.
    ///
    /// Each row of the matrix becomes a FixedSizeList entry for efficient vector operations.
    /// This format is optimized for vector search and enables Lance's full-zip encoding.
    ///
    /// # Arguments
    /// * `filename` - Type identifier (e.g., "rawinput", "sub_centroids")
    /// * `matrix` - Dense matrix to save (N rows × F cols)
    /// * `md_path` - Metadata file path for validation
    async fn save_dense(
        &self,
        key: &str,
        matrix: &DenseMatrix<f64>,
        md_path: &Path,
    ) -> StorageResult<()> {
        self.validate_initialized(md_path)?;
        let path = self.file_path(key);
        let (n_rows, n_cols) = matrix.shape();

        info!(
            "Saving dense {} matrix: {} x {} at {:?}",
            key, n_rows, n_cols, path
        );

        // Convert to Lance-optimized RecordBatch (FixedSizeList format)
        let batch = self.to_dense_record_batch(matrix)?;

        // Verify batch has correct number of rows
        if batch.num_rows() != n_rows {
            return Err(StorageError::Invalid(format!(
                "RecordBatch has {} rows but matrix has {} rows",
                batch.num_rows(),
                n_rows
            )));
        }

        // Write to Lance
        self.write_lance_batch_async(&path, batch).await?;

        info!("Dense {} matrix saved successfully", key);
        Ok(())
    }

    /// Load dense matrix from Lance-optimized vector format.
    ///
    /// Reads FixedSizeList vectors and reconstructs a column-major DenseMatrix.
    ///
    /// # Arguments
    /// * `filename` - Type identifier (e.g., "rawinput", "sub_centroids")
    ///
    /// # Returns
    /// Column-major DenseMatrix matching smartcore conventions
    async fn load_dense(&self, key: &str) -> StorageResult<DenseMatrix<f64>> {
        let path = self.file_path(key);
        info!("Loading dense {} matrix from {:?}", key, path);

        // Read all batches from Lance (may span multiple batches for large datasets)
        let batch = self.read_lance_all_batches_async(&path).await?;

        // Convert from FixedSizeList format to DenseMatrix
        let matrix = self.from_dense_record_batch(&batch)?;

        let (n_rows, n_cols) = matrix.shape();
        info!("Loaded dense {} matrix: {} x {}", key, n_rows, n_cols);

        Ok(matrix)
    }

    /// Load initial data using columnar format from a file path.
    ///
    /// Async test helper that avoids any internal blocking runtimes.
    async fn load_dense_from_file(&self, path: &Path) -> StorageResult<DenseMatrix<f64>> {
        info!("Loading dense matrix from file (async): {:?}", path);

        if !path.exists() {
            return Err(StorageError::Invalid(format!(
                "Dense file does not exist: {:?}",
                path
            )));
        }

        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| StorageError::Invalid(format!("Invalid file path: {:?}", path)))?;

        match extension {
            "lance" => {
                // Use a temporary LanceStorage rooted at the file's parent dir,
                // same pattern as save_dense_to_file_async.
                let parent = path
                    .parent()
                    .ok_or_else(|| {
                        StorageError::Invalid(format!("Path has no parent: {:?}", path))
                    })?
                    .to_str()
                    .ok_or_else(|| {
                        StorageError::Invalid(format!("Non-UTF8 parent path for {:?}", path))
                    })?
                    .to_string();

                let tmp_storage =
                    crate::lance::LanceStorage::new(parent, String::from("tmp_storage"));

                // Reuse the async Lance reader logic.
                let batch = tmp_storage.read_lance_all_batches_async(path).await?;
                let matrix = tmp_storage.from_dense_record_batch(&batch)?;
                info!(
                    "Loaded dense matrix from Lance: {} x {}",
                    matrix.shape().0,
                    matrix.shape().1
                );
                Ok(matrix)
            }
            "parquet" => {
                use arrow::datatypes::DataType;
                use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;
                use std::fs::File;

                // 1. Read from Parquet into a single RecordBatch
                let file = File::open(path)
                    .map_err(|e| StorageError::Io(format!("Failed to open parquet file: {}", e)))?;

                let builder = ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
                    StorageError::Parquet(format!("Failed to create parquet reader: {}", e))
                })?;
                let mut reader = builder.build().map_err(|e| {
                    StorageError::Parquet(format!("Failed to build parquet reader: {}", e))
                })?;

                let mut batches = Vec::new();
                #[allow(clippy::while_let_on_iterator)]
                while let Some(batch) = reader.next() {
                    let batch = batch.map_err(|e| {
                        StorageError::Parquet(format!("Failed to read parquet batch: {}", e))
                    })?;
                    batches.push(batch);
                }

                if batches.is_empty() {
                    return Err(StorageError::Invalid(format!(
                        "Empty parquet dataset at {:?}",
                        path
                    )));
                }

                let schema = batches[0].schema();
                let combined = arrow::compute::concat_batches(&schema, &batches).map_err(|e| {
                    StorageError::Parquet(format!("Failed to concatenate parquet batches: {}", e))
                })?;

                // 2. Detect layout: vector (FixedSizeList) vs old wide columnar (col_* Float64)
                let fields = schema.fields();
                let is_vector = fields.len() == 1
                    && matches!(
                        fields[0].data_type(),
                        DataType::FixedSizeList(inner, _)
                            if matches!(inner.data_type(), DataType::Float64)
                    );

                let is_wide_col = !is_vector
                    && !fields.is_empty()
                    && fields
                        .iter()
                        .all(|f| matches!(f.data_type(), DataType::Float64))
                    && fields.iter().any(|f| f.name().starts_with("col_"));

                // 3. Build DenseMatrix from the RecordBatch
                let matrix = if is_vector {
                    // New format already: vector column (FixedSizeList<Float64>)
                    // Reuse the same decoding as Lance.
                    let parent = path
                        .parent()
                        .ok_or_else(|| {
                            StorageError::Invalid(format!("Path has no parent: {:?}", path))
                        })?
                        .to_str()
                        .ok_or_else(|| {
                            StorageError::Invalid(format!("Non-UTF8 parent path for {:?}", path))
                        })?
                        .to_string();

                    let tmp_storage =
                        crate::lance::LanceStorage::new(parent, String::from("tmp_storage"));
                    tmp_storage.from_dense_record_batch(&combined)?
                } else if is_wide_col {
                    // Old wide columnar: columns like col_0, col_1, ... as Float64
                    let n_rows = combined.num_rows();
                    let n_cols = combined.num_columns();
                    if n_rows == 0 || n_cols == 0 {
                        return Err(StorageError::Invalid(format!(
                            "Cannot load empty wide-column parquet at {:?}",
                            path
                        )));
                    }

                    let mut data = Vec::with_capacity(n_rows * n_cols);
                    for col_idx in 0..n_cols {
                        let col = combined.column(col_idx);
                        let arr = col
                            .as_any()
                            .downcast_ref::<arrow_array::Float64Array>()
                            .ok_or_else(|| {
                                StorageError::Invalid(format!(
                                    "Wide-column parquet expects Float64, got {:?} in column {}",
                                    col.data_type(),
                                    col_idx
                                ))
                            })?;
                        // Build column-major storage: all rows for col 0, then col 1, ...
                        for row_idx in 0..n_rows {
                            data.push(arr.value(row_idx));
                        }
                    }

                    DenseMatrix::new(n_rows, n_cols, data, true)
                        .map_err(|e| StorageError::Invalid(e.to_string()))?
                } else {
                    return Err(StorageError::Invalid(format!(
                        "Unsupported Parquet schema at {:?}: expected FixedSizeList<Float64> \
                         or wide Float64 columns named col_*",
                        path
                    )));
                };

                info!(
                    "Loaded dense matrix from Parquet: {} x {}",
                    matrix.shape().0,
                    matrix.shape().1
                );

                Ok(matrix)
            }
            _ => Err(StorageError::Invalid(format!(
                "Unsupported file format: {}. Only .lance and .parquet are supported",
                extension
            ))),
        }
    }

    fn file_path(&self, key: &str) -> PathBuf {
        self.base_path()
            .join(format!("{}_{}.lance", self._name, key))
    }

    // =========
    // ASYNC API (matches StorageBackend)
    // =========

    async fn save_sparse(
        &self,
        key: &str,
        matrix: &CsMat<f64>,
        md_path: &Path,
    ) -> StorageResult<()> {
        self.validate_initialized(md_path)?;
        let path = self.file_path(key);
        info!(
            "Saving sparse {} matrix: {} x {}, nnz={} at {:?}",
            key,
            matrix.rows(),
            matrix.cols(),
            matrix.nnz(),
            path
        );

        let mut metadata = self.load_metadata().await?;
        let filetype = FileInfo::which_filetype(key);
        metadata.files.insert(
            key.to_string(),
            FileInfo {
                filename: format!("{}_{}.lance", self.get_name(), key),
                filetype: filetype.to_string(),
                storage_format: FileInfo::which_format(&filetype),
                rows: matrix.rows(),
                cols: matrix.cols(),
                nnz: Some(matrix.nnz()),
                size_bytes: None,
            },
        );
        self.save_metadata(&metadata).await?;

        let batch = self.to_sparse_record_batch(matrix)?;
        self.write_lance_batch_async(&path, batch).await?;
        info!("Sparse matrix {} saved successfully", filetype);
        Ok(())
    }

    async fn load_sparse(&self, key: &str) -> StorageResult<CsMat<f64>> {
        info!("Loading sparse {} matrix", key);

        let metadata = self.load_metadata().await?;
        let filetype = FileInfo::which_filetype(key);
        let file_info = metadata
            .files
            .get(key)
            .ok_or_else(|| StorageError::Invalid(format!("{key} not found in metadata")))?;

        let expected_rows = file_info.rows;
        let expected_cols = file_info.cols;
        debug!(
            "Expected dimensions from storage metadata: {} x {}",
            expected_rows, expected_cols
        );

        let path = self.file_path(key);
        let batch = self.read_lance_first_batch_async(&path).await?;
        let matrix = self.from_sparse_record_batch(batch, expected_rows, expected_cols)?;
        info!(
            "Sparse {} matrix loaded: {} x {}, nnz={}",
            filetype,
            matrix.rows(),
            matrix.cols(),
            matrix.nnz()
        );
        Ok(matrix)
    }

    async fn save_lambdas(&self, lambdas: &[f64], md_path: &Path) -> StorageResult<()> {
        self.validate_initialized(md_path)?;
        let path = self.file_path("lambdas");
        info!("Saving {} lambda values", lambdas.len());

        let schema = Schema::new(vec![Field::new("lambda", DataType::Float64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Float64Array::from(lambdas.to_vec())) as _],
        )
        .map_err(|e| StorageError::Lance(e.to_string()))?;

        self.write_lance_batch_async(&path, batch).await?;
        info!("Lambda values saved successfully");
        Ok(())
    }

    async fn load_lambdas(&self) -> StorageResult<Vec<f64>> {
        let path = self.file_path("lambdas");
        info!("Loading lambda values from {:?}", path);

        let batch = self.read_lance_first_batch_async(&path).await?;
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| StorageError::Invalid("lambda column type mismatch".into()))?;

        let lambdas: Vec<f64> = (0..arr.len()).map(|i| arr.value(i)).collect();
        info!("Loaded {} lambda values", lambdas.len());
        Ok(lambdas)
    }

    async fn save_metadata(&self, metadata: &GeneMetadata) -> StorageResult<PathBuf> {
        let path = self.metadata_path();
        info!("Saving metadata to {:?}", path);
        fs::create_dir_all(self.base_path()).map_err(|e| StorageError::Io(e.to_string()))?;
        let s = serde_json::to_string_pretty(metadata).map_err(StorageError::Serde)?;
        fs::write(&path, s).map_err(|e| StorageError::Io(e.to_string()))?;
        info!("Metadata saved successfully");
        Ok(path)
    }

    async fn load_metadata(&self) -> StorageResult<GeneMetadata> {
        let filename = self.metadata_path();
        info!("Loading metadata from {:?}", filename);
        let s = fs::read_to_string(filename).map_err(|e| StorageError::Io(e.to_string()))?;
        let md: GeneMetadata = serde_json::from_str(&s).map_err(StorageError::Serde)?;
        info!("Metadata loaded successfully");
        Ok(md)
    }

    async fn save_vector(&self, key: &str, vector: &[f64], md_path: &Path) -> StorageResult<()> {
        self.validate_initialized(md_path)?;
        let path = self.file_path(key);
        info!("Saving {} values for vector {}", vector.len(), key);

        let schema = Schema::new(vec![Field::new("element", DataType::Float64, false)]);
        let float64_array = Float64Array::from_iter_values::<Vec<f64>>(vector.into());
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(float64_array) as _])
            .map_err(|e| StorageError::Lance(e.to_string()))?;

        self.write_lance_batch_async(&path, batch).await?;
        info!("Index {} saved successfully", key);
        Ok(())
    }

    async fn save_index(&self, key: &str, vector: &[usize], md_path: &Path) -> StorageResult<()> {
        self.validate_initialized(md_path)?;
        let path = self.file_path(key);
        info!("Saving {} values for index {}", vector.len(), key);

        let schema = Schema::new(vec![Field::new("id", DataType::UInt32, false)]);
        let uint32_array = UInt32Array::from_iter_values(vector.iter().map(|&x| x as u32));
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(uint32_array) as _])
            .map_err(|e| StorageError::Lance(e.to_string()))?;

        self.write_lance_batch_async(&path, batch).await?;
        info!("Index {} saved successfully", key);
        Ok(())
    }

    async fn load_vector(&self, filename: &str) -> StorageResult<Vec<f64>> {
        let path = self.file_path(filename);
        info!("Loading vector {} from {:?}", filename, path);

        let batch = self.read_lance_first_batch_async(&path).await?;
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| StorageError::Invalid("column type mismatch".into()))?;

        let vector: Vec<f64> = (0..arr.len()).map(|i| arr.value(i)).collect();
        info!("Loaded {} vector values for {}", vector.len(), filename);
        Ok(vector)
    }

    async fn load_index(&self, filename: &str) -> StorageResult<Vec<usize>> {
        let path = self.file_path(filename);
        info!("Loading vector {} from {:?}", filename, path);

        let batch = self.read_lance_first_batch_async(&path).await?;
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| StorageError::Invalid("column type mismatch".into()))?;

        let vector: Vec<usize> = (0..arr.len()).map(|i| arr.value(i) as usize).collect();
        info!("Loaded {} vector values for {}", vector.len(), filename);
        Ok(vector)
    }

    /// Save dense matrix to file in columnar format (col_0, col_1, ..., col_N)
    ///
    /// Async test helper that avoids any internal blocking runtimes.
    #[cfg(test)]
    async fn save_dense_to_file(data: &DenseMatrix<f64>, path: &PathBuf) -> StorageResult<()> {
        use tokio::fs as tokio_fs;

        info!("Saving dense matrix to file (async): {:?}", path);

        // Ensure parent dir exists for the test file.
        if let Some(parent) = path.parent() {
            tokio_fs::try_exists(parent).await.map_err(|e| {
                StorageError::Io(format!("Failed to create dir {:?}: {}", parent, e))
            })?;
        }

        // Create a temporary storage only to store the file.
        let tmp_storage = LanceStorage::new(
            String::from(path.parent().unwrap().to_str().unwrap()),
            String::from("tmp_storage"),
        );

        let extension = path
            .extension()
            .and_then(|e| e.to_str())
            .ok_or_else(|| StorageError::Invalid(format!("Invalid file path: {:?}", path)))?;

        let (n_rows, n_cols) = data.shape();
        info!("Saving matrix: {} rows x {} cols", n_rows, n_cols);

        match extension {
            "lance" => {
                let batch = tmp_storage.to_dense_record_batch(data)?;
                debug!(
                    "Created RecordBatch with {} rows for Lance",
                    batch.num_rows()
                );

                // Verify all rows are in the batch
                if batch.num_rows() != n_rows {
                    return Err(StorageError::Invalid(format!(
                        "RecordBatch has {} rows but matrix has {} rows",
                        batch.num_rows(),
                        n_rows
                    )));
                }

                tmp_storage.write_lance_batch_async(path, batch).await?;
                info!("Saved dense matrix to Lance: {} x {}", n_rows, n_cols);
                Ok(())
            }
            "parquet" => {
                use parquet::arrow::ArrowWriter;
                use parquet::file::properties::WriterProperties;
                use std::fs::File;

                // For tests we still use sync parquet writer; directory was created with tokio_fs.
                let batch = tmp_storage.to_dense_record_batch(data)?;
                debug!(
                    "Created RecordBatch with {} rows for Parquet",
                    batch.num_rows()
                );

                if batch.num_rows() != n_rows {
                    return Err(StorageError::Invalid(format!(
                        "RecordBatch has {} rows but matrix has {} rows",
                        batch.num_rows(),
                        n_rows
                    )));
                }

                let file = File::create(path).map_err(|e| {
                    StorageError::Io(format!("Failed to create parquet file: {}", e))
                })?;

                let props = WriterProperties::builder()
                    .set_compression(parquet::basic::Compression::SNAPPY)
                    .build();

                let mut writer =
                    ArrowWriter::try_new(file, batch.schema(), Some(props)).map_err(|e| {
                        StorageError::Parquet(format!("Failed to create parquet writer: {}", e))
                    })?;

                writer
                    .write(&batch)
                    .map_err(|e| StorageError::Parquet(format!("Failed to write batch: {}", e)))?;

                writer
                    .close()
                    .map_err(|e| StorageError::Parquet(format!("Failed to close writer: {}", e)))?;

                info!("Saved dense matrix to Parquet: {} x {}", n_rows, n_cols);
                Ok(())
            }
            _ => Err(StorageError::Invalid(format!(
                "Unsupported file format: {}. Only .lance and .parquet are supported",
                extension
            ))),
        }
    }

    /// Save centroid_map (item-to-centroid assignments)
    async fn save_centroid_map(&self, map: &[usize], md_path: &Path) -> StorageResult<()> {
        self.validate_initialized(md_path)?;
        let path = self.file_path("centroid_map");
        info!("Saving {} centroid map entries", map.len());

        let schema = Schema::new(vec![Field::new("centroid_id", DataType::UInt32, false)]);
        let uint32_array = UInt32Array::from_iter_values(map.iter().map(|&x| x as u32));
        let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(uint32_array) as _])
            .map_err(|e| StorageError::Lance(e.to_string()))?;

        self.write_lance_batch_async(&path, batch).await?;
        info!("Centroid map saved successfully");
        Ok(())
    }

    /// Load centroid_map
    async fn load_centroid_map(&self) -> StorageResult<Vec<usize>> {
        let path = self.file_path("centroid_map");
        info!("Loading centroid map from {:?}", path);

        let batch = self.read_lance_first_batch_async(&path).await?;
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<UInt32Array>()
            .ok_or_else(|| StorageError::Invalid("centroid_id column type mismatch".into()))?;

        let map: Vec<usize> = (0..arr.len()).map(|i| arr.value(i) as usize).collect();
        info!("Loaded {} centroid map entries", map.len());
        Ok(map)
    }

    /// Save subcentroid_lambdas (tau values for subcentroids)
    async fn save_subcentroid_lambdas(&self, lambdas: &[f64], md_path: &Path) -> StorageResult<()> {
        self.validate_initialized(md_path)?;
        let path = self.file_path("subcentroid_lambdas");
        info!("Saving {} subcentroid lambda values", lambdas.len());

        let schema = Schema::new(vec![Field::new(
            "subcentroid_lambda",
            DataType::Float64,
            false,
        )]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Float64Array::from(lambdas.to_vec())) as _],
        )
        .map_err(|e| StorageError::Lance(e.to_string()))?;

        self.write_lance_batch_async(&path, batch).await?;
        info!("Subcentroid lambda values saved successfully");
        Ok(())
    }

    /// Load subcentroid_lambdas
    async fn load_subcentroid_lambdas(&self) -> StorageResult<Vec<f64>> {
        let path = self.file_path("subcentroid_lambdas");
        info!("Loading subcentroid lambda values from {:?}", path);

        let batch = self.read_lance_first_batch_async(&path).await?;
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| {
                StorageError::Invalid("subcentroid_lambda column type mismatch".into())
            })?;

        let lambdas: Vec<f64> = (0..arr.len()).map(|i| arr.value(i)).collect();
        info!("Loaded {} subcentroid lambda values", lambdas.len());
        Ok(lambdas)
    }

    /// Save subcentroids (dense matrix)
    async fn save_subcentroids(
        &self,
        subcentroids: &DenseMatrix<f64>,
        md_path: &Path,
    ) -> StorageResult<()> {
        self.validate_initialized(md_path)?;
        let path = self.file_path("sub_centroids");
        let (n_rows, n_cols) = subcentroids.shape();
        info!(
            "Saving subcentroids matrix {} x {} at {:?}",
            n_rows, n_cols, path
        );

        let batch = self.to_dense_record_batch(subcentroids)?;
        self.write_lance_batch_async(&path, batch).await?;
        debug!("Subcentroids matrix saved successfully");
        Ok(())
    }

    /// Load subcentroids as Vec<Vec<f64>>
    async fn load_subcentroids(&self) -> StorageResult<Vec<Vec<f64>>> {
        let path = self.file_path("sub_centroids");
        info!("Loading sub_centroids from {:?}", path);

        let batch = self.read_lance_all_batches_async(&path).await?;
        let matrix = self.from_dense_record_batch(&batch)?;

        // Convert DenseMatrix to Vec<Vec<f64>>
        let (n_rows, n_cols) = matrix.shape();
        let mut result = Vec::with_capacity(n_rows);

        for row_idx in 0..n_rows {
            let row: Vec<f64> = (0..n_cols)
                .map(|col_idx| *matrix.get((row_idx, col_idx)))
                .collect();
            result.push(row);
        }

        info!(
            "Loaded sub_centroids: {} x {} as Vec<Vec<f64>>",
            n_rows, n_cols
        );
        Ok(result)
    }

    /// Save item norms vector
    async fn save_item_norms(&self, item_norms: &[f64], md_path: &Path) -> StorageResult<()> {
        self.validate_initialized(md_path)?;
        let path = self.file_path("item_norms");
        info!("Saving {} item norm values", item_norms.len());

        let schema = Schema::new(vec![Field::new("norm", DataType::Float64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Float64Array::from(item_norms.to_vec())) as _],
        )
        .map_err(|e| StorageError::Lance(e.to_string()))?;

        self.write_lance_batch_async(&path, batch).await?;
        info!("Item norms saved successfully");
        Ok(())
    }

    /// Load item norms vector
    async fn load_item_norms(&self) -> StorageResult<Vec<f64>> {
        let path = self.file_path("item_norms");
        info!("Loading item norms from {:?}", path);

        let batch = self.read_lance_first_batch_async(&path).await?;
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| StorageError::Invalid("norm column type mismatch".into()))?;

        let norms: Vec<f64> = (0..arr.len()).map(|i| arr.value(i)).collect();
        info!("Loaded {} item norm values", norms.len());
        Ok(norms)
    }

    async fn save_cluster_assignments(
        &self,
        assignments: &[Option<usize>],
        md_path: &Path,
    ) -> StorageResult<()> {
        self.validate_initialized(md_path)?;
        let path = self.file_path("cluster_assignments");
        info!("Saving {} cluster assignments", assignments.len());

        // Convert Option<usize> to i64 (-1 for None)
        let values: Vec<i64> = assignments
            .iter()
            .map(|opt| opt.map(|v| v as i64).unwrap_or(-1))
            .collect();

        let schema = Schema::new(vec![Field::new("cluster_id", DataType::Int64, false)]);
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Int64Array::from(values)) as _],
        )
        .map_err(|e| StorageError::Lance(e.to_string()))?;

        self.write_lance_batch_async(&path, batch).await?;
        info!("Cluster assignments saved successfully");
        Ok(())
    }

    async fn load_cluster_assignments(&self) -> StorageResult<Vec<Option<usize>>> {
        let path = self.file_path("cluster_assignments");
        info!("Loading cluster assignments from {:?}", path);

        let batch = self.read_lance_first_batch_async(&path).await?;
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| StorageError::Invalid("cluster_id column type mismatch".into()))?;

        let assignments: Vec<Option<usize>> = (0..arr.len())
            .map(|i| {
                let val = arr.value(i);
                if val < 0 { None } else { Some(val as usize) }
            })
            .collect();

        info!("Loaded {} cluster assignments", assignments.len());
        Ok(assignments)
    }
}
