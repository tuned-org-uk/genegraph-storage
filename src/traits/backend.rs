use arrow::array::{Float64Array, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow_array::{Array as ArrowArray, FixedSizeListArray, RecordBatch};
use log::{debug, info, trace};
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::{CsMat, TriMat};
use std::fs;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use crate::metadata::GeneMetadata;
use crate::{StorageError, StorageResult};

/// Async storage backend for Lance-based graph and embedding data.
///
/// This trait defines the minimal async API required to persist and reload
/// all artifacts used by Javelin:
///
/// - Dense matrices (embeddings, eigenmaps, energy maps)
/// - Sparse matrices in CSR form (e.g. Laplacians, adjacency)
/// - Scalar vectors (eigenvalues, norms, generic f64 sequences)
/// - Index-like vectors (usize mappings and cluster assignments)
/// - Clustering metadata (centroid maps, subcentroids, lambdas)
/// - Global metadata describing the dataset layout and dimensions
///
/// ## Initialization
///
/// Storage must be initialized before saving any data:
///
/// 1. Call `save_metadata()` once to write an initial `*_metadata.json`.
/// 2. Subsequent `save_*` calls validate that metadata exists and is consistent.
/// 3. `exists()` can be used to detect and reuse an existing initialized store.
///
/// Filenames are conventionally:
///
/// ```ignore
/// <base dir>/<instance name or name id>_<key>.lance
/// ```
///
/// ## Async usage
///
/// All I/O functions are async and intended to be called from a Tokio runtime.
/// Implementations (e.g. `LanceStorage`) must not create their own runtimes or
/// block on I/O internally.
///
/// ## High-level flow
///
/// - Dense data:
///   - `save_dense("raw_input", &matrix, md_path)`
///   - `load_dense("raw_input")`
///
/// - Sparse data:
///   - `save_sparse("laplacian", &csr, md_path)`
///   - `load_sparse("laplacian")`
///
/// - Scalars and indices:
///   - `save_lambdas`, `load_lambdas`
///   - `save_vector`, `load_vector`
///   - `save_index`, `load_index`
///   - `save_centroid_map`, `load_centroid_map`
///   - `save_item_norms`, `load_item_norms`
///   - `save_cluster_assignments`, `load_cluster_assignments`
///
/// - Clustering structure:
///   - `save_subcentroids`, `load_subcentroids`
///   - `save_subcentroid_lambdas`, `load_subcentroid_lambdas`
///
/// Implementations are free to choose the on-disk layout as long as they honor
/// these logical keys and round-trip semantics.
pub trait StorageBackend: Send + Sync {
    /// Base directory of the instance
    fn get_base(&self) -> String;
    /// Name of the instance
    fn get_name(&self) -> String;

    ///
    /// Returns `true` and the path to the metadata file if metadata file exists and is valid,
    /// `false` otherwise.
    /// This is used to avoid overwriting existing indexes.
    fn exists(path: &str) -> (bool, Option<PathBuf>) {
        let base_path = std::path::PathBuf::from(path);
        if !base_path.exists() {
            debug!("StorageBackend: path {:?} does not exist", base_path);
            return (false, None);
        }

        // Check for any _metadata.json file in the directory
        if let Ok(entries) = std::fs::read_dir(&base_path) {
            for entry in entries.flatten() {
                let path = entry.path();
                if let Some(name) = path.file_name().and_then(|n| n.to_str())
                    && name.ends_with("_metadata.json")
                {
                    debug!("StorageBackend::exists: found metadata file at {:?}", path);
                    return (true, Some(path));
                }
            }
        }
        (false, None)
    }

    /// Returns the base directory path.
    fn base_path(&self) -> PathBuf;
    /// Returns the metadata path.
    fn metadata_path(&self) -> PathBuf;
    /// return the base path as file:// string
    fn basepath_to_uri(&self) -> String;

    /// Load initial data using columnar format from a file path.
    /// Implementations may use this as a helper for async `load_dense`.
    async fn load_dense_from_file(&self, path: &Path) -> StorageResult<DenseMatrix<f64>>;

    /// Compute the full Lance/parquet file path for a logical filetype.
    fn file_path(&self, key: &str) -> PathBuf;

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

    // =========
    // ASYNC API
    // =========

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

    /// Saves a dense matrix. Requires metadata to exist.
    async fn save_dense(
        &self,
        key: &str,
        matrix: &DenseMatrix<f64>,
        md_path: &Path,
    ) -> StorageResult<()>;

    /// Loads a dense matrix from storage.
    async fn load_dense(&self, key: &str) -> StorageResult<DenseMatrix<f64>>;

    /// Saves a sparse matrix. Requires metadata to exist.
    async fn save_sparse(
        &self,
        key: &str,
        matrix: &CsMat<f64>,
        md_path: &Path,
    ) -> StorageResult<()>;

    /// Loads a sparse matrix from storage.
    async fn load_sparse(&self, key: &str) -> StorageResult<CsMat<f64>>;

    /// Saves lambda eigenvalues. Requires metadata to exist.
    async fn save_lambdas(&self, lambdas: &[f64], md_path: &Path) -> StorageResult<()>;

    /// Loads lambda eigenvalues from storage.
    async fn load_lambdas(&self) -> StorageResult<Vec<f64>>;

    /// Initializes storage by saving metadata. Must be called first.
    async fn save_metadata(&self, metadata: &GeneMetadata) -> StorageResult<PathBuf>;

    /// Loads metadata from storage.
    async fn load_metadata(&self) -> StorageResult<GeneMetadata> {
        let filename = self.metadata_path();
        info!("Loading metadata from {:?}", filename);
        let s = fs::read_to_string(filename).map_err(|e| StorageError::Io(e.to_string()))?;
        let md: GeneMetadata = serde_json::from_str(&s).map_err(StorageError::Serde)?;
        info!("Metadata loaded successfully");
        Ok(md)
    }

    /// Save vectors that are not lambdas but indices.
    #[allow(dead_code)]
    async fn save_index(&self, key: &str, vector: &[usize], md_path: &Path) -> StorageResult<()>;

    /// save a generic f64 sequence
    async fn save_vector(&self, key: &str, vector: &[f64], md_path: &Path) -> StorageResult<()>;

    /// Save centroid_map (vector of usize mapping items to centroids)
    async fn save_centroid_map(&self, map: &[usize], md_path: &Path) -> StorageResult<()>;

    /// Load centroid_map
    async fn load_centroid_map(&self) -> StorageResult<Vec<usize>>;
    /// Save subcentroid_lambdas (tau values for subcentroids)
    async fn save_subcentroid_lambdas(&self, lambdas: &[f64], md_path: &Path) -> StorageResult<()>;
    /// Load subcentroid_lambdas
    async fn load_subcentroid_lambdas(&self) -> StorageResult<Vec<f64>>;
    /// Save subcentroids (dense matrix)
    async fn save_subcentroids(
        &self,
        subcentroids: &DenseMatrix<f64>,
        md_path: &Path,
    ) -> StorageResult<()>;
    /// Load subcentroids
    async fn load_subcentroids(&self) -> StorageResult<Vec<Vec<f64>>>;

    /// Save item norms (precomputed L2 norms for fast distance computation)
    async fn save_item_norms(&self, item_norms: &[f64], md_path: &Path) -> StorageResult<()>;

    /// Load item norms
    async fn load_item_norms(&self) -> StorageResult<Vec<f64>>;

    /// Save cluster assignments (Vec<Option<usize>>)
    async fn save_cluster_assignments(
        &self,
        assignments: &[Option<usize>],
        md_path: &Path,
    ) -> StorageResult<()>;

    /// Load cluster assignments
    async fn load_cluster_assignments(&self) -> StorageResult<Vec<Option<usize>>>;

    /// Load index or generic usize vector from storage.
    #[allow(dead_code)]
    async fn load_index(&self, key: &str) -> StorageResult<Vec<usize>>;

    async fn load_vector(&self, key: &str) -> StorageResult<Vec<f64>>;

    async fn save_dense_to_file(data: &DenseMatrix<f64>, path: &Path) -> StorageResult<()>;
}
