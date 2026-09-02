//! Lance storage backend for graph embeddings.
//!
//! Async-first implementation that matches the async `StorageBackend` trait:
//! - All I/O is async, no internal `block_on` or runtime creation.
//! - Callers (CLI, tests, services) are responsible for providing a Tokio runtime.

use std::path::{Path, PathBuf};

use arrow::array::{Array as ArrowArray, Float64Array, UInt32Array};
use arrow::datatypes::{DataType, Float64Type, Int64Type, UInt32Type};
use arrow::record_batch::RecordBatch;
use log::{debug, info};
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::CsMat;

use crate::metadata::FileInfo;
use crate::metadata::GeneMetadata;
use crate::traits::backend::StorageBackend;
use crate::traits::lance::LanceStorage;
use crate::{StorageError, StorageResult};

/// Checked `usize -> u32` conversion.
///
/// Values above `u32::MAX` must surface an error instead of being silently
/// truncated into wrong stored data (issue #51).
fn checked_u32_values(values: &[usize], what: &str) -> StorageResult<Vec<u32>> {
    values
        .iter()
        .map(|&v| {
            u32::try_from(v).map_err(|_| {
                StorageError::Overflow(format!(
                    "{} value {} exceeds u32::MAX and would be silently truncated",
                    what, v
                ))
            })
        })
        .collect()
}

/// Lance-based storage backend for ArrowSpace graph embeddings.
///
/// Stores dense and sparse matrices as Lance datasets using a columnar format
/// (`row`, `col`, `value` for sparse; `col_*` for dense) schema for efficient
/// random and columnar access.
///
/// Metadata must be seeded before any `save_*` call so the storage directory
/// is initialized; the example below is executed as a doc-test on every
/// `cargo test` run so it cannot go stale.
///
/// # Examples
///
/// ```
/// use genegraph_storage::lance_storage_graph::LanceStorageGraph;
/// use genegraph_storage::metadata::GeneMetadata;
/// use genegraph_storage::traits::backend::StorageBackend;
/// use genegraph_storage::traits::metadata::Metadata;
/// use smartcore::linalg::basic::arrays::{Array, Array2};
/// use smartcore::linalg::basic::matrix::DenseMatrix;
///
/// let base = std::env::temp_dir().join(format!("genegraph_doc_{}", std::process::id()));
/// # tokio::runtime::Runtime::new().unwrap().block_on(async {
/// let storage = LanceStorageGraph::new(
///     base.to_string_lossy().to_string(),
///     "doc_example".to_string(),
/// );
///
/// // some 2D data
/// let dense: Vec<Vec<f64>> = vec![vec![0.1, 0.4], vec![0.5, 0.2], vec![0.03, 0.8]];
/// let (nitems, nfeatures) = (dense.len(), dense[0].len());
/// let data = DenseMatrix::<f64>::from_iterator(
///     dense.iter().flatten().copied(),
///     nitems,
///     nfeatures,
///     0,
/// );
///
/// // seed metadata FIRST to initialize the storage directory
/// let md = GeneMetadata::seed_metadata("doc_example", nitems, nfeatures, &storage)
///     .await
///     .unwrap();
/// let md_path = storage.save_metadata(&md).await.unwrap();
///
/// // your data is saved in an efficient Lance format
/// storage.save_dense("my_dataset", &data, &md_path).await.unwrap();
///
/// // Loading back
/// let loaded = storage.load_dense("my_dataset").await.unwrap();
/// assert_eq!(loaded.shape(), (nitems, nfeatures));
/// # });
/// # std::fs::remove_dir_all(&base).ok();
/// ```
#[derive(Debug, Clone)]
pub struct LanceStorageGraph {
    pub(crate) base: String,
    pub(crate) name: String,
}

impl LanceStorageGraph {
    /// Creates a new Lance storage backend.
    ///
    /// This is used for on-the-fly creation. For proper setup use `Genefold<...>::seed`.
    ///
    /// # Arguments
    ///
    /// * `base` - Base directory path for all storage files
    /// * `name` - Name prefix for this storage instance
    pub fn new(base: String, name: String) -> Self {
        info!("Creating LanceStorage at base={}, name={}", base, name);
        Self { base, name }
    }

    /// Spawn a LanceStorage from an existing seeded directory (with metadata.json)
    pub async fn spawn(base_path: String) -> Result<(Self, GeneMetadata), StorageError> {
        // Reuse the generic `exists` helper from the StorageBackend trait
        let (exists, md_path) = Self::exists(&base_path);

        // Replace assert! with proper error handling
        if !exists || md_path.is_none() {
            return Err(StorageError::Invalid(format!(
                "Metadata does not exist in base path: {}",
                base_path
            )));
        }

        // Load metadata from the discovered metadata.json
        let metadata = GeneMetadata::read(md_path.unwrap()).await?;

        // Construct the LanceStorage using the metadata-provided nameid
        let storage = Self::new(base_path.clone(), metadata.name_id.clone());
        Ok((storage, metadata))
    }
}

impl LanceStorage for LanceStorageGraph {}

impl StorageBackend for LanceStorageGraph {
    fn get_base(&self) -> String {
        self.base.clone()
    }

    fn get_name(&self) -> String {
        self.name.clone()
    }

    fn base_path(&self) -> PathBuf {
        PathBuf::from(&self.base)
    }

    fn metadata_path(&self) -> PathBuf {
        self.base_path()
            .join(format!("{}_metadata.json", self.name))
    }

    /// Converts the base path for the store to a `file://` URI for Lance.
    fn basepath_to_uri(&self) -> StorageResult<String> {
        Self::path_to_uri(PathBuf::from(self.base.clone()).as_path())
    }

    /// Save dense matrix using Lance-optimized vector format.
    ///
    /// Each row of the matrix becomes a FixedSizeList entry for efficient vector operations.
    /// This format is optimized for vector search and enables Lance's full-zip encoding.
    ///
    /// # Arguments
    /// * `filename` - any name
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

        {
            // Write to Lance
            let uri = Self::path_to_uri(&path)?;
            self.write_lance_batch_async(uri, batch).await?;
            let mut md = self.load_metadata().await?;
            md = md.add_file(
                key,
                FileInfo::new(
                    format!("{}_{}.lance", self.get_name(), key),
                    "dense",
                    matrix.shape(),
                    None,
                    None,
                )?,
            );
            self.save_metadata(&md).await?;
            info!("Dense {} matrix saved successfully", key);
        }
        Ok(())
    }

    /// Load dense matrix from Lance-optimized vector format.
    ///
    /// Reads FixedSizeList vectors and reconstructs a column-major DenseMatrix.
    ///
    /// # Arguments
    /// * `filename` - any name previously assigned
    ///
    /// # Returns
    /// Column-major DenseMatrix matching smartcore conventions
    async fn load_dense(&self, key: &str) -> StorageResult<DenseMatrix<f64>> {
        let path = self.file_path(key);
        info!("Loading dense {} matrix from {:?}", key, path);

        // Read all batches from Lance (may span multiple batches for large datasets)
        let uri = Self::path_to_uri(&path)?;
        let batch = self.read_lance_all_batches_async(uri).await?;

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

                let tmp_storage = Self::new(parent, String::from("tmp_storage"));

                // Reuse the async Lance reader logic.
                let uri = Self::path_to_uri(path)?;
                let batch = tmp_storage.read_lance_all_batches_async(uri).await?;
                let matrix = tmp_storage.from_dense_record_batch(&batch)?;
                info!(
                    "Loaded dense matrix from Lance: {} x {}",
                    matrix.shape().0,
                    matrix.shape().1
                );
                Ok(matrix)
            }
            "parquet" => {
                use parquet::arrow::arrow_reader::ParquetRecordBatchReaderBuilder;

                // Parquet readers require a synchronous `Read` impl; run the
                // blocking open/read/concat on the dedicated blocking pool so
                // the async executor thread is not stalled (issue #52).
                let owned_path = path.to_path_buf();
                let combined =
                    tokio::task::spawn_blocking(move || -> StorageResult<RecordBatch> {
                        let file = std::fs::File::open(&owned_path).map_err(|e| {
                            StorageError::Io(format!("Failed to open parquet file: {}", e))
                        })?;

                        let builder =
                            ParquetRecordBatchReaderBuilder::try_new(file).map_err(|e| {
                                StorageError::Parquet(format!(
                                    "Failed to create parquet reader: {}",
                                    e
                                ))
                            })?;
                        let reader = builder.build().map_err(|e| {
                            StorageError::Parquet(format!("Failed to build parquet reader: {}", e))
                        })?;

                        let batches: Vec<RecordBatch> =
                            reader.collect::<Result<Vec<_>, _>>().map_err(|e| {
                                StorageError::Parquet(format!(
                                    "Failed to read parquet batch: {}",
                                    e
                                ))
                            })?;

                        if batches.is_empty() {
                            return Err(StorageError::Invalid(format!(
                                "Empty parquet dataset at {:?}",
                                owned_path
                            )));
                        }

                        let schema = batches[0].schema();
                        arrow::compute::concat_batches(&schema, &batches).map_err(|e| {
                            StorageError::Parquet(format!(
                                "Failed to concatenate parquet batches: {}",
                                e
                            ))
                        })
                    })
                    .await
                    .map_err(|e| {
                        StorageError::Io(format!("Parquet reader task failed: {}", e))
                    })??;

                // 2. Detect layout: vector (FixedSizeList) vs old wide columnar (col_* Float64)
                let schema = combined.schema();
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

                    let tmp_storage = Self::new(parent, String::from("tmp_storage"));
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
                        let arr = col.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
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
            .join(format!("{}_{}.lance", self.name, key))
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

        let filetype = FileInfo::which_filetype(key)?;
        {
            let mut metadata = self.load_metadata().await?;
            metadata = metadata.add_file(
                key,
                FileInfo::new(
                    format!("{}_{}.lance", self.get_name(), key),
                    filetype.as_str(),
                    (matrix.rows(), matrix.cols()),
                    Some(matrix.nnz()),
                    None,
                )?,
            );
            self.save_metadata(&metadata).await?;

            let batch = self.to_sparse_record_batch(matrix)?;
            let uri = Self::path_to_uri(&path)?;
            self.write_lance_batch_async(uri, batch).await?;
        }
        info!("Sparse matrix {} saved successfully", filetype);
        Ok(())
    }

    async fn load_sparse(&self, key: &str) -> StorageResult<CsMat<f64>> {
        info!("Loading sparse {} matrix", key);

        let metadata = self.load_metadata().await?;
        let filetype = FileInfo::which_filetype(key)?;
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
        let uri = Self::path_to_uri(&path)?;
        let batch = self.read_lance_all_batches_async(uri).await?;
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
        info!("Saving {} lambda values", lambdas.len());
        self.save_primitive_column::<Float64Type>("lambdas", "lambda", lambdas.to_vec(), md_path)
            .await
    }

    async fn load_lambdas(&self) -> StorageResult<Vec<f64>> {
        let path = self.file_path("lambdas");
        info!("Loading lambda values from {:?}", path);

        let uri = Self::path_to_uri(&path)?;
        let batch = self.read_lance_all_batches_async(uri).await?;
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Float64Array>()
            .ok_or_else(|| StorageError::Invalid("lambda column type mismatch".into()))?;

        let lambdas: Vec<f64> = (0..arr.len()).map(|i| arr.value(i)).collect();
        info!("Loaded {} lambda values", lambdas.len());
        Ok(lambdas)
    }

    async fn save_vector(&self, key: &str, vector: &[f64], md_path: &Path) -> StorageResult<()> {
        info!("Saving {} values for vector {}", vector.len(), key);
        self.save_primitive_column::<Float64Type>(key, "element", vector.to_vec(), md_path)
            .await
    }

    async fn save_index(&self, key: &str, vector: &[usize], md_path: &Path) -> StorageResult<()> {
        info!("Saving {} values for index {}", vector.len(), key);
        // Checked cast: usize values above u32::MAX must not be silently
        // truncated (issue #51).
        let values = checked_u32_values(vector, "index")?;
        self.save_primitive_column::<UInt32Type>(key, "id", values, md_path)
            .await
    }

    async fn load_vector(&self, filename: &str) -> StorageResult<Vec<f64>> {
        let path = self.file_path(filename);
        info!("Loading vector {} from {:?}", filename, path);

        let uri = Self::path_to_uri(&path)?;
        let batch = self.read_lance_all_batches_async(uri).await?;
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

        let uri = Self::path_to_uri(&path)?;
        let batch = self.read_lance_all_batches_async(uri).await?;
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
    async fn save_dense_to_file(data: &DenseMatrix<f64>, path: &Path) -> StorageResult<()> {
        info!("Saving dense matrix to file (async): {:?}", path);

        // Create missing parent directories instead of failing late in the
        // format-specific writers.
        let parent = path
            .parent()
            .filter(|p| !p.as_os_str().is_empty())
            .ok_or_else(|| {
                StorageError::Invalid(format!("path has no parent directory: {:?}", path))
            })?;
        tokio::fs::create_dir_all(parent)
            .await
            .map_err(|e| StorageError::Io(format!("create dir {:?}: {}", parent, e)))?;
        let parent_str = parent
            .to_str()
            .ok_or_else(|| StorageError::Invalid(format!("non-UTF8 parent path for {:?}", path)))?
            .to_string();

        // Temporary storage, only used to build the record batch / write lance.
        let tmp_storage = Self::new(parent_str, String::from("tmp_storage"));

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

                let uri = Self::path_to_uri(path)?;
                tmp_storage.write_lance_batch_async(uri, batch).await?;
                info!("Saved dense matrix to Lance: {} x {}", n_rows, n_cols);
                Ok(())
            }
            "parquet" => {
                use parquet::arrow::ArrowWriter;
                use parquet::file::properties::WriterProperties;
                use std::fs::File;

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

                // The parquet writer is synchronous: run it on the blocking
                // pool so the async executor thread is not stalled (see #52
                // for the matching read path).
                let owned_path = path.to_path_buf();
                tokio::task::spawn_blocking(move || -> StorageResult<()> {
                    let file = File::create(&owned_path).map_err(|e| {
                        StorageError::Io(format!("Failed to create parquet file: {}", e))
                    })?;

                    let props = WriterProperties::builder()
                        .set_compression(parquet::basic::Compression::SNAPPY)
                        .build();

                    let mut writer = ArrowWriter::try_new(file, batch.schema(), Some(props))
                        .map_err(|e| {
                            StorageError::Parquet(format!("Failed to create parquet writer: {}", e))
                        })?;

                    writer.write(&batch).map_err(|e| {
                        StorageError::Parquet(format!("Failed to write batch: {}", e))
                    })?;

                    writer.close().map_err(|e| {
                        StorageError::Parquet(format!("Failed to close writer: {}", e))
                    })?;

                    Ok(())
                })
                .await
                .map_err(|e| StorageError::Io(format!("parquet writer task failed: {}", e)))??;

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
        info!("Saving {} centroid map entries", map.len());
        // Checked cast: usize values above u32::MAX must not be silently
        // truncated (issue #51).
        let values = checked_u32_values(map, "centroid map")?;
        self.save_primitive_column::<UInt32Type>("centroid_map", "centroid_id", values, md_path)
            .await
    }

    /// Load centroid_map
    async fn load_centroid_map(&self) -> StorageResult<Vec<usize>> {
        let path = self.file_path("centroid_map");
        info!("Loading centroid map from {:?}", path);

        let uri = Self::path_to_uri(&path)?;
        let batch = self.read_lance_all_batches_async(uri).await?;
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
        info!("Saving {} subcentroid lambda values", lambdas.len());
        self.save_primitive_column::<Float64Type>(
            "subcentroid_lambdas",
            "subcentroid_lambda",
            lambdas.to_vec(),
            md_path,
        )
        .await
    }

    /// Load subcentroid_lambdas
    async fn load_subcentroid_lambdas(&self) -> StorageResult<Vec<f64>> {
        let path = self.file_path("subcentroid_lambdas");
        info!("Loading subcentroid lambda values from {:?}", path);

        let uri = Self::path_to_uri(&path)?;
        let batch = self.read_lance_all_batches_async(uri).await?;
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
        let key = "sub_centroids";
        let path = self.file_path(key);
        let (n_rows, n_cols) = subcentroids.shape();
        info!(
            "Saving subcentroids matrix {} x {} at {:?}",
            n_rows, n_cols, path
        );

        let batch = self.to_dense_record_batch(subcentroids)?;
        {
            let mut metadata = self.load_metadata().await?;
            metadata = metadata.add_file(
                key,
                FileInfo::new(
                    format!("{}_{}.lance", self.get_name(), key),
                    "vector",
                    subcentroids.shape(),
                    None,
                    None,
                )?,
            );
            self.save_metadata(&metadata).await?;

            let uri = Self::path_to_uri(&path)?;
            self.write_lance_batch_async(uri, batch).await?;
        }
        debug!("Subcentroids matrix saved successfully");
        Ok(())
    }

    /// Load subcentroids as Vec<Vec<f64>>
    async fn load_subcentroids(&self) -> StorageResult<Vec<Vec<f64>>> {
        let path = self.file_path("sub_centroids");
        info!("Loading sub_centroids from {:?}", path);

        let uri = Self::path_to_uri(&path)?;
        let batch = self.read_lance_all_batches_async(uri).await?;
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
        info!("Saving {} item norm values", item_norms.len());
        self.save_primitive_column::<Float64Type>(
            "item_norms",
            "norm",
            item_norms.to_vec(),
            md_path,
        )
        .await
    }

    /// Load item norms vector
    async fn load_item_norms(&self) -> StorageResult<Vec<f64>> {
        let path = self.file_path("item_norms");
        info!("Loading item norms from {:?}", path);

        let uri = Self::path_to_uri(&path)?;
        let batch = self.read_lance_all_batches_async(uri).await?;
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
        info!("Saving {} cluster assignments", assignments.len());

        // Convert Option<usize> to i64 (-1 for None)
        let values: Vec<i64> = assignments
            .iter()
            .map(|opt| opt.map(|v| v as i64).unwrap_or(-1))
            .collect();

        self.save_primitive_column::<Int64Type>(
            "cluster_assignments",
            "cluster_id",
            values,
            md_path,
        )
        .await
    }

    async fn load_cluster_assignments(&self) -> StorageResult<Vec<Option<usize>>> {
        use arrow::array::Int64Array;
        let path = self.file_path("cluster_assignments");
        info!("Loading cluster assignments from {:?}", path);

        let uri = Self::path_to_uri(&path)?;
        let batch = self.read_lance_all_batches_async(uri).await?;
        let arr = batch
            .column(0)
            .as_any()
            .downcast_ref::<Int64Array>()
            .ok_or_else(|| StorageError::Invalid("cluster_id column type mismatch".into()))?;

        let assignments: Vec<Option<usize>> = (0..arr.len())
            .map(|i| {
                let v = arr.value(i);
                if v < 0 { None } else { Some(v as usize) }
            })
            .collect();
        info!("Loaded {} cluster assignments", assignments.len());
        Ok(assignments)
    }
}
