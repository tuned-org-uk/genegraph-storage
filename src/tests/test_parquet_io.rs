use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{Array as ArrowArray, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use parquet::arrow::ArrowWriter;
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::StorageError;
use crate::lance_storage_graph::LanceStorageGraph;
use crate::traits::backend::StorageBackend;

fn tmp_dir_sync(name: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    let unique = format!(
        "{}_{}",
        name,
        uuid::Uuid::new_v4().to_string().replace('-', "")
    );
    d.push(format!(
        "{}_{}",
        unique,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    ));
    std::fs::create_dir_all(&d).unwrap();
    d
}

fn sample_matrix() -> DenseMatrix<f64> {
    // 4 rows x 3 cols, distinct values
    let data: Vec<f64> = (0..12).map(|i| i as f64 * 0.25 - 1.0).collect();
    DenseMatrix::new(4, 3, data, true).unwrap()
}

fn assert_matrix_eq(a: &DenseMatrix<f64>, b: &DenseMatrix<f64>) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    let (rows, cols) = a.shape();
    for r in 0..rows {
        for c in 0..cols {
            let (x, y) = (*a.get((r, c)), *b.get((r, c)));
            assert!(
                (x - y).abs() < 1e-12,
                "value mismatch at ({r},{c}): {x} != {y}"
            );
        }
    }
}

fn rt<F: std::future::Future>(fut: F) -> F::Output {
    tokio::runtime::Runtime::new().unwrap().block_on(fut)
}

fn storage_for(dir: &Path) -> LanceStorageGraph {
    LanceStorageGraph::new(dir.to_string_lossy().to_string(), "pq_test".to_string())
}

/// Round-trip through the vector layout (`vector: FixedSizeList<Float64>`),
/// which is what `save_dense_to_file` writes.
#[test]
fn parquet_roundtrip_vector_layout() {
    let data = sample_matrix();
    let dir = tmp_dir_sync("pq_vector");
    let path = dir.join("dense.parquet");

    rt(LanceStorageGraph::save_dense_to_file(&data, &path)).expect("save parquet");
    let loaded = rt(storage_for(&dir).load_dense_from_file(&path)).expect("load parquet");
    assert_matrix_eq(&data, &loaded);
}

/// Round-trip through the legacy wide layout (`col_0..col_N` Float64 columns).
#[test]
fn parquet_roundtrip_wide_layout() {
    let data = sample_matrix();
    let dir = tmp_dir_sync("pq_wide");
    let path = dir.join("wide.parquet");

    // Hand-write the wide layout: one Float64 column per matrix column.
    let (rows, cols) = data.shape();
    let fields: Vec<_> = (0..cols)
        .map(|c| Field::new(format!("col_{c}"), DataType::Float64, false))
        .collect();
    let schema = Arc::new(Schema::new(fields));
    let mut columns: Vec<Arc<dyn ArrowArray>> = Vec::with_capacity(cols);
    for c in 0..cols {
        let values: Vec<f64> = (0..rows).map(|r| *data.get((r, c))).collect();
        columns.push(Arc::new(Float64Array::from(values)));
    }
    let batch = RecordBatch::try_new(schema, columns).unwrap();

    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let loaded = rt(storage_for(&dir).load_dense_from_file(&path)).expect("load wide parquet");
    assert_matrix_eq(&data, &loaded);
}

/// Parquet files whose schema matches neither supported layout must be
/// rejected with a typed error.
#[test]
fn parquet_rejects_unsupported_schema() {
    let dir = tmp_dir_sync("pq_bad");
    let path = dir.join("strings.parquet");

    let schema = Arc::new(Schema::new(vec![Field::new("name", DataType::Utf8, false)]));
    let batch = RecordBatch::try_new(
        schema,
        vec![Arc::new(arrow::array::StringArray::from(vec!["a", "b"]))],
    )
    .unwrap();
    let file = std::fs::File::create(&path).unwrap();
    let mut writer = ArrowWriter::try_new(file, batch.schema(), None).unwrap();
    writer.write(&batch).unwrap();
    writer.close().unwrap();

    let err = rt(storage_for(&dir).load_dense_from_file(&path)).unwrap_err();
    assert!(
        matches!(err, StorageError::Invalid(_)),
        "expected Invalid for unsupported parquet schema, got {err:?}"
    );
}

/// `save_dense_to_file` must create missing parent directories instead of
/// failing with an IO error.
#[test]
fn parquet_save_creates_missing_parent_dir() {
    let data = sample_matrix();
    let dir = tmp_dir_sync("pq_mkdir");
    let path = dir.join("a").join("b").join("dense.parquet");

    rt(LanceStorageGraph::save_dense_to_file(&data, &path)).expect("save creates parents");
    assert!(path.exists(), "parquet file must exist after save");
    let loaded = rt(storage_for(&dir).load_dense_from_file(&path)).expect("reload");
    assert_matrix_eq(&data, &loaded);
}

/// A path without a parent directory must produce an error, not a panic.
#[test]
#[cfg(unix)]
fn parquet_save_root_path_returns_error() {
    let data = sample_matrix();
    let err = rt(LanceStorageGraph::save_dense_to_file(&data, Path::new("/")))
        .expect_err("root path must error");
    assert!(
        matches!(err, StorageError::Invalid(_)),
        "expected Invalid for parent-less path, got {err:?}"
    );
}
