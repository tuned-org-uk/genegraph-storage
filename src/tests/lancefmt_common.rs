use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Arc;

use arrow::array::{ArrayRef, FixedSizeListArray, Float64Array, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

pub const FIXTURES_DIR: &str = "tests/fixtures/lancefmt";

pub(crate) fn fixture_path(name: &str) -> PathBuf {
    Path::new(FIXTURES_DIR).join(format!("{name}.lance"))
}

pub(crate) fn fixture_uri(name: &str) -> String {
    let abs = std::fs::canonicalize(fixture_path(name)).expect("fixture must exist");
    url::Url::from_file_path(&abs)
        .expect("fixture path is absolute")
        .to_string()
}

/// Deterministic formulas shared by the generator and the conformance tests:
/// the tests recompute the expected values instead of trusting the fixtures.
pub(crate) const F64_SMALL_ROWS: usize = 10;
pub(crate) const F64_MULTIPAGE_BATCHES: usize = 5;
pub(crate) const F64_MULTIPAGE_ROWS_PER_BATCH: usize = 1_000;
pub(crate) const U32_ROWS: usize = 1_000;
pub(crate) const FSL_ROWS: usize = 50;
pub(crate) const FSL_DIMS: i32 = 8;
pub(crate) const SPARSE_TRIPLETS: usize = 1_000;

pub(crate) fn f64_schema() -> Schema {
    Schema::new(vec![Field::new("lambda", DataType::Float64, false)])
}

pub(crate) fn f64_batch() -> RecordBatch {
    f64_batch_from(f64_small_values())
}

pub(crate) fn f64_batch_from(values: Vec<f64>) -> RecordBatch {
    batch_of(
        f64_schema(),
        vec![Arc::new(Float64Array::from(values)) as ArrayRef],
    )
}

pub(crate) fn f64_multipage_batch(b: usize) -> Vec<f64> {
    (0..F64_MULTIPAGE_ROWS_PER_BATCH)
        .map(|i| ((b * 31 + i) % 977) as f64 / 7.0)
        .collect()
}

pub(crate) fn f64_small_values() -> Vec<f64> {
    (0..F64_SMALL_ROWS).map(|i| i as f64 * 0.5).collect()
}

pub(crate) fn u32_schema() -> Schema {
    Schema::new(vec![Field::new("id", DataType::UInt32, false)])
}

pub(crate) fn u32_batch() -> RecordBatch {
    batch_of(
        u32_schema(),
        vec![Arc::new(UInt32Array::from(u32_values())) as ArrayRef],
    )
}

pub(crate) fn u32_values() -> Vec<u32> {
    (0..U32_ROWS).map(|i| (i as u32).wrapping_mul(31)).collect()
}

pub(crate) fn fsl_values() -> Vec<f64> {
    (0..FSL_ROWS)
        .flat_map(|i| {
            (0..FSL_DIMS).map(move |j| (i * FSL_DIMS as usize + j as usize) as f64 * 0.125)
        })
        .collect()
}

pub(crate) fn fsl_schema() -> Schema {
    let value_field = Field::new("item", DataType::Float64, false);
    Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(Arc::new(value_field), FSL_DIMS),
        false,
    )])
}

pub(crate) fn fsl_batch() -> RecordBatch {
    let values = Float64Array::from(fsl_values());
    let list = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float64, false)),
        FSL_DIMS,
        Arc::new(values),
        None,
    );
    batch_of(fsl_schema(), vec![Arc::new(list) as ArrayRef])
}

pub(crate) fn sparse_schema() -> Schema {
    let mut metadata = HashMap::new();
    metadata.insert("rows".to_string(), "100".to_string());
    metadata.insert("cols".to_string(), "50".to_string());
    metadata.insert("nnz".to_string(), format!("{SPARSE_TRIPLETS}"));

    Schema::new(vec![
        Field::new("row", DataType::UInt32, false),
        Field::new("col", DataType::UInt32, false),
        Field::new("value", DataType::Float64, false),
    ])
    .with_metadata(metadata)
}

pub(crate) fn sparse_batch() -> RecordBatch {
    let rows: Vec<u32> = (0..SPARSE_TRIPLETS).map(|i| (i % 100) as u32).collect();
    let cols: Vec<u32> = (0..SPARSE_TRIPLETS).map(|i| (i % 50) as u32).collect();
    let vals: Vec<f64> = (0..SPARSE_TRIPLETS).map(|i| i as f64 + 1.0).collect();

    batch_of(
        sparse_schema(),
        vec![
            Arc::new(UInt32Array::from(rows)) as ArrayRef,
            Arc::new(UInt32Array::from(cols)) as ArrayRef,
            Arc::new(Float64Array::from(vals)) as ArrayRef,
        ],
    )
}

fn batch_of(schema: Schema, columns: Vec<ArrayRef>) -> RecordBatch {
    RecordBatch::try_new(Arc::new(schema), columns).expect("fixture batch is well-formed")
}
