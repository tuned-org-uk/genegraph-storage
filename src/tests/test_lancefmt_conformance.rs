use arrow::array::{Array, FixedSizeListArray, Float64Array, UInt32Array};
use arrow::datatypes::DataType;
use arrow::record_batch::RecordBatch;
use futures::StreamExt;
use lance::Dataset;

use crate::tests::lancefmt_common as fx;

/// Mirrors `LanceStorage::read_lance_all_batches_async`: open with the
/// official crate, read ALL batches, concat into one batch.
async fn read_all(name: &str) -> RecordBatch {
    let uri = fx::fixture_uri(name);
    let dataset = Dataset::open(&uri).await.expect("official lance open");
    let scanner = dataset.scan();
    let mut stream = scanner.try_into_stream().await.expect("scan");

    let mut batches = Vec::new();
    while let Some(batch_result) = stream.next().await {
        batches.push(batch_result.expect("batch read"));
    }
    assert!(!batches.is_empty(), "fixture {name} must be non-empty");

    let schema = batches[0].schema();
    arrow::compute::concat_batches(&schema, &batches).expect("concat")
}

fn assert_column_f64(actual: &RecordBatch, expected: &[f64], column: usize) {
    let arr = actual
        .column(column)
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("f64 column");
    assert_eq!(arr.len(), expected.len(), "row count mismatch");
    for (i, e) in expected.iter().enumerate() {
        assert_eq!(arr.value(i), *e, "value mismatch at row {i}");
    }
}

fn assert_column_u32(actual: &RecordBatch, expected: &[u32], column: usize) {
    let arr = actual
        .column(column)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .expect("u32 column");
    assert_eq!(arr.len(), expected.len(), "row count mismatch");
    for (i, e) in expected.iter().enumerate() {
        assert_eq!(arr.value(i), *e, "value mismatch at row {i}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn conformance_float64_nonnull() {
    let batch = read_all("float64_nonnull").await;
    let expected = fx::f64_batch();
    assert_eq!(batch.schema(), expected.schema());
    assert_column_f64(&batch, &fx::f64_small_values(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn conformance_float64_multipage() {
    let batch = read_all("float64_multipage").await;

    let mut expected: Vec<f64> = Vec::new();
    for b in 0..fx::F64_MULTIPAGE_BATCHES {
        expected.extend(fx::f64_multipage_batch(b));
    }
    // The fixture was written with LANCE_FILE_WRITER_MAX_PAGE_BYTES=1024:
    // exact value equality across the whole range proves the M2 reader must
    // handle page boundaries in order (the official writer produced many).
    assert_column_f64(&batch, &expected, 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn conformance_uint32_nonnull() {
    let batch = read_all("uint32_nonnull").await;
    let expected = fx::u32_batch();
    assert_eq!(batch.schema(), expected.schema());
    assert_column_u32(&batch, &fx::u32_values(), 0);
}

#[tokio::test(flavor = "multi_thread")]
async fn conformance_int64_nonnull() {
    let batch = read_all("int64_nonnull").await;
    let expected = fx::i64_batch();
    assert_eq!(batch.schema(), expected.schema());

    let arr = batch
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .expect("i64 column");
    let expected_values = fx::i64_values();
    assert_eq!(arr.len(), expected_values.len());
    for (i, e) in expected_values.iter().enumerate() {
        assert_eq!(arr.value(i), *e, "value mismatch at row {i}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn conformance_fsl_f64_nonnull() {
    let batch = read_all("fsl_f64_nonnull").await;

    // Conformance finding (M0): the official writer does NOT preserve the
    // nullability of the FixedSizeList CHILD field (item: Float64 is read
    // back nullable). Top-level field nullability and list size are
    // preserved. Our reader/writer must match this behavior, so the schema
    // assertion here intentionally tolerates child nullability.
    let schema = batch.schema();
    let field = schema.field(0);
    assert_eq!(field.name(), "vector");
    assert!(!field.is_nullable(), "top-level field must stay non-null");
    assert!(matches!(
        field.data_type(),
        DataType::FixedSizeList(inner, 8) if matches!(inner.data_type(), DataType::Float64)
    ));

    let list = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("FixedSizeList column");
    assert_eq!(list.len(), fx::FSL_ROWS);
    assert_eq!(list.value_length(), fx::FSL_DIMS);

    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("f64 values child");
    let expected_values = fx::fsl_values();
    for (i, e) in expected_values.iter().enumerate() {
        assert_eq!(values.value(i), *e, "value mismatch at flat index {i}");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn conformance_sparse_triplet_schema_metadata() {
    let batch = read_all("sparse_triplet_meta").await;
    let expected = fx::sparse_batch();

    // Schema metadata (rows/cols/nnz) must survive a lance round-trip:
    // `from_sparse_record_batch` validates dimensions against it (#46).
    let schema = batch.schema();
    let metadata = schema.metadata();
    assert_eq!(
        metadata.get("rows").map(String::as_str),
        Some("100"),
        "rows metadata must round-trip"
    );
    assert_eq!(
        metadata.get("cols").map(String::as_str),
        Some("50"),
        "cols metadata must round-trip"
    );
    assert_eq!(
        metadata.get("nnz").map(String::as_str),
        Some("1000"),
        "nnz metadata must round-trip"
    );

    assert_eq!(batch.schema(), expected.schema());
    let rows: Vec<u32> = (0..fx::SPARSE_TRIPLETS).map(|i| (i % 100) as u32).collect();
    let cols: Vec<u32> = (0..fx::SPARSE_TRIPLETS).map(|i| (i % 50) as u32).collect();
    let vals: Vec<f64> = (0..fx::SPARSE_TRIPLETS).map(|i| i as f64 + 1.0).collect();
    assert_column_u32(&batch, &rows, 0);
    assert_column_u32(&batch, &cols, 1);
    assert_column_f64(&batch, &vals, 2);
}
