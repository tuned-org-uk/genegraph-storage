use std::path::{Path, PathBuf};
use std::sync::Arc;

use futures::StreamExt;

use arrow::array::{Array, FixedSizeListArray, Float64Array, UInt32Array};
use arrow::datatypes::Schema as ArrowSchema;
use arrow::record_batch::RecordBatch;

use crate::lancefmt::{scan_all, write_dataset};
use crate::tests::lancefmt_common as fx;

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

fn read_with_official(dir: &Path) -> RecordBatch {
    let abs = std::fs::canonicalize(dir).expect("dataset dir exists");
    let uri = url::Url::from_file_path(&abs).unwrap().to_string();
    let rt = tokio::runtime::Runtime::new().unwrap();
    rt.block_on(async move {
        let dataset = lance::Dataset::open(&uri).await.expect("official open");
        let schema = Arc::new(ArrowSchema::from(dataset.schema()));
        let stream = dataset.scan().try_into_stream().await.expect("scan");
        let batches: Vec<RecordBatch> =
            futures::StreamExt::collect::<Vec<_>>(stream.map(|b| b.expect("batch"))).await;
        arrow::compute::concat_batches(&schema, &batches).expect("concat")
    })
}

fn assert_batch_f64(batch: &RecordBatch, expected: &[f64], col: usize, what: &str) {
    let arr = batch
        .column(col)
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap_or_else(|| panic!("{what}: expected f64 column"));
    assert_eq!(arr.len(), expected.len(), "{what}: row count");
    for (i, e) in expected.iter().enumerate() {
        assert_eq!(arr.value(i), *e, "{what}: value mismatch at row {i}");
    }
}

fn assert_batch_u32(batch: &RecordBatch, expected: &[u32], col: usize, what: &str) {
    let arr = batch
        .column(col)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap_or_else(|| panic!("{what}: expected u32 column"));
    assert_eq!(arr.len(), expected.len(), "{what}: row count");
    for (i, e) in expected.iter().enumerate() {
        assert_eq!(arr.value(i), *e, "{what}: value mismatch at row {i}");
    }
}

fn assert_batch_fsl(batch: &RecordBatch, expected: &[f64], dim: i32, what: &str) {
    let list = batch
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap_or_else(|| panic!("{what}: expected fsl column"));
    assert_eq!(list.value_length(), dim, "{what}: dimension");
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("fsl child f64");
    assert_eq!(values.len(), expected.len(), "{what}: item count");
    for (i, e) in expected.iter().enumerate() {
        assert_eq!(values.value(i), *e, "{what}: item mismatch at {i}");
    }
}

fn multipage_expected() -> Vec<f64> {
    let mut expected = Vec::new();
    for b in 0..fx::F64_MULTIPAGE_BATCHES {
        expected.extend(fx::f64_multipage_batch(b));
    }
    expected
}

#[test]
fn impl_roundtrip_float64_nonnull() {
    let batch = fx::f64_batch();
    let dir = tmp_dir_sync("lancefmt_impl_f64");
    write_dataset(&batch, &dir).expect("write");
    let loaded = scan_all(&dir).expect("our scan");
    assert_batch_f64(&loaded, &fx::f64_small_values(), 0, "ours->ours f64");
}

#[test]
fn impl_roundtrip_uint32_nonnull() {
    let batch = fx::u32_batch();
    let dir = tmp_dir_sync("lancefmt_impl_u32");
    write_dataset(&batch, &dir).expect("write");
    let loaded = scan_all(&dir).expect("our scan");
    assert_batch_u32(&loaded, &fx::u32_values(), 0, "ours->ours u32");
}

#[test]
fn impl_roundtrip_fsl_nonnull() {
    let batch = fx::fsl_batch();
    let dir = tmp_dir_sync("lancefmt_impl_fsl");
    write_dataset(&batch, &dir).expect("write");
    let loaded = scan_all(&dir).expect("our scan");
    assert_batch_fsl(&loaded, &fx::fsl_values(), fx::FSL_DIMS, "ours->ours fsl");
}

#[test]
fn impl_roundtrip_sparse_with_schema_metadata() {
    let batch = fx::sparse_batch();
    let dir = tmp_dir_sync("lancefmt_impl_sparse");
    write_dataset(&batch, &dir).expect("write");
    let loaded = scan_all(&dir).expect("our scan");

    let schema = loaded.schema();
    assert_eq!(
        schema.metadata().get("rows").map(String::as_str),
        Some("100")
    );
    assert_eq!(
        schema.metadata().get("cols").map(String::as_str),
        Some("50")
    );

    let rows: Vec<u32> = (0..fx::SPARSE_TRIPLETS).map(|i| (i % 100) as u32).collect();
    let cols: Vec<u32> = (0..fx::SPARSE_TRIPLETS).map(|i| (i % 50) as u32).collect();
    let vals: Vec<f64> = (0..fx::SPARSE_TRIPLETS).map(|i| i as f64 + 1.0).collect();
    assert_batch_u32(&loaded, &rows, 0, "ours->ours sparse row");
    assert_batch_u32(&loaded, &cols, 1, "ours->ours sparse col");
    assert_batch_f64(&loaded, &vals, 2, "ours->ours sparse value");
}

#[test]
fn impl_roundtrip_multichunk_f64() {
    let batch = fx::f64_multipage_record_batch();
    let dir = tmp_dir_sync("lancefmt_impl_multipage");
    write_dataset(&batch, &dir).expect("write");
    let loaded = scan_all(&dir).expect("our scan");
    assert_batch_f64(&loaded, &multipage_expected(), 0, "ours->ours multichunk");
}

#[test]
fn impl_official_reads_our_f64() {
    let batch = fx::f64_batch();
    let dir = tmp_dir_sync("lancefmt_impl_off_f64");
    write_dataset(&batch, &dir).expect("write");
    let loaded = read_with_official(&dir);
    assert_batch_f64(&loaded, &fx::f64_small_values(), 0, "ours->official f64");
}

#[test]
fn impl_official_reads_our_uint32() {
    let batch = fx::u32_batch();
    let dir = tmp_dir_sync("lancefmt_impl_off_u32");
    write_dataset(&batch, &dir).expect("write");
    let loaded = read_with_official(&dir);
    assert_batch_u32(&loaded, &fx::u32_values(), 0, "ours->official u32");
}

#[test]
fn impl_official_reads_our_fsl() {
    let batch = fx::fsl_batch();
    let dir = tmp_dir_sync("lancefmt_impl_off_fsl");
    write_dataset(&batch, &dir).expect("write");
    let loaded = read_with_official(&dir);
    assert_batch_fsl(
        &loaded,
        &fx::fsl_values(),
        fx::FSL_DIMS,
        "ours->official fsl",
    );
}

#[test]
fn impl_official_reads_our_multichunk() {
    let batch = fx::f64_multipage_record_batch();
    let dir = tmp_dir_sync("lancefmt_impl_off_multi");
    write_dataset(&batch, &dir).expect("write");
    let loaded = read_with_official(&dir);
    assert_batch_f64(
        &loaded,
        &multipage_expected(),
        0,
        "ours->official multichunk",
    );
}

#[test]
fn impl_official_reads_our_sparse_metadata() {
    let batch = fx::sparse_batch();
    let dir = tmp_dir_sync("lancefmt_impl_off_sparse");
    write_dataset(&batch, &dir).expect("write");
    let loaded = read_with_official(&dir);
    let schema = loaded.schema();
    assert_eq!(
        schema.metadata().get("rows").map(String::as_str),
        Some("100"),
        "official must read back schema metadata we wrote"
    );
}

#[test]
fn impl_our_reader_reads_official_fixtures_f64() {
    let dir = Path::new(fx::FIXTURES_DIR).join("float64_nonnull.lance");
    let loaded = scan_all(&dir).expect("our scan of official fixture");
    assert_batch_f64(&loaded, &fx::f64_small_values(), 0, "official->ours f64");
}

#[test]
fn impl_our_reader_reads_official_fixtures_uint32() {
    let dir = Path::new(fx::FIXTURES_DIR).join("uint32_nonnull.lance");
    let loaded = scan_all(&dir).expect("our scan of official fixture");
    assert_batch_u32(
        &loaded,
        &fx::u32_values(),
        0,
        "official->ours u32 (bitpacked)",
    );
}

#[test]
fn impl_our_reader_reads_official_fixtures_fsl() {
    let dir = Path::new(fx::FIXTURES_DIR).join("fsl_f64_nonnull.lance");
    let loaded = scan_all(&dir).expect("our scan of official fixture");
    assert_batch_fsl(
        &loaded,
        &fx::fsl_values(),
        fx::FSL_DIMS,
        "official->ours fsl",
    );
}

#[test]
fn impl_our_reader_reads_official_fixtures_sparse() {
    let dir = Path::new(fx::FIXTURES_DIR).join("sparse_triplet_meta.lance");
    let loaded = scan_all(&dir).expect("our scan of official fixture");
    let rows: Vec<u32> = (0..fx::SPARSE_TRIPLETS).map(|i| (i % 100) as u32).collect();
    assert_batch_u32(&loaded, &rows, 0, "official->ours sparse row");
}

#[test]
fn impl_our_reader_reads_official_fixtures_multichunk() {
    let dir = Path::new(fx::FIXTURES_DIR).join("float64_multipage.lance");
    let loaded = scan_all(&dir).expect("our scan of official fixture");
    assert_batch_f64(
        &loaded,
        &multipage_expected(),
        0,
        "official->ours multichunk",
    );
}
