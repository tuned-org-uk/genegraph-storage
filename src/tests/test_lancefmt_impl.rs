use std::path::{Path, PathBuf};

use arrow::array::{Array, FixedSizeListArray, Float64Array, UInt32Array};
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

#[test]
fn impl_overwrite_bumps_manifest_version() {
    // M4: overwriting an existing dataset must write a NEW manifest version
    // (readers see the latest), not clobber version 1.
    let dir = tmp_dir_sync("lancefmt_impl_overwrite");
    write_dataset(&fx::f64_batch(), &dir).expect("first write");
    write_dataset(&fx::f64_batch_from(vec![9.5, -2.0, 7.25]), &dir).expect("overwrite");

    let versions: Vec<String> = std::fs::read_dir(dir.join("_versions"))
        .unwrap()
        .map(|e| e.unwrap().file_name().to_string_lossy().to_string())
        .collect();
    assert!(
        versions.iter().any(|v| v == "1.manifest"),
        "version 1 must be preserved, got {versions:?}"
    );
    assert!(
        versions.iter().any(|v| v == "2.manifest"),
        "overwrite must create version 2, got {versions:?}"
    );
    let hint = std::fs::read_to_string(dir.join("_versions/latest_version_hint.json")).unwrap();
    assert_eq!(
        hint, "{\"version\":2}",
        "hint must point at the new version"
    );

    let loaded = scan_all(&dir).expect("scan after overwrite");
    assert_batch_f64(&loaded, &[9.5, -2.0, 7.25], 0, "overwrite round-trip");
}

#[test]
fn impl_roundtrip_int64_nonnull() {
    let batch = fx::i64_batch();
    let dir = tmp_dir_sync("lancefmt_impl_i64");
    write_dataset(&batch, &dir).expect("write");
    let loaded = scan_all(&dir).expect("our scan");
    assert_batch_i64(&loaded, &fx::i64_values(), 0, "ours->ours i64");
}

fn assert_batch_i64(batch: &RecordBatch, expected: &[i64], col: usize, what: &str) {
    let arr = batch
        .column(col)
        .as_any()
        .downcast_ref::<arrow::array::Int64Array>()
        .unwrap_or_else(|| panic!("{what}: expected i64 column"));
    assert_eq!(arr.len(), expected.len(), "{what}: row count");
    for (i, e) in expected.iter().enumerate() {
        assert_eq!(arr.value(i), *e, "{what}: value mismatch at row {i}");
    }
}

// ---------------------------------------------------------------------------
// RFC #81-P2: width-generic leaves (Float32 / UInt64 / UInt8 Flat, FSL<Float32>)
// ---------------------------------------------------------------------------

fn f32_values() -> Vec<f32> {
    // bit-exact round-trip: include negatives, subnormals and infinities
    [
        0.0f32,
        -0.0,
        1.5,
        -2.25,
        f32::MIN_POSITIVE / 4.0,
        f32::MAX,
        f32::MIN,
        f32::INFINITY,
        0.123_456_79,
        -0.9999999,
    ]
    .to_vec()
}

fn assert_batch_f32(batch: &RecordBatch, expected: &[f32], col: usize, what: &str) {
    let arr = batch
        .column(col)
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .unwrap_or_else(|| panic!("{what}: expected f32 column"));
    assert_eq!(arr.len(), expected.len(), "{what}: row count");
    for (i, e) in expected.iter().enumerate() {
        assert_eq!(
            arr.value(i).to_bits(),
            e.to_bits(),
            "{what}: bit-exact value mismatch at row {i}"
        );
    }
}

#[test]
fn impl_roundtrip_float32_nonnull() {
    let values = f32_values();
    let arr = arrow::array::Float32Array::from(values.clone());
    let schema = arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "weight",
        arrow::datatypes::DataType::Float32,
        false,
    )]);
    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![std::sync::Arc::new(arr) as _],
    )
    .unwrap();
    let dir = tmp_dir_sync("lancefmt_impl_f32");
    write_dataset(&batch, &dir).expect("write");
    let loaded = scan_all(&dir).expect("our scan");
    assert_batch_f32(&loaded, &values, 0, "ours->ours f32");
}

#[test]
fn impl_roundtrip_uint64_nonnull() {
    let values: Vec<u64> = vec![0, 1, u32::MAX as u64, u32::MAX as u64 + 7, u64::MAX];
    let arr = arrow::array::UInt64Array::from(values.clone());
    let schema = arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "id",
        arrow::datatypes::DataType::UInt64,
        false,
    )]);
    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![std::sync::Arc::new(arr) as _],
    )
    .unwrap();
    let dir = tmp_dir_sync("lancefmt_impl_u64");
    write_dataset(&batch, &dir).expect("write");
    let loaded = scan_all(&dir).expect("our scan");
    let out = loaded
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::UInt64Array>()
        .expect("u64 column");
    for (i, e) in values.iter().enumerate() {
        assert_eq!(out.value(i), *e, "u64 value mismatch at {i}");
    }
}

#[test]
fn impl_roundtrip_uint8_nonnull() {
    let values: Vec<u8> = vec![0, 1, 42, 127, 128, 254, 255];
    let arr = arrow::array::UInt8Array::from(values.clone());
    let schema = arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "q",
        arrow::datatypes::DataType::UInt8,
        false,
    )]);
    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![std::sync::Arc::new(arr) as _],
    )
    .unwrap();
    let dir = tmp_dir_sync("lancefmt_impl_u8");
    write_dataset(&batch, &dir).expect("write");
    let loaded = scan_all(&dir).expect("our scan");
    let out = loaded
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::UInt8Array>()
        .expect("u8 column");
    for (i, e) in values.iter().enumerate() {
        assert_eq!(out.value(i), *e, "u8 value mismatch at {i}");
    }
}

#[test]
fn impl_roundtrip_fsl_float32_nonnull() {
    let dim = 3i32;
    let values: Vec<f32> = (0..9).map(|i| i as f32 * 0.5 - 1.0).collect();
    let child = std::sync::Arc::new(arrow::datatypes::Field::new(
        "item",
        arrow::datatypes::DataType::Float32,
        false,
    ));
    let list = FixedSizeListArray::new(
        child,
        dim,
        std::sync::Arc::new(arrow::array::Float32Array::from(values.clone())),
        None,
    );
    let schema = arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "vector",
        arrow::datatypes::DataType::FixedSizeList(
            std::sync::Arc::new(arrow::datatypes::Field::new(
                "item",
                arrow::datatypes::DataType::Float32,
                false,
            )),
            dim,
        ),
        false,
    )]);
    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![std::sync::Arc::new(list) as _],
    )
    .unwrap();
    let dir = tmp_dir_sync("lancefmt_impl_fsl_f32");
    write_dataset(&batch, &dir).expect("write");
    let loaded = scan_all(&dir).expect("our scan");
    let list = loaded
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("fsl column");
    assert_eq!(list.value_length(), dim);
    let out = list
        .values()
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .expect("fsl child f32");
    for (i, e) in values.iter().enumerate() {
        assert_eq!(
            out.value(i).to_bits(),
            e.to_bits(),
            "fsl f32 item mismatch at {i}"
        );
    }
}

/// Graph edge-list columns (RFC #81-P3) across chunk boundaries: 2500 rows
/// with `src`/`dst` UInt32 and `weight` Float32 exercise multiple 512/1024-
/// value chunks in one page.
#[test]
fn impl_roundtrip_graph_edge_list_multichunk() {
    let n = 2500usize;
    let src: Vec<u32> = (0..n as u32).map(|i| i % 97).collect();
    let dst: Vec<u32> = (0..n as u32).map(|i| (i * 31 + 5) % 97).collect();
    let weight: Vec<f32> = (0..n as u32).map(|i| (i as f32 / 97.0) - 12.5).collect();

    let schema = arrow::datatypes::Schema::new(vec![
        arrow::datatypes::Field::new("src", arrow::datatypes::DataType::UInt32, false),
        arrow::datatypes::Field::new("dst", arrow::datatypes::DataType::UInt32, false),
        arrow::datatypes::Field::new("weight", arrow::datatypes::DataType::Float32, false),
    ]);
    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![
            std::sync::Arc::new(arrow::array::UInt32Array::from(src.clone())) as _,
            std::sync::Arc::new(arrow::array::UInt32Array::from(dst.clone())) as _,
            std::sync::Arc::new(arrow::array::Float32Array::from(weight.clone())) as _,
        ],
    )
    .unwrap();
    let dir = tmp_dir_sync("lancefmt_impl_graph_edges");
    write_dataset(&batch, &dir).expect("write");
    let loaded = scan_all(&dir).expect("our scan");
    assert_eq!(loaded.num_rows(), n);

    let s = loaded
        .column(0)
        .as_any()
        .downcast_ref::<arrow::array::UInt32Array>()
        .unwrap();
    let d = loaded
        .column(1)
        .as_any()
        .downcast_ref::<arrow::array::UInt32Array>()
        .unwrap();
    let w = loaded
        .column(2)
        .as_any()
        .downcast_ref::<arrow::array::Float32Array>()
        .unwrap();
    for i in 0..n {
        assert_eq!(s.value(i), src[i], "src mismatch at {i}");
        assert_eq!(d.value(i), dst[i], "dst mismatch at {i}");
        assert_eq!(
            w.value(i).to_bits(),
            weight[i].to_bits(),
            "weight mismatch at {i}"
        );
    }
}

/// #95: wide vectors where the 16KiB budget cannot hold two rows
/// (`rows_per_chunk` collapses to 1) must still round-trip — one page per
/// row, each page a single final chunk.
#[test]
fn impl_roundtrip_fsl_wide_vectors() {
    let dim = 3000i32; // 16KiB / (3000*8) == 0 -> rows_per_chunk would be 1
    let values: Vec<f64> = (0..dim * 3).map(|i| (i % 17) as f64 - 8.0).collect();
    let child = std::sync::Arc::new(arrow::datatypes::Field::new(
        "item",
        arrow::datatypes::DataType::Float64,
        false,
    ));
    let list = FixedSizeListArray::new(
        child.clone(),
        dim,
        std::sync::Arc::new(Float64Array::from(values.clone())),
        None,
    );
    let schema = arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "vector",
        arrow::datatypes::DataType::FixedSizeList(child, dim),
        false,
    )]);
    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![std::sync::Arc::new(list) as _],
    )
    .unwrap();
    let dir = tmp_dir_sync("lancefmt_impl_fsl_wide");
    write_dataset(&batch, &dir).expect("write wide fsl");
    let loaded = scan_all(&dir).expect("our scan");
    let out = loaded
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .expect("fsl column");
    assert_eq!(out.value_length(), dim);
    assert_eq!(out.len(), 3, "three wide rows");
    let flat = out
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .expect("fsl child f64");
    for (i, e) in values.iter().enumerate() {
        assert_eq!(flat.value(i), *e, "wide fsl value mismatch at {i}");
    }
}

/// A single row wider than the 32768-byte chunk-metadata limit is rejected
/// explicitly instead of being mis-encoded.
#[test]
fn impl_rejects_fsl_row_beyond_chunk_limit() {
    let dim = 4200i32; // 4200*8 = 33600 > 32760
    let child = std::sync::Arc::new(arrow::datatypes::Field::new(
        "item",
        arrow::datatypes::DataType::Float64,
        false,
    ));
    let list = FixedSizeListArray::new(
        child.clone(),
        dim,
        std::sync::Arc::new(Float64Array::from(vec![0.0; dim as usize])),
        None,
    );
    let schema = arrow::datatypes::Schema::new(vec![arrow::datatypes::Field::new(
        "vector",
        arrow::datatypes::DataType::FixedSizeList(child, dim),
        false,
    )]);
    let batch = RecordBatch::try_new(
        std::sync::Arc::new(schema),
        vec![std::sync::Arc::new(list) as _],
    )
    .unwrap();
    let dir = tmp_dir_sync("lancefmt_impl_fsl_too_wide");
    let err = write_dataset(&batch, &dir).unwrap_err();
    assert!(
        matches!(err, crate::StorageError::UnsupportedFormat(_)),
        "expected UnsupportedFormat, got {err:?}"
    );
}
