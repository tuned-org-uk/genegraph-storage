use std::sync::Arc;

use arrow::record_batch::{RecordBatch, RecordBatchIterator};

use crate::tests::lancefmt_common as fx;
use lance::Dataset;
use lance::dataset::{WriteMode, WriteParams};

/// Golden-fixture generator for the lancefmt conformance harness (#75 M0).
///
/// Explicitly invoked, never in CI:
/// `cargo test --release lancefmt_gen_fixtures -- --ignored --nocapture`
///
/// Writes files with the OFFICIAL lance crate into `tests/fixtures/lancefmt/`;
/// the generated directories are committed so the M2 in-house reader can be
/// tested against them deterministically (and later against the official
/// reader in dev-dependency round-trips).
#[tokio::test(flavor = "multi_thread")]
#[ignore = "explicit fixture generation: cargo test lancefmt_gen_fixtures -- --ignored"]
async fn lancefmt_gen_fixtures() {
    std::fs::create_dir_all(fx::FIXTURES_DIR).expect("fixtures dir");

    write_batches("float64_nonnull", fx::f64_schema(), vec![fx::f64_batch()]).await;

    // Small pages force the official writer to split the column across many
    // pages: the M2 reader must decode page boundaries correctly.
    write_batches(
        "float64_multipage",
        fx::f64_schema(),
        (0..fx::F64_MULTIPAGE_BATCHES)
            .map(|b| fx::f64_batch_from(fx::f64_multipage_batch(b)))
            .collect(),
    )
    .await;

    write_batches("uint32_nonnull", fx::u32_schema(), vec![fx::u32_batch()]).await;
    write_batches("int64_nonnull", fx::i64_schema(), vec![fx::i64_batch()]).await;
    write_batches("fsl_f64_nonnull", fx::fsl_schema(), vec![fx::fsl_batch()]).await;
    write_batches(
        "sparse_triplet_meta",
        fx::sparse_schema(),
        vec![fx::sparse_batch()],
    )
    .await;

    println!("fixtures written to {}", fx::FIXTURES_DIR);
}

async fn write_batches(name: &str, schema: arrow::datatypes::Schema, batches: Vec<RecordBatch>) {
    // SAFETY: single-threaded fixture generation; the env var tunes the
    // official writer's page size for the multipage fixture only.
    unsafe { std::env::set_var("LANCE_FILE_WRITER_MAX_PAGE_BYTES", "1024") };

    let path = fx::fixture_path(name);
    if path.exists() {
        std::fs::remove_dir_all(&path).expect("remove stale fixture");
    }
    // canonicalize the existing parent dir: the fixture itself is written
    // fresh, so only the directory is guaranteed to exist
    let abs_dir = std::fs::canonicalize(fx::FIXTURES_DIR).expect("fixtures dir");
    let uri = url::Url::from_file_path(abs_dir.join(format!("{name}.lance")))
        .unwrap()
        .to_string();

    let reader = RecordBatchIterator::new(batches.into_iter().map(Ok), Arc::new(schema));
    let params = WriteParams {
        mode: WriteMode::Create,
        ..WriteParams::default()
    };
    Dataset::write(reader, &uri, Some(params))
        .await
        .expect("official lance write");

    unsafe { std::env::remove_var("LANCE_FILE_WRITER_MAX_PAGE_BYTES") };
}
