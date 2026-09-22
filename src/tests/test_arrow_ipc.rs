//! Arrow-IPC interop path (#142): `.arrow` file format, `.arrows` streaming
//! format, generation scoping, and byte-layout conformance. Runs only with
//! the `arrow-ipc` Cargo feature (off by default).

use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::ipc::reader::FileReader;
use arrow::record_batch::RecordBatch;
use rand::rngs::StdRng;
use rand::{RngExt, SeedableRng};
use smartcore::linalg::basic::arrays::Array;
use smartcore::linalg::basic::matrix::DenseMatrix;

use crate::graph::{GraphEdge, StoredGraph};
use crate::lance_storage_graph::LanceStorageGraph;
use crate::metadata::{CollectionKind as MetaKind, GeneMetadata};
use crate::tests::tmp_dir;
use crate::traits::backend::StorageBackend;
use crate::traits::metadata::Metadata;

async fn seeded_storage(name: &str) -> (std::path::PathBuf, LanceStorageGraph) {
    let base = tmp_dir(name).await;
    let storage = LanceStorageGraph::new(base.to_string_lossy().to_string(), name.to_string())
        .expect("valid instance name");
    GeneMetadata::seed_metadata(name, 4, 4, &storage)
        .await
        .expect("seed metadata");
    (base, storage)
}

fn sample_matrix(rows: usize, cols: usize) -> DenseMatrix<f64> {
    let mut rng = StdRng::seed_from_u64(3407);
    let data: Vec<f64> = (0..rows * cols)
        .map(|_| rng.random::<f64>() - 0.5)
        .collect();
    DenseMatrix::new(rows, cols, data, true).unwrap()
}

fn assert_matrix_exact(a: &DenseMatrix<f64>, b: &DenseMatrix<f64>) {
    assert_eq!(a.shape(), b.shape(), "shape mismatch");
    let (rows, cols) = a.shape();
    for r in 0..rows {
        for c in 0..cols {
            assert_eq!(
                *a.get((r, c)),
                *b.get((r, c)),
                "value mismatch at ({r},{c}): IPC must round-trip f64 exactly"
            );
        }
    }
}

fn vector_batch(dim: usize, values: Vec<f64>) -> RecordBatch {
    let child = Arc::new(Field::new("item", DataType::Float64, false));
    let list = FixedSizeListArray::new(
        child.clone(),
        dim as i32,
        Arc::new(Float64Array::from(values)),
        None,
    );
    let schema = Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(child, dim as i32),
        false,
    )]);
    RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list)]).unwrap()
}

/// The vector-layout batch for a matrix, hand-built exactly as the crate's
/// dense conversion builds it (schema name included).
fn dense_batch(data: &DenseMatrix<f64>) -> RecordBatch {
    let (rows, cols) = data.shape();
    let values: Vec<f64> = (0..rows)
        .flat_map(|r| (0..cols).map(move |c| *data.get((r, c))))
        .collect();
    vector_batch(cols, values)
}

/// `save_dense_to_ipc` / `load_dense_from_ipc`: exact `DenseMatrix<f64>`
/// round-trip through the Arrow IPC file format, registered in the
/// metadata registry as an `arrow-ipc` artifact.
#[tokio::test(flavor = "multi_thread")]
async fn save_load_dense_ipc_roundtrip() {
    let (_base, storage) = seeded_storage("ipc_dense").await;
    let md_path = storage.metadata_path();
    let data = sample_matrix(5, 7);

    let path = storage
        .save_dense_to_ipc("dense_ipc", &data, &md_path)
        .await
        .expect("save_dense_to_ipc");

    assert!(
        path.to_string_lossy()
            .ends_with("ipc_dense_dense_ipc.arrow"),
        "IPC file artifacts live at {{instance}}_{{key}}.arrow: {path:?}"
    );
    assert!(path.is_file(), "IPC artifact must exist after save");

    let loaded = storage
        .load_dense_from_ipc("dense_ipc")
        .await
        .expect("load_dense_from_ipc");
    assert_matrix_exact(&data, &loaded);

    // registry: filetype stays semantic ("dense"), the format fact names
    // arrow-ipc, and the collection kind is the vector-space kind.
    let md = storage.load_metadata().await.unwrap();
    let info = md.files.get("dense_ipc").expect("registered dense_ipc");
    assert_eq!(info.filetype, "dense");
    assert_eq!(info.storage_format, "arrow-ipc");
    assert_eq!(info.kind, Some(MetaKind::VectorSpace));
    assert_eq!(info.filename, "ipc_dense_dense_ipc.arrow");
}

/// `open_ipc_stream_writer`: three batches flushed incrementally to a
/// `.arrows` stream file reload as one `DenseMatrix` in write order.
#[tokio::test(flavor = "multi_thread")]
async fn ipc_stream_writer_concatenates_batches_in_order() {
    let (_base, storage) = seeded_storage("ipc_stream").await;

    let dim = 4usize;
    let batch_values = |b: usize| -> Vec<f64> {
        (0..4)
            .flat_map(|r| (0..dim).map(move |c| (b * 4 + r) as f64 + c as f64 / 10.0))
            .collect()
    };
    let schema = vector_batch(dim, batch_values(0)).schema();

    let mut writer = storage
        .open_ipc_stream_writer("stream_dense", schema)
        .await
        .expect("open_ipc_stream_writer");
    for b in 0..3 {
        writer
            .write_batch(&vector_batch(dim, batch_values(b)))
            .expect("write_batch");
    }
    let path = writer.finish().expect("finish");

    assert!(
        path.to_string_lossy()
            .ends_with("ipc_stream_stream_dense.arrows"),
        "stream artifacts live at {{instance}}_{{key}}.arrows: {path:?}"
    );
    assert!(path.is_file(), "stream artifact must exist after finish");

    let loaded = storage
        .load_dense_from_ipc("stream_dense")
        .await
        .expect("reload stream as one dense matrix");
    assert_eq!(loaded.shape(), (12, dim), "three batches concatenate");
    for b in 0..3 {
        for r in 0..4 {
            for c in 0..dim {
                assert_eq!(
                    *loaded.get((b * 4 + r, c)),
                    (b * 4 + r) as f64 + c as f64 / 10.0,
                    "batch {b} row {r} must land at global row {} in write order",
                    b * 4 + r
                );
            }
        }
    }
}

/// Graph collections round-trip through the Arrow IPC file format and
/// agree with the Lance `save_graph`/`load_graph` path, CSR included.
#[tokio::test(flavor = "multi_thread")]
async fn graph_ipc_roundtrip_matches_lance_load_graph() {
    let (_base, storage) = seeded_storage("ipc_graph").await;
    let md_path = storage.metadata_path();

    let mut rng = StdRng::seed_from_u64(3407);
    let edges: Vec<GraphEdge> = (0..14)
        .map(|_| {
            GraphEdge::weighted(
                rng.random_range(0..6),
                rng.random_range(0..6),
                rng.random::<f64>() - 0.5,
            )
        })
        .collect();

    let ipc_path = storage
        .save_graph_to_ipc("g_ipc", &edges, &md_path)
        .await
        .expect("save_graph_to_ipc");
    assert!(
        ipc_path
            .to_string_lossy()
            .ends_with("ipc_graph_g_ipc.arrow"),
        "graph IPC artifacts live at {{instance}}_{{key}}.arrow: {ipc_path:?}"
    );

    // the same edges through the canonical Lance path, for comparison
    storage
        .save_graph("g_lance", &edges, &md_path)
        .await
        .expect("save_graph");

    let ipc_graph: StoredGraph = storage
        .load_graph_from_ipc("g_ipc")
        .await
        .expect("load_graph_from_ipc");
    let lance_graph = storage.load_graph("g_lance").await.expect("load_graph");

    assert_eq!(ipc_graph, lance_graph, "IPC and Lance graph paths agree");
    let ipc_csr = ipc_graph.to_csr().expect("CSR from IPC graph");
    let lance_csr = lance_graph.to_csr().expect("CSR from Lance graph");
    assert_eq!(ipc_csr.rows(), lance_csr.rows());
    assert_eq!(ipc_csr.cols(), lance_csr.cols());
    assert_eq!(ipc_csr.nnz(), lance_csr.nnz());
    for (v, (r, c)) in ipc_csr.iter() {
        assert_eq!(
            lance_csr.get(r, c),
            Some(v),
            "CSR value mismatch at ({r},{c})"
        );
    }

    let md = storage.load_metadata().await.unwrap();
    let info = md.files.get("g_ipc").expect("registered g_ipc");
    assert_eq!(info.filetype, "graph");
    assert_eq!(info.storage_format, "arrow-ipc");
    assert_eq!(info.kind, Some(MetaKind::Graph));
}

/// Generations stay independent on the IPC path: generation 0 and
/// generation 1 artifacts resolve separately, generation-scoped.
#[tokio::test(flavor = "multi_thread")]
async fn ipc_generation_scoping_resolves_independently() {
    let base = tmp_dir("ipc_generations").await;
    let logical = LanceStorageGraph::new(base.to_string_lossy().to_string(), "ds".to_string())
        .expect("valid instance name");

    let g0 = logical.scoped_generation(0);
    let g1 = logical.scoped_generation(1);
    GeneMetadata::seed_metadata("ds", 4, 4, &g0)
        .await
        .expect("seed generation 0");
    GeneMetadata::seed_metadata("ds", 4, 4, &g1)
        .await
        .expect("seed generation 1");

    let a = sample_matrix(3, 2);
    let b = sample_matrix(4, 3);
    let p0 = g0
        .save_dense_to_ipc("dense", &a, &g0.metadata_path())
        .await
        .expect("save generation 0");
    let p1 = g1
        .save_dense_to_ipc("dense", &b, &g1.metadata_path())
        .await
        .expect("save generation 1");

    assert!(
        p0.ends_with("ds__g0_dense.arrow"),
        "generation 0 artifact name carries the generation stamp: {p0:?}"
    );
    assert!(
        p1.ends_with("ds__g1_dense.arrow"),
        "generation 1 artifact name carries the generation stamp: {p1:?}"
    );

    assert_matrix_exact(
        &a,
        &g0.load_dense_from_ipc("dense").await.expect("load gen 0"),
    );
    assert_matrix_exact(
        &b,
        &g1.load_dense_from_ipc("dense").await.expect("load gen 1"),
    );
}

/// Golden-fixture conformance: the writer emits the Arrow IPC *file*
/// byte layout — `ARROW1\0\0` magic up front and the spec trailer
/// (footer flatbuffer, i32 footer-length prefix, `ARROW1` tail). The
/// standard arrow-crate reader opens the result, and a file written by
/// the standard crate loads through the backend reader.
#[tokio::test(flavor = "multi_thread")]
async fn ipc_file_byte_layout_conformance() {
    let (_base, storage) = seeded_storage("ipc_bytes").await;
    let md_path = storage.metadata_path();
    let data = sample_matrix(3, 5);

    let path = storage
        .save_dense_to_ipc("layout", &data, &md_path)
        .await
        .expect("save_dense_to_ipc");

    // byte-layout invariants of the Arrow IPC *file* format
    let bytes = std::fs::read(&path).expect("read IPC file");
    assert!(
        bytes.len() > 18,
        "IPC file must carry header, payload and trailer"
    );
    assert_eq!(
        &bytes[0..8],
        b"ARROW1\0\0",
        "file must start with the Arrow IPC file magic"
    );
    assert_eq!(
        &bytes[bytes.len() - 6..],
        b"ARROW1",
        "file must end with the ARROW1 trailer magic"
    );
    let footer_len = i32::from_le_bytes(
        bytes[bytes.len() - 10..bytes.len() - 6]
            .try_into()
            .expect("footer length prefix"),
    );
    assert!(
        footer_len > 0 && (footer_len as usize) + 10 <= bytes.len(),
        "footer length prefix {footer_len} must describe the footer before the trailer"
    );

    // standard-reader direction: the arrow crate opens our artifact
    let file = std::fs::File::open(&path).unwrap();
    let mut reader =
        FileReader::try_new(file, None).expect("standard FileReader opens the artifact");
    assert_eq!(reader.num_batches(), 1);
    let batch = reader.next().unwrap().expect("batch read");
    assert_eq!(batch.num_rows(), 3, "standard reader sees the payload");

    // interop direction: a file written by the plain arrow crate (no
    // genegraph writer involved) loads through the same IPC reader.
    let foreign = path.with_file_name("ipc_bytes_foreign.arrow");
    let file = std::fs::File::create(&foreign).unwrap();
    let mut writer =
        arrow::ipc::writer::FileWriter::try_new(file, dense_batch(&data).schema().as_ref())
            .expect("foreign FileWriter");
    writer.write(&dense_batch(&data)).expect("foreign write");
    writer.finish().expect("foreign finish");

    let loaded = storage
        .load_dense_from_ipc("foreign")
        .await
        .expect("foreign-written file loads through the IPC reader");
    assert_matrix_exact(&data, &loaded);
}

/// `load_dense_from_ipc` on a missing key surfaces a typed error naming
/// both candidate artifacts instead of a bare IO error.
#[tokio::test(flavor = "multi_thread")]
async fn ipc_missing_artifact_is_a_typed_error() {
    let (_base, storage) = seeded_storage("ipc_missing").await;

    let err = storage
        .load_dense_from_ipc("absent")
        .await
        .expect_err("missing artifact must error");
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "expected Invalid for a missing IPC artifact, got {err:?}"
    );
    let msg = err.to_string();
    assert!(msg.contains(".arrow"), "error names the file format: {msg}");
    assert!(
        msg.contains(".arrows"),
        "error names the stream format: {msg}"
    );
}
/// One rule, every format (#142 + the layer rules): the Zarr backend gets
/// the dense and stream IPC paths (dense fits Zarr roots), and keeps its
/// typed rejection for graph collections.
#[tokio::test(flavor = "multi_thread")]
async fn zarr_ipc_dense_roundtrip_graph_rejected_and_stream_writes() {
    let base = tmp_dir("ipc_zarr").await;
    let storage = {
        use crate::zarr_storage::ZarrStorage;
        let s = ZarrStorage::new(&base, "zarr_ipc").expect("valid label");
        GeneMetadata::seed_metadata("zarr_ipc", 4, 4, &s)
            .await
            .expect("seed registry");
        s
    };
    let md_path = storage.metadata_path();

    // dense round trip through the file format, registered as arrow-ipc
    let data = sample_matrix(4, 6);
    let path = storage
        .save_dense_to_ipc("dense_ipc", &data, &md_path)
        .await
        .expect("zarr save_dense_to_ipc");
    assert!(
        path.to_string_lossy().ends_with("dense_ipc.arrow"),
        "Zarr IPC artifacts live at {{root}}/{{key}}.arrow: {path:?}"
    );
    assert_matrix_exact(
        &data,
        &storage
            .load_dense_from_ipc("dense_ipc")
            .await
            .expect("zarr reload"),
    );

    let md = storage.load_metadata().await.unwrap();
    let info = md.files.get("dense_ipc.arrow").expect("registered rel key");
    assert_eq!(info.filetype, "dense");
    assert_eq!(info.storage_format, "arrow-ipc");
    drop(md);

    // stream: three flushes reload concatenated in order
    let dim = 3usize;
    let batch_values = |b: usize| -> Vec<f64> {
        (0..2)
            .flat_map(|r| (0..dim).map(move |c| (b * 2 + r) as f64 + c as f64 / 10.0))
            .collect()
    };
    let schema = vector_batch(dim, batch_values(0)).schema();
    let mut writer = storage
        .open_ipc_stream_writer("stream_dense", schema)
        .await
        .expect("zarr open_ipc_stream_writer");
    for b in 0..3 {
        writer
            .write_batch(&vector_batch(dim, batch_values(b)))
            .expect("write_batch");
    }
    writer.finish().expect("finish");
    let loaded = storage
        .load_dense_from_ipc("stream_dense")
        .await
        .expect("zarr stream reload");
    assert_eq!(loaded.shape(), (6, dim));
    for b in 0..3 {
        for r in 0..2 {
            for c in 0..dim {
                assert_eq!(
                    *loaded.get((b * 2 + r, c)),
                    (b * 2 + r) as f64 + c as f64 / 10.0
                );
            }
        }
    }

    // graph collections keep their typed rejection on Zarr
    let edges = vec![GraphEdge::weighted(0, 1, 0.5)];
    let err = storage
        .save_graph_to_ipc("g", &edges, &md_path)
        .await
        .expect_err("graph IPC stays rejected on Zarr");
    assert!(
        matches!(err, crate::StorageError::UnsupportedFiletype(_)),
        "expected UnsupportedFiletype, got {err:?}"
    );
    let err = storage
        .load_graph_from_ipc("g")
        .await
        .expect_err("graph IPC stays rejected on Zarr");
    assert!(matches!(err, crate::StorageError::UnsupportedFiletype(_)));
}
