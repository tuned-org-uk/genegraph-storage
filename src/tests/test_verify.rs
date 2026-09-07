//! #115: dataset verification inspection primitive.
//!
//! A consumer `verify` cross-checks the stamped dataset schema (read via
//! `lancefmt::read_schema`, no column decode) against the registry
//! descriptor facts. The expectation set follows what each writer stamps:
//! legacy dense/sparse artifacts carry `rows`/`cols` (no `kind`); RFC-#81
//! collections carry `kind` (graphs also carry the graph facts). Random
//! seed 3407.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use smartcore::linalg::basic::arrays::Array2;
use smartcore::linalg::basic::matrix::DenseMatrix;
use sprs::TriMat;

use crate::catalog::{Catalog, LocalRegistry};
use crate::generations;
use crate::graph::GraphEdge;
use crate::lance_storage_graph::LanceStorageGraph;
use crate::lancefmt::verify::{DatasetFacts, verify_dataset_schema};
use crate::metadata::GeneMetadata;
use crate::traits::backend::StorageBackend;
use crate::traits::metadata::Metadata;

use super::tmp_dir;

async fn seeded_storage(name: &str) -> (std::path::PathBuf, LanceStorageGraph) {
    let base = tmp_dir(name).await;
    let storage = LanceStorageGraph::new(base.to_string_lossy().to_string(), name.to_string())
        .expect("valid instance name");
    GeneMetadata::seed_metadata(name, 4, 4, &storage)
        .await
        .expect("seed metadata");
    (base, storage)
}

fn graph_edges() -> Vec<GraphEdge> {
    vec![
        GraphEdge::weighted(0, 1, 0.25),
        GraphEdge::weighted(1, 2, -0.5),
        GraphEdge::weighted(2, 0, 2.0),
    ]
}

/// A consistent scalar (vector-space) collection verifies against its
/// stamped facts: the writer stamps `kind` only, so a facts set asserting
/// `kind` must pass (and must not assert rows/cols).
#[tokio::test(flavor = "multi_thread")]
async fn test_verify_consistent_scalar_collection() {
    let (base, storage) = seeded_storage("verify_scalars").await;

    let mut rng = StdRng::seed_from_u64(3407);
    let values: Vec<f64> = (0..8).map(|_| rng.random_range(-1.0..1.0)).collect();
    let path = base.join("lambdas.lance");
    storage.save_scalars_to_path(&path, &values).await.unwrap();

    let facts = DatasetFacts {
        kind: Some("vector-space".to_string()),
        ..DatasetFacts::default()
    };
    storage
        .verify_collection_from_path(&path, &facts)
        .await
        .expect("consistent scalar collection verifies");
}

/// A kind mismatch is a typed `Invalid` naming the `kind` fact.
#[tokio::test(flavor = "multi_thread")]
async fn test_verify_kind_mismatch_names_the_fact() {
    let (base, storage) = seeded_storage("verify_kind").await;

    let values = vec![0.5, -0.25, 1.0];
    let path = base.join("lambdas.lance");
    storage.save_scalars_to_path(&path, &values).await.unwrap();

    let facts = DatasetFacts {
        kind: Some("graph".to_string()),
        ..DatasetFacts::default()
    };
    let err = storage
        .verify_collection_from_path(&path, &facts)
        .await
        .unwrap_err();
    match err {
        crate::StorageError::Invalid(msg) => {
            assert!(msg.contains("kind"), "error must name the fact: {msg}");
            assert!(msg.contains("graph"), "error must name both values: {msg}");
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

/// A graph collection verifies through both registry bridges
/// (`from_file_info` and `from_descriptor`) with the stamped facts
/// (`kind`, `num_nodes`, `weight_type`, `node_id_width`).
#[tokio::test(flavor = "multi_thread")]
async fn test_verify_graph_collection_from_registry_bridges() {
    let (base, storage) = seeded_storage("verify_graph").await;
    let md_path = storage.metadata_path();

    let edges = graph_edges();
    storage
        .save_graph("adj", &edges, &md_path)
        .await
        .expect("save_graph");

    // Bridge 1: the registry FileInfo (kind is explicit on RFC-#81 saves).
    let md = storage.load_metadata().await.unwrap();
    let info = md.files.get("adj").expect("graph registry entry");
    let facts = DatasetFacts::from_file_info(info);
    assert_eq!(facts.kind.as_deref(), Some("graph"));
    assert_eq!(facts.num_nodes, Some(3));
    assert_eq!(facts.weight_type.as_deref(), Some("f64"));
    assert_eq!(facts.node_id_width.as_deref(), Some("u32"));
    storage
        .verify_collection("adj", &facts)
        .await
        .expect("graph verifies against FileInfo facts");

    // Bridge 2: the catalog descriptor.
    let registry = LocalRegistry::new(md, base.clone());
    let descriptor = registry.describe_table("adj").expect("descriptor");
    let facts = DatasetFacts::from_descriptor(&descriptor);
    assert_eq!(facts.kind.as_deref(), Some("graph"));
    assert_eq!(facts.num_nodes, Some(3));
    storage
        .verify_collection("adj", &facts)
        .await
        .expect("graph verifies against descriptor facts");

    // The lancefmt-level helper agrees (no backend involved).
    let dir = generations::artifact_file_path(base.as_path(), storage.get_name().as_str(), "adj");
    verify_dataset_schema(&dir, &facts).expect("direct lancefmt verify");
}

/// A `num_nodes` mismatch is a typed `DimensionMismatch` naming the fact.
#[tokio::test(flavor = "multi_thread")]
async fn test_verify_num_nodes_mismatch_is_dimension_mismatch() {
    let (_base, storage) = seeded_storage("verify_num_nodes").await;
    let md_path = storage.metadata_path();

    storage
        .save_graph("adj", &graph_edges(), &md_path)
        .await
        .expect("save_graph");

    let md = storage.load_metadata().await.unwrap();
    let info = md.files.get("adj").unwrap();
    let mut facts = DatasetFacts::from_file_info(info);
    facts.num_nodes = Some(64);

    let err = storage.verify_collection("adj", &facts).await.unwrap_err();
    match err {
        crate::StorageError::DimensionMismatch { expected, found } => {
            assert!(
                expected.contains("num_nodes"),
                "expected names fact: {expected}"
            );
            assert!(found.contains("num_nodes"), "found names fact: {found}");
            assert!(
                expected.contains("64"),
                "expected carries the value: {expected}"
            );
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

/// A weight-width mismatch is a typed `Invalid` naming the fact.
#[tokio::test(flavor = "multi_thread")]
async fn test_verify_weight_type_mismatch_is_invalid() {
    let (_base, storage) = seeded_storage("verify_weight").await;
    let md_path = storage.metadata_path();

    storage
        .save_graph("adj", &graph_edges(), &md_path)
        .await
        .expect("save_graph");

    let md = storage.load_metadata().await.unwrap();
    let info = md.files.get("adj").unwrap();
    let mut facts = DatasetFacts::from_file_info(info);
    facts.weight_type = Some("f32".to_string());

    let err = storage.verify_collection("adj", &facts).await.unwrap_err();
    match err {
        crate::StorageError::Invalid(msg) => {
            assert!(
                msg.contains("weight_type"),
                "error must name the fact: {msg}"
            );
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

/// Shape stamps live on sparse artifacts only (`to_sparse_record_batch`):
/// the bridge asserts `rows`/`cols`/`nnz` there, a wrong shape is a typed
/// `DimensionMismatch`, and dense datasets stamp nothing so their facts
/// set is empty (nothing to assert from the registry).
#[tokio::test(flavor = "multi_thread")]
async fn test_verify_sparse_shape_facts() {
    let (_base, storage) = seeded_storage("verify_sparse").await;
    let md_path = storage.metadata_path();

    let mut trimat = TriMat::new((4, 4));
    trimat.add_triplet(0, 1, 0.5);
    trimat.add_triplet(2, 3, -1.5);
    let csr = trimat.to_csr();
    storage
        .save_sparse("laplacian", &csr, &md_path)
        .await
        .expect("save_sparse");

    let md = storage.load_metadata().await.unwrap();
    let info = md.files.get("laplacian").expect("sparse registry entry");
    let facts = DatasetFacts::from_file_info(info);
    assert_eq!(facts.kind, None, "legacy sparse carries no kind stamp");
    assert_eq!(facts.rows, Some(4));
    assert_eq!(facts.cols, Some(4));
    assert_eq!(facts.nnz, Some(2));
    storage
        .verify_collection("laplacian", &facts)
        .await
        .expect("consistent sparse dataset verifies");

    let mut wrong = facts.clone();
    wrong.rows = Some(9);
    let err = storage
        .verify_collection("laplacian", &wrong)
        .await
        .unwrap_err();
    match err {
        crate::StorageError::DimensionMismatch { expected, found } => {
            assert!(
                expected.contains("rows"),
                "expected names the fact: {expected}"
            );
            assert!(found.contains("rows"), "found names the fact: {found}");
        }
        other => panic!("expected DimensionMismatch, got {other:?}"),
    }
}

/// Dense datasets stamp nothing (the vector layout carries no schema
/// metadata): the bridge derives an empty expectation set, so a dense
/// dataset verifies trivially and no registry fact over-asserts.
#[tokio::test(flavor = "multi_thread")]
async fn test_verify_dense_dataset_stamps_nothing() {
    let (_base, storage) = seeded_storage("verify_dense").await;
    let md_path = storage.metadata_path();

    let data = DenseMatrix::<f64>::from_iterator((0..6u32).map(|i| f64::from(i) + 0.5), 3, 2, 0);
    storage
        .save_dense("rawinput", &data, &md_path)
        .await
        .expect("save_dense");

    let md = storage.load_metadata().await.unwrap();
    let info = md.files.get("rawinput").expect("dense registry entry");
    let facts = DatasetFacts::from_file_info(info);
    assert_eq!(facts, DatasetFacts::default(), "dense stamps nothing");

    storage
        .verify_collection("rawinput", &facts)
        .await
        .expect("consistent dense dataset verifies");
}

/// A facts set asserting a stamp the writer never writes fails typed and
/// names the missing fact.
#[tokio::test(flavor = "multi_thread")]
async fn test_verify_missing_stamp_fails_typed() {
    let (base, storage) = seeded_storage("verify_missing").await;

    let path = base.join("lambdas.lance");
    storage
        .save_scalars_to_path(&path, &[0.1, 0.2])
        .await
        .unwrap();

    let facts = DatasetFacts {
        num_nodes: Some(7),
        ..DatasetFacts::default()
    };
    let err = storage
        .verify_collection_from_path(&path, &facts)
        .await
        .unwrap_err();
    match err {
        crate::StorageError::Invalid(msg) => {
            assert!(msg.contains("num_nodes"), "error must name the fact: {msg}");
        }
        other => panic!("expected Invalid, got {other:?}"),
    }
}

/// The schema-side extraction returns exactly what the dataset stamps
/// (inspection only — no decode), so the two sides of the cross-check are
/// built from the same grammar.
#[tokio::test(flavor = "multi_thread")]
async fn test_from_schema_extracts_stamped_facts() {
    let (base, storage) = seeded_storage("verify_schema").await;
    let md_path = storage.metadata_path();

    storage
        .save_graph("adj", &graph_edges(), &md_path)
        .await
        .expect("save_graph");

    let dir = generations::artifact_file_path(base.as_path(), storage.get_name().as_str(), "adj");
    let schema = crate::lancefmt::read_schema(&dir).expect("read_schema");
    let facts = DatasetFacts::from_schema(&schema);

    assert_eq!(facts.kind.as_deref(), Some("graph"));
    assert_eq!(facts.num_nodes, Some(3));
    assert_eq!(facts.weight_type.as_deref(), Some("f64"));
    assert_eq!(facts.node_id_width.as_deref(), Some("u32"));
    assert_eq!(facts.weighted.as_deref(), Some("true"));
    assert_eq!(facts.rows, None, "graph datasets carry no rows stamp");
}

/// A vectors collection stamps `kind` only: the descriptor bridge must not
/// over-assert `rows`/`cols` (the registry shape is a computed fact, not a
/// dataset stamp), so a consistent vectors collection verifies.
#[tokio::test(flavor = "multi_thread")]
async fn test_descriptor_bridge_does_not_overassert_vector_facts() {
    use std::sync::Arc;

    use arrow::array::{FixedSizeListArray, Float64Array};
    use arrow::datatypes::{DataType, Field, Schema};
    use arrow::record_batch::RecordBatch;

    let (base, storage) = seeded_storage("verify_vectors").await;
    let md_path = storage.metadata_path();

    let child = Arc::new(Field::new("item", DataType::Float64, false));
    let list = FixedSizeListArray::new(
        child.clone(),
        2,
        Arc::new(Float64Array::from(vec![0.5, 1.5, -2.5, 3.0])),
        None,
    );
    let schema = Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(child, 2),
        false,
    )]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list) as _]).unwrap();
    storage
        .save_vectors("vecs", &batch, &md_path)
        .await
        .expect("save_vectors");

    let md = storage.load_metadata().await.unwrap();
    let registry = LocalRegistry::new(md, base.clone());
    let descriptor = registry.describe_table("vecs").expect("descriptor");
    let facts = DatasetFacts::from_descriptor(&descriptor);
    assert_eq!(facts.kind.as_deref(), Some("vector-space"));
    assert_eq!(facts.rows, None, "vectors datasets stamp no rows");
    storage
        .verify_collection("vecs", &facts)
        .await
        .expect("consistent vectors collection verifies");
}
