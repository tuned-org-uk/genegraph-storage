//! Named collections, graphs and linkage (RFC #81-P1..P4).

use std::collections::BTreeMap;
use std::path::PathBuf;
use std::sync::Arc;

use arrow::array::{FixedSizeListArray, Float32Array, Float64Array, UInt32Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use smartcore::linalg::basic::arrays::Array2;
use sprs::{CsMat, TriMat};

use crate::catalog::{Catalog, CollectionKind, LocalRegistry, TableDescriptor};
use crate::graph::{GraphEdge, GraphWriteOptions, NodeIdWidth, StoredGraph, WeightType};
use crate::lance_storage_graph::LanceStorageGraph;
use crate::metadata::GeneMetadata;
use crate::tests::tmp_dir;
use crate::traits::backend::StorageBackend;
use crate::traits::metadata::Metadata;

async fn seeded_storage(name: &str) -> (PathBuf, LanceStorageGraph) {
    let base = tmp_dir(name).await;
    let storage = LanceStorageGraph::new(base.to_string_lossy().to_string(), name.to_string());
    GeneMetadata::seed_metadata(name, 4, 4, &storage)
        .await
        .expect("seed metadata");
    (base, storage)
}

fn f64_vector_batch(ids: &[u32], vectors: &[Vec<f64>]) -> RecordBatch {
    let dim = vectors[0].len() as i32;
    let flat: Vec<f64> = vectors.iter().flatten().copied().collect();
    let child = Arc::new(Field::new("item", DataType::Float64, false));
    let list =
        FixedSizeListArray::new(child.clone(), dim, Arc::new(Float64Array::from(flat)), None);
    let schema = Schema::new(vec![
        Field::new("item_id", DataType::UInt32, false),
        Field::new("vector", DataType::FixedSizeList(child, dim), false),
    ]);
    RecordBatch::try_new(
        Arc::new(schema),
        vec![
            Arc::new(UInt32Array::from(ids.to_vec())) as _,
            Arc::new(list) as _,
        ],
    )
    .unwrap()
}

// ---------------------------------------------------------------------------
// P2: save_vectors / load_vectors
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn vectors_f64_roundtrip_with_id_column() {
    let (_base, storage) = seeded_storage("vectors_f64").await;
    let md_path = storage.metadata_path();

    let ids = vec![10u32, 20, 30];
    let vectors = vec![
        vec![0.5, -1.25, 3.0, 7.0],
        vec![0.0; 4],
        vec![-2.5, 1.0, 0.125, 9.5],
    ];
    let batch = f64_vector_batch(&ids, &vectors);
    storage
        .save_vectors("embeddings", &batch, &md_path)
        .await
        .expect("save_vectors");

    let loaded = storage
        .load_vectors("embeddings")
        .await
        .expect("load_vectors");
    assert_eq!(loaded.num_rows(), 3);
    let ids_out = loaded
        .column(0)
        .as_any()
        .downcast_ref::<UInt32Array>()
        .unwrap();
    for (i, e) in ids.iter().enumerate() {
        assert_eq!(ids_out.value(i), *e);
    }
    let list = loaded
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap();
    let values = list
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    for (i, e) in vectors.iter().flatten().enumerate() {
        assert_eq!(values.value(i), *e, "vector value mismatch at {i}");
    }

    // dataset-level kind stamped into the schema metadata (RFC #81-P1)
    assert_eq!(
        loaded.schema().metadata().get("kind").map(String::as_str),
        Some("vector-space")
    );

    // registry-level kind + properties
    let md = storage.load_metadata().await.unwrap();
    let info = md.files.get("embeddings").unwrap();
    assert_eq!(info.kind, Some(CollectionKind::VectorSpace));
    assert_eq!(info.properties.get("graph"), None);
}

#[tokio::test(flavor = "multi_thread")]
async fn vectors_f32_roundtrip_bit_exact() {
    let (_base, storage) = seeded_storage("vectors_f32").await;
    let md_path = storage.metadata_path();

    let dim = 2i32;
    let values: Vec<f32> = vec![
        0.0,
        -0.0,
        1.5,
        -2.25,
        f32::MIN_POSITIVE / 8.0,
        f32::INFINITY,
        0.123_456_79,
        -0.999_999_9,
    ];
    let child = Arc::new(Field::new("item", DataType::Float32, false));
    let list = FixedSizeListArray::new(
        child.clone(),
        dim,
        Arc::new(Float32Array::from(values.clone())),
        None,
    );
    let schema = Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(child, dim),
        false,
    )]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list) as _]).unwrap();

    storage
        .save_vectors("f32_maps", &batch, &md_path)
        .await
        .expect("save_vectors");
    let loaded = storage
        .load_vectors("f32_maps")
        .await
        .expect("load_vectors");
    let out = loaded
        .column(0)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap()
        .values()
        .as_any()
        .downcast_ref::<Float32Array>()
        .unwrap();
    assert_eq!(out.len(), values.len());
    for (i, e) in values.iter().enumerate() {
        assert_eq!(
            out.value(i).to_bits(),
            e.to_bits(),
            "f32 bit-exact mismatch at {i}"
        );
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn vectors_rejects_invalid_schemas() {
    let (_base, storage) = seeded_storage("vectors_invalid").await;
    let md_path = storage.metadata_path();

    // no FixedSizeList column
    let schema = Schema::new(vec![Field::new("x", DataType::Float64, false)]);
    let batch = RecordBatch::try_new(
        Arc::new(schema),
        vec![Arc::new(Float64Array::from(vec![1.0])) as _],
    )
    .unwrap();
    let err = storage
        .save_vectors("bad", &batch, &md_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "got {err:?}"
    );

    // nullable column
    let child = Arc::new(Field::new("item", DataType::Float64, false));
    let schema = Schema::new(vec![Field::new(
        "vector",
        DataType::FixedSizeList(child, 1),
        true,
    )]);
    let list = FixedSizeListArray::new(
        Arc::new(Field::new("item", DataType::Float64, false)),
        1,
        Arc::new(Float64Array::from(vec![1.0])),
        None,
    );
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list) as _]).unwrap();
    let err = storage
        .save_vectors("bad", &batch, &md_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "got {err:?}"
    );

    // reserved user property
    let batch = f64_vector_batch(&[1], &[vec![1.0]]);
    let mut props = BTreeMap::new();
    props.insert("kind".to_string(), "graph".to_string());
    let err = storage
        .save_vectors_with("bad", &batch, &props, &md_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "got {err:?}"
    );
}

/// P1 compat shim: legacy fixed-key `save_dense` keeps working alongside the
/// named-collection API, and both land in the registry with correct kinds.
#[tokio::test(flavor = "multi_thread")]
async fn legacy_save_dense_works_alongside_save_vectors() {
    let (_base, storage) = seeded_storage("legacy_compat").await;
    let md_path = storage.metadata_path();

    let dense = vec![vec![1.0, 2.0], vec![3.0, 4.0]];
    let matrix = smartcore::linalg::basic::matrix::DenseMatrix::<f64>::from_iterator(
        dense.iter().flatten().copied(),
        2,
        2,
        0,
    );
    storage
        .save_dense("rawinput", &matrix, &md_path)
        .await
        .expect("legacy save_dense");

    let batch = f64_vector_batch(&[0, 1], &dense);
    storage
        .save_vectors("modern", &batch, &md_path)
        .await
        .expect("save_vectors");

    let md = storage.load_metadata().await.unwrap();
    assert_eq!(md.files["rawinput"].kind, Some(CollectionKind::VectorSpace));
    assert_eq!(md.files["modern"].kind, Some(CollectionKind::VectorSpace));

    let registry = LocalRegistry::new(md, storage.base_path());
    let desc = registry.describe_table("rawinput").unwrap();
    assert_eq!(desc.kind, CollectionKind::VectorSpace);
    assert_eq!(
        desc.properties.get("kind").map(String::as_str),
        Some("vector-space")
    );
}

// ---------------------------------------------------------------------------
// P3: save_graph / load_graph / to_csr
// ---------------------------------------------------------------------------

/// Deterministic pseudo-random graph generator (no proptest dependency).
/// Weights are full-precision f64: the default (f64) weight width stores
/// them exactly, like sparse-matrix values (#106).
fn generated_graph(n_nodes: u64, n_edges: usize, seed: u64, weighted: bool) -> Vec<GraphEdge> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n_edges)
        .map(|_| {
            let src = rng.random_range(0..n_nodes);
            let dst = rng.random_range(0..n_nodes);
            if weighted {
                GraphEdge::weighted(src, dst, rng.random::<f64>() - 0.5)
            } else {
                GraphEdge::unweighted(src, dst)
            }
        })
        .collect()
}

fn reference_csr(edges: &[GraphEdge], n: u64) -> CsMat<f64> {
    let mut trimat = TriMat::new((n as usize, n as usize));
    for e in edges {
        trimat.add_triplet(e.src as usize, e.dst as usize, e.weight.unwrap_or(1.0));
    }
    trimat.to_csr()
}

fn assert_graph_round_trip(graph: &StoredGraph, edges: &[GraphEdge], n: u64, weighted: bool) {
    assert_eq!(graph.edges.len(), edges.len(), "edge count");
    assert_eq!(graph.num_nodes, n, "node count");
    assert_eq!(graph.weighted, weighted);
    for (got, want) in graph.edges.iter().zip(edges.iter()) {
        assert_eq!(got.src, want.src);
        assert_eq!(got.dst, want.dst);
        match (got.weight, want.weight) {
            (Some(a), Some(b)) => assert_eq!(a.to_bits(), b.to_bits(), "weight bits"),
            (None, None) => {}
            other => panic!("weight presence mismatch: {other:?}"),
        }
    }
    // CSR conversion matches a locally built reference
    let csr = graph.to_csr().unwrap();
    let reference = reference_csr(edges, n);
    assert_eq!(csr.rows(), reference.rows());
    assert_eq!(csr.nnz(), reference.nnz());
    for (v, (r, c)) in csr.iter() {
        let expected = reference.get(r, c).copied().unwrap_or(0.0);
        assert_eq!(*v, expected, "csr value at ({r},{c})");
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn graph_weighted_u32_roundtrip_generated() {
    let (_base, storage) = seeded_storage("graph_u32").await;
    let md_path = storage.metadata_path();

    // sizes cross the u32/f32 chunk boundaries (512/1024 values per chunk)
    for (n_nodes, n_edges, seed) in [(7u64, 1usize, 1), (50, 513, 2), (97, 1500, 3)] {
        let edges = generated_graph(n_nodes, n_edges, seed, true);
        storage
            .save_graph(&format!("g_{seed}"), &edges, &md_path)
            .await
            .expect("save_graph");
        let graph = storage
            .load_graph(&format!("g_{seed}"))
            .await
            .expect("load_graph");
        assert_eq!(graph.node_id_width, NodeIdWidth::U32);
        assert_graph_round_trip(&graph, &edges, n_nodes, true);
    }
}

#[tokio::test(flavor = "multi_thread")]
async fn graph_topology_only_u64_roundtrip() {
    let (_base, storage) = seeded_storage("graph_u64").await;
    let md_path = storage.metadata_path();

    // ids far above u32::MAX force the u64 schema; num_nodes keeps an
    // isolated vertex (id u32::MAX + 2 exists, id u32::MAX + 3 does not)
    let n = u32::MAX as u64 + 4;
    let edges = vec![
        GraphEdge::unweighted(0, n - 2),
        GraphEdge::unweighted(n - 2, n - 1),
        GraphEdge::unweighted(n - 1, 0),
    ];
    let options = GraphWriteOptions::with_width(NodeIdWidth::U64);
    storage
        .save_graph_with("topology", &edges, &options, &md_path)
        .await
        .expect("save_graph_with u64");

    let graph = storage.load_graph("topology").await.expect("load_graph");
    assert_eq!(graph.node_id_width, NodeIdWidth::U64);
    assert!(!graph.weighted);
    // Edge-list exactness only: CSR conversion allocates an O(num_nodes)
    // indptr (~34 GB here) and is exercised separately on a small graph.
    assert_eq!(graph.edges, edges);
    assert_eq!(graph.num_nodes, n);

    // unweighted CSR convention (weight 1.0) on a small-node u64 graph
    let small_edges = vec![GraphEdge::unweighted(0, 2), GraphEdge::unweighted(2, 1)];
    storage
        .save_graph_with("topology_small", &small_edges, &options, &md_path)
        .await
        .expect("save_graph_with u64 small");
    let small = storage
        .load_graph("topology_small")
        .await
        .expect("load_graph u64 small");
    assert_graph_round_trip(&small, &small_edges, 3, false);
    let csr = small.to_csr().unwrap();
    assert_eq!(
        csr.get(0, 2).copied(),
        Some(1.0),
        "topology-only edges get weight 1.0"
    );
}

#[tokio::test(flavor = "multi_thread")]
async fn graph_rejects_invalid_inputs() {
    let (_base, storage) = seeded_storage("graph_invalid").await;
    let md_path = storage.metadata_path();

    // empty edge list
    let err = storage
        .save_graph("empty", &[], &md_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "got {err:?}"
    );

    // mixed weighted/topology edges
    let mixed = vec![GraphEdge::weighted(0, 1, 0.5), GraphEdge::unweighted(1, 2)];
    let err = storage
        .save_graph("mixed", &mixed, &md_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "got {err:?}"
    );

    // u32 schema with a too-large id: Overflow, never truncation (#51)
    let big = vec![GraphEdge::weighted(0, u32::MAX as u64 + 1, 0.5)];
    let err = storage.save_graph("big", &big, &md_path).await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Overflow(_)),
        "got {err:?}"
    );

    // u64 schema accepts it
    storage
        .save_graph_with(
            "big",
            &big,
            &GraphWriteOptions::with_width(NodeIdWidth::U64),
            &md_path,
        )
        .await
        .expect("u64 schema accepts ids above u32::MAX");

    // num_nodes below max id + 1
    let edges = vec![GraphEdge::unweighted(0, 5)];
    let options = GraphWriteOptions {
        num_nodes: Some(4),
        ..Default::default()
    };
    let err = storage
        .save_graph_with("small", &edges, &options, &md_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "got {err:?}"
    );

    // explicit num_nodes keeps isolated vertices
    let options = GraphWriteOptions {
        num_nodes: Some(10),
        ..Default::default()
    };
    storage
        .save_graph_with("sparse_graph", &edges, &options, &md_path)
        .await
        .expect("num_nodes override");
    let graph = storage.load_graph("sparse_graph").await.unwrap();
    assert_eq!(graph.num_nodes, 10, "isolated vertices preserved");
    let csr = graph.to_csr().unwrap();
    assert_eq!(csr.rows(), 10);
}

/// P3 + P1: the graph collection is registered with kind `graph` and its
/// layout facts ride the registry properties.
#[tokio::test(flavor = "multi_thread")]
async fn graph_registry_kind_and_properties() {
    let (_base, storage) = seeded_storage("graph_registry").await;
    let md_path = storage.metadata_path();

    let edges = generated_graph(5, 9, 7, true);
    storage
        .save_graph("laplacian_v2", &edges, &md_path)
        .await
        .unwrap();

    let md = storage.load_metadata().await.unwrap();
    let info = md.files.get("laplacian_v2").unwrap();
    assert_eq!(info.kind, Some(CollectionKind::Graph));
    assert_eq!(
        info.properties.get("node_id_width").map(String::as_str),
        Some("u32")
    );
    assert_eq!(
        info.properties.get("weighted").map(String::as_str),
        Some("true")
    );
    assert_eq!(
        info.properties.get("num_nodes").map(String::as_str),
        Some("5")
    );

    // dataset-level schema metadata carries the same facts
    let batch = crate::lancefmt::scan_all(&storage.file_path("laplacian_v2")).unwrap();
    let schema = batch.schema();
    let meta = schema.metadata();
    assert_eq!(meta.get("kind").map(String::as_str), Some("graph"));
    assert_eq!(meta.get("weighted").map(String::as_str), Some("true"));
}

// ---------------------------------------------------------------------------
// P4: vector-space <-> graph linkage
// ---------------------------------------------------------------------------

#[tokio::test(flavor = "multi_thread")]
async fn vector_space_links_graph_end_to_end() {
    let (base, storage) = seeded_storage("linkage_e2e").await;
    let md_path = storage.metadata_path();

    // the linked graph
    let edges = generated_graph(4, 6, 11, true);
    storage
        .save_graph("space_graph", &edges, &md_path)
        .await
        .expect("save_graph");

    // the vector space referencing it by name (properties.graph)
    let vectors: Vec<Vec<f64>> = (0..4)
        .map(|row| (0..4).map(|c| 0.1 * (row * 4 + c) as f64).collect())
        .collect();
    let batch = f64_vector_batch(&[0, 1, 2, 3], &vectors);
    let mut props = BTreeMap::new();
    props.insert("graph".to_string(), "space_graph".to_string());
    props.insert("tier".to_string(), "hot".to_string());
    storage
        .save_vectors_with("space_vectors", &batch, &props, &md_path)
        .await
        .expect("save_vectors_with");

    // exact-value round trip through load_vectors
    let loaded = storage
        .load_vectors("space_vectors")
        .await
        .expect("load_vectors");
    assert_eq!(loaded.num_rows(), 4);
    assert_eq!(
        loaded.schema().metadata().get("graph").map(String::as_str),
        Some("space_graph"),
        "user properties survive the round trip as dataset metadata"
    );

    // catalog-level helper resolves vectors + linked graph descriptors
    let md = storage.load_metadata().await.unwrap();
    let registry = LocalRegistry::new(md, base);
    let vs = registry
        .describe_vector_space("space_vectors")
        .expect("describe_vector_space");
    assert_eq!(vs.vectors.kind, CollectionKind::VectorSpace);
    let graph = vs.graph.expect("linked graph");
    assert_eq!(graph.name, "space_graph");
    assert_eq!(graph.kind, CollectionKind::Graph);
    assert_eq!(graph.properties.get("nnz").map(String::as_str), Some("6"));
}

// ---------------------------------------------------------------------------
// Commit-actor serialization (duva base concurrency model)
// ---------------------------------------------------------------------------

/// Concurrent `save_*` calls must not lose each other's registry entries:
/// every metadata read-modify-write cycle is serialized through the
/// instance's commit actor.
#[tokio::test(flavor = "multi_thread")]
async fn concurrent_saves_do_not_lose_registry_entries() {
    let (_base, storage) = seeded_storage("commit_actor").await;
    let md_path = storage.metadata_path();

    let mut handles = Vec::new();
    for k in 0..12u32 {
        let storage = storage.clone();
        let md_path = md_path.clone();
        handles.push(tokio::spawn(async move {
            let vector = vec![k as f64, (k * 2) as f64, (k * 3) as f64];
            storage
                .save_vector(&format!("vec_{k}"), &vector, &md_path)
                .await
                .expect("save_vector");
        }));
    }
    for h in handles {
        h.await.expect("join");
    }

    let md = storage.load_metadata().await.unwrap();
    for k in 0..12u32 {
        assert!(
            md.files.contains_key(&format!("vec_{k}")),
            "registry entry vec_{k} was lost to a concurrent writer"
        );
    }

    // RYOW: every committed artifact is loadable with exact values
    for k in 0..12u32 {
        let loaded = storage.load_vector(&format!("vec_{k}")).await.unwrap();
        assert_eq!(loaded, vec![k as f64, (k * 2) as f64, (k * 3) as f64]);
    }
}

/// Review PR #96 finding 1: registry-reserved user properties are rejected
/// on the direct write paths, same rule as `Catalog::register_table` — a
/// caller-provided `rows`/`nnz`/... must not shadow computed facts.
#[tokio::test(flavor = "multi_thread")]
async fn write_paths_reject_registry_reserved_properties() {
    let (_base, storage) = seeded_storage("reserved_props").await;
    let md_path = storage.metadata_path();

    let batch = f64_vector_batch(&[0, 1], &[vec![1.0, 2.0], vec![3.0, 4.0]]);

    for reserved in [
        "rows",
        "cols",
        "nnz",
        "filetype",
        "storage_format",
        "size_bytes",
    ] {
        let mut props = BTreeMap::new();
        props.insert(reserved.to_string(), "999".to_string());
        let err = storage
            .save_vectors_with("bad_vecs", &batch, &props, &md_path)
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::StorageError::Invalid(_)),
            "expected Invalid for '{reserved}', got {err:?}"
        );

        let edges = vec![GraphEdge::weighted(0, 1, 0.5)];
        let options = GraphWriteOptions {
            properties: props,
            ..Default::default()
        };
        let err = storage
            .save_graph_with("bad_graph", &edges, &options, &md_path)
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::StorageError::Invalid(_)),
            "expected Invalid for graph '{reserved}', got {err:?}"
        );
    }

    // nothing was written for the rejected collections
    let md = storage.load_metadata().await.unwrap();
    assert!(!md.files.contains_key("bad_vecs"));
    assert!(!md.files.contains_key("bad_graph"));
}

// ---------------------------------------------------------------------------
// Registry-free collection writes (#106)
// ---------------------------------------------------------------------------

/// #106 review: the registry publish is the single commit point — a live
/// registry entry must never exist without its durable artifact. When the
/// artifact write itself fails, no registry entry may remain and the
/// registry file must be untouched. (The registry-first ordering shipped
/// in 0.53.0 violated this: discovery treated the collection as live
/// while the dataset was never written.)
#[tokio::test(flavor = "multi_thread")]
async fn failed_artifact_write_leaves_no_live_registry_entry() {
    let (_base, storage) = seeded_storage("artifact_write_fail").await;
    let md_path = storage.metadata_path();
    let before = tokio::fs::read(&md_path).await.unwrap();

    // force the artifact write to fail: a regular file occupies the
    // dataset directory location, so the lancefmt writer cannot create it
    let collide = storage.file_path("vecs");
    tokio::fs::write(&collide, b"not a dataset").await.unwrap();

    let batch = f64_vector_batch(&[0, 1], &[vec![0.5, 1.5], vec![-2.5, 3.0]]);
    let err = storage
        .save_vectors_with("vecs", &batch, &BTreeMap::new(), &md_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Io(_)),
        "expected the forced artifact-write failure, got {err:?}"
    );
    assert_eq!(
        tokio::fs::read(&md_path).await.unwrap(),
        before,
        "the registry file must be untouched when the artifact write fails"
    );
    let md = storage.load_metadata().await.unwrap();
    assert!(
        !md.files.contains_key("vecs"),
        "a live registry entry must not point at a missing artifact"
    );

    // same contract for graph collections
    let collide = storage.file_path("adj");
    tokio::fs::write(&collide, b"not a dataset").await.unwrap();
    let edges = vec![GraphEdge::weighted(0, 1, 0.5)];
    let err = storage
        .save_graph("adj", &edges, &md_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Io(_)),
        "expected the forced artifact-write failure, got {err:?}"
    );
    let md = storage.load_metadata().await.unwrap();
    assert!(
        !md.files.contains_key("adj"),
        "a live registry entry must not point at a missing artifact"
    );
}

/// #106 review: when the registry publish fails (here: the metadata path
/// holds the consumer's own document, so the read-modify-write cannot
/// parse it), the artifact was already written and remains as
/// *unreferenced* residue — recoverable garbage a later sweep reclaims.
/// That is the acceptable side of the failure split: nothing is logically
/// committed (the registry document is byte-identical), so discovery
/// never treats the residue as a live collection.
#[tokio::test(flavor = "multi_thread")]
async fn failed_registry_publish_leaves_only_unreferenced_residue() {
    let base = tmp_dir("orphan_artifact").await;
    let storage =
        LanceStorageGraph::new(base.to_string_lossy().to_string(), "app_owned".to_string());
    let md_path = storage.metadata_path();

    // the consumer's own metadata document: any GeneMetadata read fails
    // (the registry/app-metadata split, #106)
    let app_metadata = r#"{"commit_pointer":"app__g7"}"#;
    tokio::fs::write(&md_path, app_metadata).await.unwrap();

    let batch = f64_vector_batch(&[0, 1], &[vec![0.5, 1.5], vec![-2.5, 3.0]]);
    let err = storage
        .save_vectors_with("vecs", &batch, &BTreeMap::new(), &md_path)
        .await
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::Serde(_)), "got {err:?}");

    // nothing is logically committed: the consumer's document is untouched
    assert_eq!(
        tokio::fs::read_to_string(&md_path).await.unwrap(),
        app_metadata
    );

    // the artifact exists as unreferenced residue: crash recovery is a
    // later sweep/gc, never a live entry pointing at missing data
    assert!(storage.file_path("vecs").exists());
}

/// The #106 acceptance: a consumer whose metadata path holds its own
/// metadata document (an ArrowSpaceMetadata-style commit pointer, not a
/// GeneMetadata registry) saves and loads collections with no GeneMetadata
/// read or write occurring. Any GeneMetadata read of the foreign document
/// would fail, so a successful cycle proves no read; a byte-identical file
/// afterwards proves no write.
#[tokio::test(flavor = "multi_thread")]
async fn registry_free_collection_writes_never_touch_gene_metadata() {
    let base = tmp_dir("registry_free").await;
    let storage =
        LanceStorageGraph::new(base.to_string_lossy().to_string(), "app_owned".to_string());

    // the consumer's own single commit pointer at the instance metadata path
    let app_metadata = r#"{"commit_pointer":"app__g7","format":"arrowspace"}"#;
    tokio::fs::write(storage.metadata_path(), app_metadata)
        .await
        .unwrap();

    // registry-free writes still stamp dataset-level collection metadata
    // and leave registry ownership with the consumer
    let mut props = BTreeMap::new();
    props.insert("graph".to_string(), "adj".to_string());
    let batch = f64_vector_batch(&[0, 1], &[vec![0.5, 1.5], vec![-2.5, 3.0]]);
    storage
        .save_vectors_to_path(&storage.file_path("vecs"), &batch, &props)
        .await
        .expect("save_vectors_to_path");

    let edges = vec![
        GraphEdge::weighted(0, 1, 0.123_456_789_012_345),
        GraphEdge::weighted(1, 2, -0.999_999_999_999_999),
    ];
    let options = GraphWriteOptions {
        weight_type: WeightType::F64,
        ..Default::default()
    };
    storage
        .save_graph_to_path(&storage.file_path("adj"), &edges, &options)
        .await
        .expect("save_graph_to_path");

    // loads: the name-based readers (registry-free, path-resolved) and the
    // path-based readers round-trip exact values
    let loaded_vecs = storage.load_vectors("vecs").await.expect("load_vectors");
    assert_eq!(loaded_vecs.num_rows(), 2);
    assert_eq!(
        loaded_vecs
            .schema()
            .metadata()
            .get("kind")
            .map(String::as_str),
        Some("vector-space")
    );
    assert_eq!(
        loaded_vecs
            .schema()
            .metadata()
            .get("graph")
            .map(String::as_str),
        Some("adj"),
        "user properties survive as dataset metadata"
    );
    let values = loaded_vecs
        .column(1)
        .as_any()
        .downcast_ref::<FixedSizeListArray>()
        .unwrap()
        .values()
        .as_any()
        .downcast_ref::<Float64Array>()
        .unwrap();
    assert_eq!(values.value(0), 0.5);
    assert_eq!(values.value(3), 3.0);

    let loaded_adj = storage.load_graph("adj").await.expect("load_graph");
    assert_eq!(loaded_adj.weight_type, WeightType::F64);
    assert_eq!(loaded_adj.edges, edges);
    assert_eq!(loaded_adj.num_nodes, 3);
    let again = storage
        .load_graph_from_path(&storage.file_path("adj"))
        .await
        .expect("load_graph_from_path");
    assert_eq!(again.edges, edges);
    let vecs_again = storage
        .load_vectors_from_path(&storage.file_path("vecs"))
        .await
        .expect("load_vectors_from_path");
    assert_eq!(vecs_again.num_rows(), 2);

    // the consumer's metadata document is byte-identical: no write occurred,
    // and (being unreadable as GeneMetadata) no read occurred either
    let after = tokio::fs::read_to_string(storage.metadata_path())
        .await
        .unwrap();
    assert_eq!(after, app_metadata);
}

// ---------------------------------------------------------------------------
// P3: schema-declared weight widths (#106)
// ---------------------------------------------------------------------------

/// f64-declared weights round-trip bit-identically, including values an
/// f32 store corrupts (the measured genefold blocker: 100% of entries
/// changed, max abs diff 4.3e-7 on a real 65x65 laplacian).
#[tokio::test(flavor = "multi_thread")]
async fn graph_f64_weights_roundtrip_bit_exact() {
    let (_base, storage) = seeded_storage("graph_f64_weights").await;
    let md_path = storage.metadata_path();

    let weights = [
        0.0,
        -0.0,
        0.123_456_789_012_345,
        -0.999_999_999_999_999,
        f64::MIN_POSITIVE,
        1.0e-300,
        -2.5e-17,
        0.5,  // f32-exact control value
        1.0,  // ordinary endpoint value
        -1.0, // ordinary endpoint value
    ];
    let n: u64 = 65;
    let edges: Vec<GraphEdge> = weights
        .iter()
        .enumerate()
        .map(|(i, w)| GraphEdge::weighted(i as u64 % n, (i as u64 + 7) % n, *w))
        .collect();
    let options = GraphWriteOptions {
        weight_type: WeightType::F64,
        num_nodes: Some(n),
        ..Default::default()
    };
    storage
        .save_graph_with("laplacian_f64", &edges, &options, &md_path)
        .await
        .expect("save_graph_with f64 weights");

    let graph = storage
        .load_graph("laplacian_f64")
        .await
        .expect("load_graph");
    assert_eq!(graph.weight_type, WeightType::F64);
    assert!(graph.weighted);
    assert_eq!(graph.num_nodes, n);
    assert_eq!(graph.edges.len(), edges.len());
    for (got, want) in graph.edges.iter().zip(edges.iter()) {
        assert_eq!(got.src, want.src);
        assert_eq!(got.dst, want.dst);
        assert_eq!(
            got.weight.unwrap().to_bits(),
            want.weight.unwrap().to_bits(),
            "f64 weight bits mismatch for {}",
            want.weight.unwrap()
        );
    }

    // CSR conversion is exact at every stored coordinate (#106 acceptance)
    let csr = graph.to_csr().unwrap();
    assert_eq!(csr.rows(), n as usize);
    for (v, (r, c)) in csr.iter() {
        let want = edges
            .iter()
            .find(|e| e.src == r as u64 && e.dst == c as u64)
            .unwrap()
            .weight
            .unwrap();
        assert_eq!(v.to_bits(), want.to_bits(), "csr value at ({r},{c})");
    }

    // the weight width is stamped at both the dataset and registry level
    let stored = crate::lancefmt::scan_all(&storage.file_path("laplacian_f64")).unwrap();
    assert_eq!(
        stored.schema().field(2).data_type(),
        &DataType::Float64,
        "weight column is Float64 at the declared width"
    );
    assert_eq!(
        stored
            .schema()
            .metadata()
            .get("weight_type")
            .map(String::as_str),
        Some("f64")
    );
    let md = storage.load_metadata().await.unwrap();
    assert_eq!(
        md.files["laplacian_f64"]
            .properties
            .get("weight_type")
            .map(String::as_str),
        Some("f64")
    );
}

/// A generated laplacian-shaped graph (65 nodes, full-precision f64
/// weights, edge count crossing the lancefmt chunk boundaries) round-trips
/// exactly with `WeightType::F64`.
#[tokio::test(flavor = "multi_thread")]
async fn graph_f64_weights_generated_roundtrip() {
    let (_base, storage) = seeded_storage("graph_f64_generated").await;
    let md_path = storage.metadata_path();

    let n = 65u64;
    let mut rng = StdRng::seed_from_u64(106);
    let edges: Vec<GraphEdge> = (0..1500)
        .map(|_| {
            let src = rng.random_range(0..n);
            let dst = rng.random_range(0..n);
            // full double precision: essentially no value is f32-exact
            GraphEdge::weighted(src, dst, rng.random::<f64>() - 0.5)
        })
        .collect();
    let options = GraphWriteOptions {
        weight_type: WeightType::F64,
        num_nodes: Some(n),
        ..Default::default()
    };
    storage
        .save_graph_with("laplacian_gen", &edges, &options, &md_path)
        .await
        .expect("save_graph_with");

    let graph = storage
        .load_graph("laplacian_gen")
        .await
        .expect("load_graph");
    assert_eq!(graph.weight_type, WeightType::F64);
    assert_graph_round_trip(&graph, &edges, n, true);
}

/// The default weight width is `f64` — the same width and exactness as
/// the `value` column of the legacy sparse-matrix artifacts genefold-vd
/// stores: full-precision weights save via plain `save_graph` with no
/// declarations and round-trip bit-identically (#106).
#[tokio::test(flavor = "multi_thread")]
async fn graph_default_weights_are_f64_like_sparse_values() {
    let (_base, storage) = seeded_storage("graph_default_f64").await;
    let md_path = storage.metadata_path();

    let edges = vec![
        GraphEdge::weighted(0, 1, 0.123_456_789_012_345),
        GraphEdge::weighted(1, 2, -0.999_999_999_999_999),
        GraphEdge::weighted(2, 0, f64::MIN_POSITIVE),
    ];
    storage
        .save_graph("laplacian", &edges, &md_path)
        .await
        .expect("default save of full-precision weights");

    let graph = storage.load_graph("laplacian").await.expect("load_graph");
    assert_eq!(graph.weight_type, WeightType::F64);
    for (got, want) in graph.edges.iter().zip(edges.iter()) {
        assert_eq!(
            got.weight.unwrap().to_bits(),
            want.weight.unwrap().to_bits()
        );
    }
    let stored = crate::lancefmt::scan_all(&storage.file_path("laplacian")).unwrap();
    assert_eq!(
        stored.schema().field(2).data_type(),
        &DataType::Float64,
        "the default weight column is Float64, like the sparse value column"
    );
    assert_eq!(
        stored
            .schema()
            .metadata()
            .get("weight_type")
            .map(String::as_str),
        Some("f64")
    );
}

/// An explicitly declared `f32` weight width halves the storage bytes but
/// never narrows silently: inexact f64 weights are rejected with guidance
/// (the #51 invariant, float edition), out-of-range values surface
/// Overflow, and f32-representable weights round-trip bit-exactly.
#[tokio::test(flavor = "multi_thread")]
async fn graph_explicit_f32_rejects_inexact_f64_weights() {
    let (_base, storage) = seeded_storage("graph_f32_guard").await;
    let md_path = storage.metadata_path();

    let f32_options = GraphWriteOptions {
        weight_type: WeightType::F32,
        ..Default::default()
    };

    // inexact narrowing: rejected with guidance towards the f64 width
    let inexact = vec![GraphEdge::weighted(0, 1, 0.123_456_789_012_345)];
    let err = storage
        .save_graph_with("inexact", &inexact, &f32_options, &md_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "got {err:?}"
    );
    assert!(
        err.to_string().contains("WeightType::F64"),
        "the error must point at the f64 width: {err}"
    );

    // out-of-f32-range finite values surface Overflow, never silent
    // truncation (#51)
    let overflow = vec![GraphEdge::weighted(0, 1, 1.0e40)];
    let err = storage
        .save_graph_with("overflow", &overflow, &f32_options, &md_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Overflow(_)),
        "got {err:?}"
    );

    // f32-representable weights store as Float32 and round-trip bit-exactly
    let exact = vec![
        GraphEdge::weighted(0, 1, f64::from(0.5f32)),
        GraphEdge::weighted(1, 2, f64::from(-0.25f32)),
    ];
    storage
        .save_graph_with("exact", &exact, &f32_options, &md_path)
        .await
        .expect("f32-exact save at the declared width");
    let graph = storage.load_graph("exact").await.expect("load_graph");
    assert_eq!(graph.weight_type, WeightType::F32);
    for (got, want) in graph.edges.iter().zip(exact.iter()) {
        assert_eq!(
            got.weight.unwrap().to_bits(),
            want.weight.unwrap().to_bits()
        );
    }
    let stored = crate::lancefmt::scan_all(&storage.file_path("exact")).unwrap();
    assert_eq!(stored.schema().field(2).data_type(), &DataType::Float32);
    assert_eq!(
        stored
            .schema()
            .metadata()
            .get("weight_type")
            .map(String::as_str),
        Some("f32")
    );

    // topology-only collections carry no weight column regardless of the
    // declared width; the loaded graph reports the declared width
    let topo_f32 = vec![GraphEdge::unweighted(0, 1)];
    storage
        .save_graph_with("topo_f32", &topo_f32, &f32_options, &md_path)
        .await
        .expect("topology-only with F32 options");
    let graph = storage.load_graph("topo_f32").await.unwrap();
    assert!(!graph.weighted);
    assert!(graph.edges.iter().all(|e| e.weight.is_none()));
    assert_eq!(graph.weight_type, WeightType::F32);

    // non-finite weights are persisted faithfully at the f32 width:
    // infinities narrow bit-exactly; NaN is preserved as NaN
    // (classification semantics only — the payload is not guaranteed
    // bit-exact, unlike the f64 width)
    let nonfinite = vec![
        GraphEdge::weighted(0, 1, f64::INFINITY),
        GraphEdge::weighted(1, 2, f64::NEG_INFINITY),
        GraphEdge::weighted(2, 0, f64::NAN),
    ];
    storage
        .save_graph_with("nonfinite", &nonfinite, &f32_options, &md_path)
        .await
        .expect("non-finite weights are persisted faithfully");
    let graph = storage.load_graph("nonfinite").await.unwrap();
    assert_eq!(
        graph.edges[0].weight.unwrap().to_bits(),
        f64::INFINITY.to_bits()
    );
    assert_eq!(
        graph.edges[1].weight.unwrap().to_bits(),
        f64::NEG_INFINITY.to_bits()
    );
    assert!(graph.edges[2].weight.unwrap().is_nan());
}

/// Layering contract (#106): the storage layer persists weights
/// faithfully — no domain assumptions, no normalization. Values outside
/// any convention (including non-finite ones) save and round-trip
/// bit-identically at the default f64 width; normalization transforms
/// belong to the producer (genefold-vd).
#[tokio::test(flavor = "multi_thread")]
async fn graph_weights_persist_faithfully_without_a_range_check() {
    let (_base, storage) = seeded_storage("graph_faithful").await;
    let md_path = storage.metadata_path();

    let weights = [
        2.5,
        -7.25,
        1.0e40,
        -1.0e40,
        f64::INFINITY,
        f64::NEG_INFINITY,
        f64::NAN,
    ];
    let edges: Vec<GraphEdge> = weights
        .iter()
        .enumerate()
        .map(|(i, w)| GraphEdge::weighted(i as u64, (i + 1) as u64, *w))
        .collect();
    storage
        .save_graph("faithful", &edges, &md_path)
        .await
        .expect("no weight range is enforced by default");

    let graph = storage.load_graph("faithful").await.expect("load_graph");
    assert_eq!(graph.weight_type, WeightType::F64);
    for (got, want) in graph.edges.iter().zip(edges.iter()) {
        assert_eq!(got.src, want.src);
        assert_eq!(got.dst, want.dst);
        let w = want.weight.unwrap();
        if w.is_nan() {
            assert!(got.weight.unwrap().is_nan());
        } else {
            assert_eq!(got.weight.unwrap().to_bits(), w.to_bits());
        }
    }
}

/// The opt-in compliance check (#106): `weight_range` asserts that every
/// weight already lies in the producer's declared closed interval — it
/// catches a forgotten upstream transform but never transforms values
/// itself. Endpoints are inclusive, NaN lies in no interval, and an
/// inverted interval is a rejected configuration.
#[tokio::test(flavor = "multi_thread")]
async fn graph_weight_range_is_an_opt_in_compliance_check() {
    let (_base, storage) = seeded_storage("graph_weight_range").await;
    let md_path = storage.metadata_path();

    let ranged = GraphWriteOptions {
        weight_range: Some((0.0, 1.0)),
        ..Default::default()
    };

    // in-interval weights (endpoints included) pass the assertion
    let ok = vec![
        GraphEdge::weighted(0, 1, 0.0),
        GraphEdge::weighted(1, 2, 1.0),
        GraphEdge::weighted(2, 0, 0.123_456_789_012_345),
    ];
    storage
        .save_graph_with("normalized", &ok, &ranged, &md_path)
        .await
        .expect("in-interval weights pass the opt-in check");

    // violations — including NaN, which is in no interval — are rejected
    for (i, bad) in [-0.5, 1.5, f64::NAN, f64::INFINITY].iter().enumerate() {
        let edges = vec![GraphEdge::weighted(0, 1, *bad)];
        let err = storage
            .save_graph_with(&format!("bad_{i}"), &edges, &ranged, &md_path)
            .await
            .unwrap_err();
        assert!(
            matches!(err, crate::StorageError::Invalid(_)),
            "expected Invalid for weight {bad}, got {err:?}"
        );
        assert!(
            err.to_string().contains("[0.0, 1.0]"),
            "the error must name the declared interval: {err}"
        );
    }

    // an inverted interval is a rejected configuration
    let inverted = GraphWriteOptions {
        weight_range: Some((1.0, 0.0)),
        ..Default::default()
    };
    let edges = vec![GraphEdge::weighted(0, 1, 0.5)];
    let err = storage
        .save_graph_with("inverted", &edges, &inverted, &md_path)
        .await
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "got {err:?}"
    );

    // topology-only collections have no weights to assert: a declared
    // range is vacuous, not an error
    let topo = vec![GraphEdge::unweighted(0, 1)];
    storage
        .save_graph_with("topo_ranged", &topo, &ranged, &md_path)
        .await
        .expect("no weights, nothing to assert");
}

// ---------------------------------------------------------------------------
// Scalar collections (#106): kind=VectorSpace is uniformly loadable
// ---------------------------------------------------------------------------

/// `CollectionKind::for_filetype("vector")` maps scalar artifacts to
/// VectorSpace; `load_scalars` makes that kind uniformly loadable instead
/// of requiring the legacy fixed-key readers.
#[tokio::test(flavor = "multi_thread")]
async fn scalar_collections_load_via_load_scalars() {
    let (_base, storage) = seeded_storage("scalar_collections").await;
    let md_path = storage.metadata_path();

    // genefold's lambdas: a single Float64 column, registered as
    // kind=VectorSpace through the legacy "vector" filetype shim
    let lambdas: Vec<f64> = vec![0.0, 1.5, -2.25, f64::MIN_POSITIVE, 1.0e-300, 0.1];
    storage
        .save_lambdas(&lambdas, &md_path)
        .await
        .expect("save_lambdas");

    let md = storage.load_metadata().await.unwrap();
    assert_eq!(md.files["lambdas"].kind, Some(CollectionKind::VectorSpace));

    // the collections reader round-trips the exact values
    let loaded = storage.load_scalars("lambdas").await.expect("load_scalars");
    assert_eq!(loaded.len(), lambdas.len());
    for (got, want) in loaded.iter().zip(lambdas.iter()) {
        assert_eq!(got.to_bits(), want.to_bits());
    }

    // non-scalar collections are rejected: vector spaces go through
    // load_vectors, graphs through load_graph
    let batch = f64_vector_batch(&[0, 1], &[vec![1.0, 2.0], vec![3.0, 4.0]]);
    storage
        .save_vectors("vecs", &batch, &md_path)
        .await
        .unwrap();
    let err = storage.load_scalars("vecs").await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "got {err:?}"
    );

    let edges = vec![GraphEdge::weighted(0, 1, 0.5)];
    storage.save_graph("g", &edges, &md_path).await.unwrap();
    let err = storage.load_scalars("g").await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "got {err:?}"
    );

    // the logical kind is enforced, not just the physical shape (#106
    // review): a single Float64 column that is not a stamped vector-space
    // collection (a foreign one-column artifact) is rejected
    let bare = {
        let schema = Schema::new(vec![Field::new("value", DataType::Float64, false)]);
        RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(Float64Array::from(vec![1.0, 2.0])) as _],
        )
        .unwrap()
    };
    crate::lancefmt::write_dataset(&bare, &storage.file_path("foreign")).unwrap();
    let err = storage.load_scalars("foreign").await.unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "got {err:?}"
    );
    assert!(
        err.to_string().contains("kind"),
        "the rejection must name the kind mismatch: {err}"
    );
}

/// Review PR #96 finding 4: a `kind` property that contradicts the typed
/// `kind` field is rejected instead of silently overriding it.
#[tokio::test(flavor = "multi_thread")]
async fn register_table_rejects_kind_mismatch() {
    let base = tmp_dir("catalog_m_c1").await;
    let mut registry = LocalRegistry::new(GeneMetadata::new("catalog_test"), base.to_path_buf());

    let err = registry
        .register_table(TableDescriptor {
            name: "clashing".to_string(),
            format: "lance".to_string(),
            base_location: base.join("catalog_test_clashing.lance"),
            kind: CollectionKind::VectorSpace,
            properties: BTreeMap::from([("kind".to_string(), "graph".to_string())]),
        })
        .unwrap_err();
    assert!(
        matches!(err, crate::StorageError::Invalid(_)),
        "got {err:?}"
    );
    assert!(!registry.table_exists("clashing").unwrap());

    // agreement between the two sources of truth is fine
    registry
        .register_table(TableDescriptor {
            name: "agreeing".to_string(),
            format: "lance".to_string(),
            base_location: base.join("catalog_test_agreeing.lance"),
            kind: CollectionKind::Graph,
            properties: BTreeMap::from([("kind".to_string(), "graph".to_string())]),
        })
        .expect("agreeing kinds register");
    assert_eq!(
        registry.describe_table("agreeing").unwrap().kind,
        CollectionKind::Graph
    );
}
