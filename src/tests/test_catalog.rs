use std::collections::{BTreeMap, HashMap};
use std::path::Path;

use crate::catalog::{Catalog, CollectionKind, LocalRegistry, TableDescriptor};
use crate::lance_storage_graph::LanceStorageGraph;
use crate::metadata::GeneMetadata;
use crate::tests::tmp_dir;
use crate::traits::metadata::Metadata;

fn registry_with(base: &Path, name_id: &str) -> LocalRegistry {
    let metadata = GeneMetadata::new(name_id)
        .with_base(base.to_path_buf())
        .with_dimensions(100, 50);
    let raw_info = metadata
        .new_fileinfo("rawinput", "dense", (100, 50), None, None)
        .expect("dense filetype");
    let adj_info = metadata
        .new_fileinfo("adjacency", "sparse", (100, 100), Some(500), None)
        .expect("sparse filetype");
    let metadata = metadata
        .add_file("rawinput", raw_info)
        .add_file("adjacency", adj_info);
    LocalRegistry::new(metadata, base.to_path_buf())
}

#[tokio::test(flavor = "multi_thread")]
async fn catalog_lists_and_describes_tables() {
    let base = tmp_dir("catalog_m_c1").await;
    let name_id = "catalog_test";
    let registry = registry_with(&base, name_id);

    let tables = registry.list_tables().expect("list");
    let names: Vec<_> = tables.iter().map(|t| t.name.clone()).collect();
    assert_eq!(names, vec!["adjacency", "rawinput"], "sorted by name");
    assert!(tables.iter().all(|t| t.format == "lance"));

    let adj = registry.describe_table("adjacency").expect("describe");
    assert_eq!(
        adj.base_location,
        base.join(format!("{name_id}_adjacency.lance"))
    );
    assert_eq!(
        adj.properties.get("filetype").map(String::as_str),
        Some("sparse")
    );
    assert_eq!(adj.properties.get("nnz").map(String::as_str), Some("500"));

    assert!(registry.table_exists("rawinput").unwrap());
    assert!(!registry.table_exists("missing").unwrap());
    assert!(registry.describe_table("missing").is_err());
}

#[tokio::test(flavor = "multi_thread")]
async fn catalog_register_and_deregister_roundtrip() {
    let base = tmp_dir("catalog_m_c1").await;
    let mut registry = registry_with(&base, "catalog_test");

    registry
        .register_table(TableDescriptor {
            name: "norms".to_string(),
            format: "lance".to_string(),
            base_location: base.join("catalog_test_norms.lance"),
            kind: CollectionKind::VectorSpace,
            properties: HashMap::from([
                ("filetype".to_string(), "vector".to_string()),
                ("rows".to_string(), "100".to_string()),
                ("cols".to_string(), "1".to_string()),
            ])
            .into_iter()
            .collect(),
        })
        .expect("register");

    assert!(registry.table_exists("norms").unwrap());
    let norms = registry.describe_table("norms").unwrap();
    assert_eq!(norms.base_location, base.join("catalog_test_norms.lance"));
    assert_eq!(
        norms.properties.get("rows").map(String::as_str),
        Some("100")
    );

    registry.deregister_table("norms").expect("deregister");
    assert!(!registry.table_exists("norms").unwrap());
    assert!(registry.deregister_table("norms").is_err());

    // the mutated metadata must be recoverable for persistence
    let metadata = registry.into_metadata();
    assert!(!metadata.files.contains_key("norms"));
    assert!(metadata.files.contains_key("rawinput"));
}

#[tokio::test(flavor = "multi_thread")]
async fn catalog_rejects_non_lance_format_and_bad_properties() {
    let base = tmp_dir("catalog_m_c1").await;
    let mut registry = registry_with(&base, "catalog_test");

    let err = registry
        .register_table(TableDescriptor {
            name: "delta_thing".to_string(),
            format: "delta".to_string(),
            base_location: base.join("delta_thing"),
            kind: CollectionKind::Table,
            properties: Default::default(),
        })
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::UnsupportedFormat(_)));

    let err = registry
        .register_table(TableDescriptor {
            name: "bad_props".to_string(),
            format: "lance".to_string(),
            base_location: base.join("bad_props.lance"),
            kind: CollectionKind::Table,
            properties: HashMap::from([("rows".to_string(), "many".to_string())])
                .into_iter()
                .collect(),
        })
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)));
}

/// RFC #81-P1: legacy artifact keys become pre-seeded collections — kinds are
/// derived from the filetype (dense/vector -> vector-space, sparse -> graph)
/// and descriptors carry a `kind` property alongside the typed field.
#[tokio::test(flavor = "multi_thread")]
async fn catalog_derives_kinds_for_legacy_keys() {
    let base = tmp_dir("catalog_m_c1").await;
    let registry = registry_with(&base, "catalog_test");

    let raw = registry.describe_table("rawinput").unwrap();
    assert_eq!(raw.kind, CollectionKind::VectorSpace);
    assert_eq!(
        raw.properties.get("kind").map(String::as_str),
        Some("vector-space")
    );

    let adj = registry.describe_table("adjacency").unwrap();
    assert_eq!(adj.kind, CollectionKind::Graph);
    assert_eq!(
        adj.properties.get("kind").map(String::as_str),
        Some("graph")
    );
}

/// RFC #81-P1: user properties and an explicit kind round-trip through
/// `register_table` -> `describe_table`, and are persisted inside the
/// metadata `files` map (registry-level kinds).
#[tokio::test(flavor = "multi_thread")]
async fn catalog_kinds_and_user_properties_roundtrip() {
    let base = tmp_dir("catalog_m_c1").await;
    let mut registry = registry_with(&base, "catalog_test");

    registry
        .register_table(TableDescriptor {
            name: "edges".to_string(),
            format: "lance".to_string(),
            base_location: base.join("catalog_test_edges.lance"),
            kind: CollectionKind::Graph,
            properties: BTreeMap::from([
                ("filetype".to_string(), "graph".to_string()),
                ("rows".to_string(), "7".to_string()),
                ("cols".to_string(), "7".to_string()),
                ("nnz".to_string(), "12".to_string()),
                ("node_id_width".to_string(), "u32".to_string()),
            ]),
        })
        .expect("register graph collection");

    let desc = registry.describe_table("edges").unwrap();
    assert_eq!(desc.kind, CollectionKind::Graph);
    assert_eq!(
        desc.properties.get("node_id_width").map(String::as_str),
        Some("u32"),
        "user properties pass through"
    );

    // the user property must be stored on the FileInfo itself
    let metadata = registry.into_metadata();
    let info = metadata.files.get("edges").unwrap();
    assert_eq!(info.kind, Some(CollectionKind::Graph));
    assert_eq!(
        info.properties.get("node_id_width").map(String::as_str),
        Some("u32")
    );
}

/// RFC #81-P1: pre-0.28 metadata JSON (no `kind` / `properties` fields on
/// FileInfo) still deserializes and the registry derives collection kinds.
#[tokio::test(flavor = "multi_thread")]
async fn catalog_legacy_metadata_json_still_loads() {
    let legacy_json = r#"{
        "name_id": "legacy_instance",
        "nrows": 10,
        "ncols": 5,
        "base": "/tmp/legacy_instance",
        "files": {
            "rawinput": {
                "filename": "legacy_instance_rawinput.lance",
                "filetype": "dense",
                "storage_format": "lance fixed-row",
                "rows": 10,
                "cols": 5,
                "nnz": null,
                "size_bytes": null
            }
        },
        "created_at": "2026-01-01T00:00:00Z"
    }"#;
    let metadata: GeneMetadata = serde_json::from_str(legacy_json).expect("legacy JSON loads");
    assert!(metadata.files["rawinput"].kind.is_none());
    assert!(metadata.files["rawinput"].properties.is_empty());

    let registry = LocalRegistry::new(metadata, std::path::PathBuf::from("/tmp/legacy_instance"));
    let desc = registry.describe_table("rawinput").unwrap();
    assert_eq!(desc.kind, CollectionKind::VectorSpace);
}

/// RFC #81-P4: `describe_vector_space` returns the vector-space descriptor
/// plus the graph collection referenced by the `graph` user property.
#[tokio::test(flavor = "multi_thread")]
async fn catalog_describe_vector_space_links_graph() {
    let base = tmp_dir("catalog_m_c1").await;
    let mut registry = registry_with(&base, "catalog_test");

    registry
        .register_table(TableDescriptor {
            name: "embeddings".to_string(),
            format: "lance".to_string(),
            base_location: base.join("catalog_test_embeddings.lance"),
            kind: CollectionKind::VectorSpace,
            properties: BTreeMap::from([
                ("filetype".to_string(), "vectors".to_string()),
                ("rows".to_string(), "100".to_string()),
                ("cols".to_string(), "50".to_string()),
                ("graph".to_string(), "adjacency".to_string()),
            ]),
        })
        .expect("register vector space");

    let vs = registry.describe_vector_space("embeddings").unwrap();
    assert_eq!(vs.vectors.name, "embeddings");
    let graph = vs.graph.expect("linked graph");
    assert_eq!(graph.name, "adjacency");
    assert_eq!(graph.kind, CollectionKind::Graph);

    // a vector space without a graph link has no linked descriptor
    registry
        .register_table(TableDescriptor {
            name: "bare".to_string(),
            format: "lance".to_string(),
            base_location: base.join("catalog_test_bare.lance"),
            kind: CollectionKind::VectorSpace,
            properties: BTreeMap::from([
                ("filetype".to_string(), "vectors".to_string()),
                ("rows".to_string(), "4".to_string()),
                ("cols".to_string(), "4".to_string()),
            ]),
        })
        .expect("register bare vector space");
    assert!(
        registry
            .describe_vector_space("bare")
            .unwrap()
            .graph
            .is_none()
    );

    // kind mismatches and dangling links are errors, not guesses
    let err = registry.describe_vector_space("adjacency").unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)));
    assert!(registry.describe_vector_space("missing").is_err());

    registry
        .register_table(TableDescriptor {
            name: "dangling".to_string(),
            format: "lance".to_string(),
            base_location: base.join("catalog_test_dangling.lance"),
            kind: CollectionKind::VectorSpace,
            properties: BTreeMap::from([
                ("filetype".to_string(), "vectors".to_string()),
                ("graph".to_string(), "no_such_graph".to_string()),
            ]),
        })
        .expect("register dangling vector space");
    let err = registry.describe_vector_space("dangling").unwrap_err();
    assert!(matches!(err, crate::StorageError::Invalid(_)));
}

/// The registry resolves against a real LanceStorageGraph dataset: locations
/// point at the actual `*.lance` directories.
#[tokio::test(flavor = "multi_thread")]
async fn catalog_locations_match_storage_layout() {
    let base = tmp_dir("catalog_m_c1_storage").await;
    let name_id = "catalog_storage";
    let storage = LanceStorageGraph::new(base.to_string_lossy().to_string(), name_id.to_string());
    let metadata = GeneMetadata::seed_metadata(name_id, 10, 5, &storage)
        .await
        .expect("seed");

    let registry = LocalRegistry::new(metadata, base.to_path_buf());
    assert!(
        registry.list_tables().unwrap().is_empty(),
        "fresh dataset has no tables"
    );
    assert_eq!(
        registry.describe_table("rawinput").unwrap_err().to_string(),
        "Invalid data: table 'rawinput' is not registered"
    );
}
