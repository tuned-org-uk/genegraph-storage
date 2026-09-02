use std::collections::HashMap;
use std::path::Path;

use crate::catalog::{Catalog, LocalRegistry, TableDescriptor};
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
            properties: Default::default(),
        })
        .unwrap_err();
    assert!(matches!(err, crate::StorageError::UnsupportedFormat(_)));

    let err = registry
        .register_table(TableDescriptor {
            name: "bad_props".to_string(),
            format: "lance".to_string(),
            base_location: base.join("bad_props.lance"),
            properties: HashMap::from([("rows".to_string(), "many".to_string())])
                .into_iter()
                .collect(),
        })
        .unwrap_err();
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
