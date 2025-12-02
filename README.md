# genegraph-storage

A Lance storage layer for graph-based vector databases.

A storage layer for [`javelin-tui`](https://github.com/tuned-org-uk/javelin-tui) and [`arrowspace`](https://github.com/Mec-iS/arrowspace-rs).

## Usage
```rust
// instantiate a storage
let storage = LanceStorage::new(
    "/tmp".to_string_lossy().to_string(),
    "basic_test".to_string(),
);

// some 2D data
let dense: Vec<Vec<f64>> = vec![
    vec![0.1, 0.4, 0.5, 0.2, 0.9],
    vec![0.4, 0.5, 0.2, 0.9, 0.3],
    vec![0.03, 0.8, 0.56, 0.2, 0.9],
    vec![0.1, 0.4, 0.5, 0.34, 0.9],
    vec![0.05, 0.4, 0.2, 0.3, 0.7]
];

let (nitems, nfeatures) = dense.len(), dense[0].len(); 
let data =
    DenseMatrix::<f64>::from_iterator(
        dense.iter().flatten().map(|x| *x), nitems, nfeatures, 0);

// Save metadata FIRST to initialize the storage directory
let md = GeneMetadata::seed_metadata(name, nitems, nfeatures, &storage.clone())
    .await
    .unwrap()
    .with_dimensions(nitems, nfeatures)
    .add_file(
        "rawinput",
        FileInfo::new(
            expected_filename.clone(),
            "dense",
            (nitems, nfeatures),
            None,
            None,
        ),
    );

// your data is saved in an efficient format
storage
    .save_dense("rawinput", &data, &md_path)
    .await
    .unwrap();

// Loading back
let md_path = storage.save_metadata(&md).await.unwrap();
let loaded = storage.load_dense("rawinput").await.unwrap();
```

## Contributing
See `.github/` directory.
