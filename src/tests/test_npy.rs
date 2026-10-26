use std::path::PathBuf;

fn temp_file(name: &str) -> PathBuf {
    std::env::temp_dir().join(format!("{name}-{}.npy", uuid::Uuid::new_v4()))
}

#[test]
fn npy_f64_roundtrips_matrix_and_empty_shapes() {
    let path = temp_file("npy-f64");
    let expected = vec![0.0, -0.0, 1.25, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
    crate::npy::write_f64(&path, &[2, 3], &expected).unwrap();
    let (shape, actual) = crate::npy::read_f64(&path).unwrap();
    assert_eq!(shape, [2, 3]);
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let empty = temp_file("npy-empty");
    crate::npy::write_f64(&empty, &[2, 0], &[]).unwrap();
    assert_eq!(crate::npy::read_f64(&empty).unwrap(), (vec![2, 0], vec![]));
}

#[test]
fn npy_reads_numpy_generated_f64_and_u64_fixtures() {
    let root = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/npy");
    assert_eq!(
        crate::npy::read_f64(&root.join("numpy_f64_c_order.npy")).unwrap(),
        (vec![2, 2], vec![0.25, -1.5, 3.0, 9.25])
    );
    assert_eq!(
        crate::npy::read_u64(&root.join("numpy_u64_indices.npy")).unwrap(),
        (vec![3], vec![0, 2u64.pow(32) + 5, u64::MAX])
    );
}

#[test]
fn npy_u64_preserves_full_unsigned_range() {
    let path = temp_file("npy-u64");
    let values = [0, u32::MAX as u64 + 5, u64::MAX];
    crate::npy::write_u64(&path, &[3], &values).unwrap();
    assert_eq!(
        crate::npy::read_u64(&path).unwrap(),
        (vec![3], values.to_vec())
    );
}

#[test]
fn npy_dense_matrix_file_io_preserves_row_order() {
    use crate::traits::backend::StorageBackend;
    use smartcore::linalg::basic::arrays::Array;
    use smartcore::linalg::basic::matrix::DenseMatrix;

    let root = std::env::temp_dir().join(format!("npy-dense-{}", uuid::Uuid::new_v4()));
    let path = root.join("dense.npy");
    let matrix = DenseMatrix::new(2, 3, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], true).unwrap();
    tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(crate::lance_storage_graph::LanceStorageGraph::save_dense_to_file(&matrix, &path))
        .unwrap();
    let storage = crate::lance_storage_graph::LanceStorageGraph::new(
        root.to_string_lossy().to_string(),
        "npy_test".to_string(),
    )
    .unwrap();
    let loaded = tokio::runtime::Runtime::new()
        .unwrap()
        .block_on(storage.load_dense_from_file(&path))
        .unwrap();
    assert_eq!(loaded.shape(), (2, 3));
    for row in 0..2 {
        for col in 0..3 {
            assert_eq!(*loaded.get((row, col)), *matrix.get((row, col)));
        }
    }
}

#[test]
fn npy_rejects_wrong_dtype_for_requested_reader() {
    let path = temp_file("npy-dtype");
    crate::npy::write_u64(&path, &[1], &[1]).unwrap();
    assert!(matches!(
        crate::npy::read_f64(&path),
        Err(crate::StorageError::UnsupportedFormat(_))
    ));
}

#[test]
fn npy_rejects_shape_mismatch_and_truncated_payload() {
    let path = temp_file("npy-bad");
    assert!(matches!(
        crate::npy::write_f64(&path, &[2, 3], &[1.0]),
        Err(crate::StorageError::DimensionMismatch { .. })
    ));
    crate::npy::write_f64(&path, &[1], &[1.0]).unwrap();
    let mut bytes = std::fs::read(&path).unwrap();
    bytes.pop();
    std::fs::write(&path, bytes).unwrap();
    assert!(matches!(
        crate::npy::read_f64(&path),
        Err(crate::StorageError::Invalid(_))
    ));
}
