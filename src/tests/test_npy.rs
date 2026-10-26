#[tokio::test]
async fn npy_f64_roundtrips_matrix_scalar_and_empty_shapes() {
    let root = crate::tests::tmp_dir("npy-f64").await;
    let path = root.join("matrix.npy");
    let expected = vec![0.0, -0.0, 1.25, f64::INFINITY, f64::NEG_INFINITY, f64::NAN];
    crate::npy::write_f64(&path, &[2, 3], &expected).unwrap();
    let (shape, actual) = crate::npy::read_f64(&path).unwrap();
    assert_eq!(shape, [2, 3]);
    assert_eq!(
        actual.iter().map(|v| v.to_bits()).collect::<Vec<_>>(),
        expected.iter().map(|v| v.to_bits()).collect::<Vec<_>>()
    );
    let scalar = root.join("scalar.npy");
    crate::npy::write_f64(&scalar, &[], &[42.5]).unwrap();
    assert_eq!(crate::npy::read_f64(&scalar).unwrap(), (vec![], vec![42.5]));
    let empty = root.join("empty.npy");
    crate::npy::write_f64(&empty, &[2, 0], &[]).unwrap();
    assert_eq!(crate::npy::read_f64(&empty).unwrap(), (vec![2, 0], vec![]));
}

#[tokio::test]
async fn npy_reads_numpy_generated_f64_and_u64_fixtures() {
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

#[tokio::test]
async fn npy_u64_preserves_full_unsigned_range() {
    let root = crate::tests::tmp_dir("npy-u64").await;
    let path = root.join("values.npy");
    let values = [0, u32::MAX as u64 + 5, u64::MAX];
    crate::npy::write_u64(&path, &[3], &values).unwrap();
    assert_eq!(
        crate::npy::read_u64(&path).unwrap(),
        (vec![3], values.to_vec())
    );
}

#[tokio::test]
async fn npy_dense_matrix_file_io_preserves_row_order() {
    use crate::traits::backend::StorageBackend;
    use smartcore::linalg::basic::arrays::Array;
    use smartcore::linalg::basic::matrix::DenseMatrix;

    let root = crate::tests::tmp_dir("npy-dense").await;
    let path = root.join("dense.npy");
    let matrix = DenseMatrix::new(2, 3, vec![1.0, 4.0, 2.0, 5.0, 3.0, 6.0], true).unwrap();
    crate::lance_storage_graph::LanceStorageGraph::save_dense_to_file(&matrix, &path)
        .await
        .unwrap();
    let storage = crate::lance_storage_graph::LanceStorageGraph::new(
        root.to_string_lossy().to_string(),
        "npy_test".to_string(),
    )
    .unwrap();
    let loaded = storage.load_dense_from_file(&path).await.unwrap();
    assert_eq!(loaded.shape(), (2, 3));
    for row in 0..2 {
        for col in 0..3 {
            assert_eq!(*loaded.get((row, col)), *matrix.get((row, col)));
        }
    }
}

#[tokio::test]
async fn npy_rejects_wrong_dtype_for_requested_reader() {
    let root = crate::tests::tmp_dir("npy-dtype").await;
    let path = root.join("values.npy");
    crate::npy::write_u64(&path, &[1], &[1]).unwrap();
    assert!(matches!(
        crate::npy::read_f64(&path),
        Err(crate::StorageError::UnsupportedFormat(_))
    ));
}

#[tokio::test]
async fn npy_rejects_shape_mismatch_and_truncated_payload() {
    let root = crate::tests::tmp_dir("npy-bad").await;
    let path = root.join("values.npy");
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
