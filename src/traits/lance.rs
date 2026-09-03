use std::collections::BTreeMap;
use std::path::Path;
use std::sync::Arc;

use arrow::array::{
    ArrayRef, Float32Array, Float64Array, PrimitiveArray, UInt32Array, UInt64Array,
};
use arrow::datatypes::{ArrowPrimitiveType, DataType, Field, Schema};
use arrow::record_batch::RecordBatch;
use log::{debug, info};

use crate::catalog::RESERVED_PROPERTIES;
use crate::graph::{
    GraphEdge, GraphWriteOptions, NodeIdWidth, RESERVED_METADATA_KEYS, StoredGraph, WeightType,
};
use crate::metadata::FileInfo;
use crate::traits::backend::StorageBackend;
use crate::{StorageError, StorageResult};

/// Rejects user properties that would shadow computed registry facts
/// (`RESERVED_PROPERTIES`, review of PR #96): the direct `save_*_with`
/// write paths must enforce the same rule as `Catalog::register_table`, or
/// a caller-provided `rows`/`nnz`/... would silently override the computed
/// descriptor values.
pub(crate) fn reject_reserved_user_properties(
    properties: &BTreeMap<String, String>,
) -> StorageResult<()> {
    for key in properties.keys() {
        if RESERVED_PROPERTIES.contains(&key.as_str()) {
            return Err(StorageError::Invalid(format!(
                "user property '{key}' is reserved (computed from the stored artifact)"
            )));
        }
    }
    Ok(())
}

/// Resolves a `file://` URI (as produced by `path_to_uri`) to a local path.
fn uri_to_path(uri: &str) -> StorageResult<std::path::PathBuf> {
    let url = url::Url::parse(uri)
        .map_err(|e| StorageError::Invalid(format!("bad dataset URI `{uri}`: {e}")))?;
    url.to_file_path().map_err(|_| {
        StorageError::Invalid(format!("dataset URI is not a local file path: `{uri}`"))
    })
}

/// Stamps collection metadata (RFC #81-P1) into the batch schema: the
/// dataset-level `kind`, writer-computed pairs, and the user properties.
/// Existing schema metadata is preserved; user properties may not shadow
/// reserved keys and the `kind` value always wins.
pub(crate) fn with_collection_metadata(
    batch: &RecordBatch,
    kind: &str,
    fixed: &[(&'static str, String)],
    user_properties: &BTreeMap<String, String>,
) -> StorageResult<RecordBatch> {
    for key in user_properties.keys() {
        if RESERVED_METADATA_KEYS.contains(&key.as_str()) {
            return Err(StorageError::Invalid(format!(
                "user property '{key}' is reserved by the collection metadata"
            )));
        }
    }
    let mut metadata = batch.schema().metadata().clone();
    for (k, v) in fixed {
        metadata.insert(k.to_string(), v.clone());
    }
    for (k, v) in user_properties {
        metadata.insert(k.clone(), v.clone());
    }
    metadata.insert("kind".to_string(), kind.to_string());

    let schema = Arc::new(batch.schema().as_ref().clone().with_metadata(metadata));
    RecordBatch::try_new(schema, batch.columns().to_vec())
        .map_err(|e| StorageError::Lance(e.to_string()))
}

/// Validates a vector-space schema (RFC #81-P2): at least one
/// `FixedSizeList<Float64|Float32>` column, no nullable top-level fields,
/// and any additional (id/property) columns limited to the lancefmt scalar
/// subset. Returns the dimension of the first vector column.
pub(crate) fn validate_vector_space_schema(schema: &Schema) -> StorageResult<(i32, DataType)> {
    let fields = schema.fields();
    if fields.is_empty() {
        return Err(StorageError::Invalid(
            "vector-space schema has no columns".into(),
        ));
    }
    let mut vector_dim: Option<(i32, DataType)> = None;
    for f in fields {
        if f.is_nullable() {
            return Err(StorageError::Invalid(format!(
                "vector-space schemas require non-nullable columns; '{}' is nullable",
                f.name()
            )));
        }
        match f.data_type() {
            DataType::FixedSizeList(child, dim)
                if matches!(child.data_type(), DataType::Float64 | DataType::Float32) =>
            {
                if vector_dim.is_none() {
                    vector_dim = Some((*dim, child.data_type().clone()));
                }
            }
            DataType::Float32
            | DataType::Float64
            | DataType::UInt8
            | DataType::UInt32
            | DataType::UInt64
            | DataType::Int64 => {}
            other => {
                return Err(StorageError::Invalid(format!(
                    "vector-space id/property column '{}' has unsupported type {other:?}; \
                     expected FixedSizeList<Float64|Float32> or a scalar column",
                    f.name()
                )));
            }
        }
    }
    vector_dim.ok_or_else(|| {
        StorageError::Invalid(
            "vector-space schema requires at least one FixedSizeList<Float64|Float32> column"
                .into(),
        )
    })
}

/// Validates a vector-space batch and stamps the dataset-level collection
/// metadata: the shared write path of the registry-coupled
/// `save_vectors_with` and the registry-free `save_vectors_to_path`
/// (#106). Returns the stamped batch and the vector dimension.
pub(crate) fn vector_space_collection_batch(
    batch: &RecordBatch,
    properties: &BTreeMap<String, String>,
) -> StorageResult<(RecordBatch, i32)> {
    reject_reserved_user_properties(properties)?;
    if batch.num_rows() == 0 {
        return Err(StorageError::Invalid(
            "empty vector-space collections are not supported".into(),
        ));
    }
    let (dim, _item_type) = validate_vector_space_schema(batch.schema().as_ref())?;
    let stamped = with_collection_metadata(batch, "vector-space", &[], properties)?;
    Ok((stamped, dim))
}

/// Validates an edge list and converts it into the canonical graph
/// collection batch (the shared write path of `save_graph_with` and the
/// registry-free `save_graph_to_path`, #106): non-empty, uniformly
/// weighted-or-topology edges, node ids within the declared width
/// (Overflow instead of truncation, #51), a consistent `num_nodes`, and
/// weights exactly representable at the declared weight width (silent
/// narrowing rejected). Returns the stamped batch and the effective node
/// count.
pub(crate) fn graph_record_batch(
    edges: &[GraphEdge],
    options: &GraphWriteOptions,
) -> StorageResult<(RecordBatch, u64)> {
    reject_reserved_user_properties(&options.properties)?;
    if edges.is_empty() {
        return Err(StorageError::Invalid(
            "empty graph collections are not supported".into(),
        ));
    }
    // A collection is fully weighted or topology-only; mixed edges are
    // rejected instead of being coerced.
    let weighted = edges[0].weight.is_some();
    if edges.iter().any(|e| e.weight.is_some() != weighted) {
        return Err(StorageError::Invalid(
            "graph collection must be either fully weighted or topology-only; \
             mixed edges are rejected"
                .into(),
        ));
    }
    // The schema declares the storage-type limit; ids above it surface
    // Overflow instead of being silently truncated (consistent with #51).
    let max_id = edges.iter().map(|e| e.src.max(e.dst)).max().unwrap_or(0);
    let width = options.node_id_width;
    if width == NodeIdWidth::U32 && max_id > u32::MAX as u64 {
        return Err(StorageError::Overflow(format!(
            "node id {max_id} exceeds u32::MAX; save with NodeIdWidth::U64"
        )));
    }
    let num_nodes = match options.num_nodes {
        Some(n) if n < max_id + 1 => {
            return Err(StorageError::Invalid(format!(
                "num_nodes {n} is smaller than the highest node id + 1 ({})",
                max_id + 1
            )));
        }
        Some(n) => n,
        None => max_id + 1,
    };
    // Weight handling (#106):
    // - Layering: the crate persists weights faithfully — no domain
    //   assumptions, no normalization. Producer transforms
    //   (x/(1+x), 1-exp(-x), ...) happen upstream; `weight_range` is
    //   only an opt-in bounds assertion on already-computed values
    //   (it catches a forgotten transform). NaN lies in no interval;
    //   an inverted interval is a rejected configuration.
    if let Some((lo, hi)) = options.weight_range
        && !matches!(
            lo.partial_cmp(&hi),
            Some(std::cmp::Ordering::Less | std::cmp::Ordering::Equal)
        )
    {
        return Err(StorageError::Invalid(format!(
            "weight_range ({lo:?}, {hi:?}) is not a valid interval: expected low <= high"
        )));
    }
    if weighted {
        if let Some((lo, hi)) = options.weight_range {
            for e in edges {
                let w = e.weight.unwrap();
                if !(lo..=hi).contains(&w) {
                    return Err(StorageError::Invalid(format!(
                        "graph weight {w} is outside the declared weight_range [{lo:?}, {hi:?}]"
                    )));
                }
            }
        }
        // f32 narrowing guard — the #51 invariant, float edition: a
        // declared f32 store must hold every weight exactly, because a
        // silent f64 -> f32 -> f64 narrowing corrupts full-precision
        // values (measured on a real laplacian: 100% of entries change,
        // max abs diff 4.3e-7). The default f64 width never narrows.
        if options.weight_type == WeightType::F32 {
            for e in edges {
                let w = e.weight.unwrap();
                let narrowed = w as f32;
                if w.is_finite() && !narrowed.is_finite() {
                    return Err(StorageError::Overflow(format!(
                        "weight {w} exceeds the f32 range; save with WeightType::F64"
                    )));
                }
                let exact = f64::from(narrowed).to_bits() == w.to_bits()
                    || (w.is_nan() && narrowed.is_nan());
                if !exact {
                    return Err(StorageError::Invalid(format!(
                        "weight {w} is not exactly representable as f32 and would be \
                         silently narrowed; save with WeightType::F64 or pre-narrow the \
                         value with `as f32`"
                    )));
                }
            }
        }
    }

    let (src, dst): (ArrayRef, ArrayRef) = match width {
        NodeIdWidth::U32 => (
            Arc::new(UInt32Array::from_iter_values(
                edges.iter().map(|e| e.src as u32),
            )),
            Arc::new(UInt32Array::from_iter_values(
                edges.iter().map(|e| e.dst as u32),
            )),
        ),
        NodeIdWidth::U64 => (
            Arc::new(UInt64Array::from_iter_values(edges.iter().map(|e| e.src))),
            Arc::new(UInt64Array::from_iter_values(edges.iter().map(|e| e.dst))),
        ),
    };
    let mut fields = vec![
        Field::new("src", width.data_type(), false),
        Field::new("dst", width.data_type(), false),
    ];
    let mut columns: Vec<ArrayRef> = vec![src, dst];
    if weighted {
        fields.push(Field::new("weight", options.weight_type.data_type(), false));
        let weights = edges.iter().map(|e| e.weight.unwrap());
        columns.push(match options.weight_type {
            WeightType::F32 => {
                Arc::new(Float32Array::from_iter_values(weights.map(|w| w as f32))) as ArrayRef
            }
            WeightType::F64 => Arc::new(Float64Array::from_iter_values(weights)) as ArrayRef,
        });
    }
    let batch = RecordBatch::try_new(Arc::new(Schema::new(fields)), columns)
        .map_err(|e| StorageError::Lance(e.to_string()))?;
    // Dataset-level kind + graph layout facts (RFC #81-P1/P3, #106).
    let batch = with_collection_metadata(
        &batch,
        "graph",
        &[
            ("node_id_width", width.as_str().to_string()),
            ("weighted", weighted.to_string()),
            ("num_nodes", num_nodes.to_string()),
            ("weight_type", options.weight_type.as_str().to_string()),
        ],
        &options.properties,
    )?;
    Ok((batch, num_nodes))
}

/// Parses a graph collection batch back into a [`StoredGraph`] (the
/// counterpart of [`graph_record_batch`], shared by `load_graph` and the
/// registry-free `load_graph_from_path`): shared src/dst width, a
/// `Float32|Float64` weight column (#106), lossless f32 upcasting, and
/// the node count from dataset metadata.
pub(crate) fn stored_graph_from_batch(batch: RecordBatch) -> StorageResult<StoredGraph> {
    let schema = batch.schema();
    let n_cols = schema.fields().len();
    if !(2..=3).contains(&n_cols) {
        return Err(StorageError::Invalid(format!(
            "graph edge-list schema expects src/dst (and optionally weight) columns, \
             found {n_cols}"
        )));
    }
    let width = match (schema.field(0).data_type(), schema.field(1).data_type()) {
        (DataType::UInt32, DataType::UInt32) => NodeIdWidth::U32,
        (DataType::UInt64, DataType::UInt64) => NodeIdWidth::U64,
        (s, d) => {
            return Err(StorageError::Invalid(format!(
                "graph src/dst columns must share width UInt32|UInt64, found {s:?}/{d:?}"
            )));
        }
    };
    let weighted = n_cols == 3;
    let weight_type = if weighted {
        match schema.field(2).data_type() {
            DataType::Float32 => WeightType::F32,
            DataType::Float64 => WeightType::F64,
            other => {
                return Err(StorageError::Invalid(format!(
                    "graph weight column must be Float32|Float64, found {other:?}"
                )));
            }
        }
    } else {
        // Topology-only: report the declared width from the dataset
        // metadata (the F64 default when unstamped).
        schema
            .metadata()
            .get("weight_type")
            .and_then(|v| WeightType::parse(v).ok())
            .unwrap_or(WeightType::F64)
    };

    let ids = |col: usize, what: &'static str| -> StorageResult<Vec<u64>> {
        let arr = batch.column(col);
        if arr.null_count() != 0 {
            return Err(StorageError::Invalid(format!(
                "graph '{what}' column contains nulls"
            )));
        }
        match width {
            NodeIdWidth::U32 => {
                let a = arr.as_any().downcast_ref::<UInt32Array>().ok_or_else(|| {
                    StorageError::Invalid(format!("graph '{what}' column type mismatch"))
                })?;
                Ok((0..a.len()).map(|i| a.value(i) as u64).collect())
            }
            NodeIdWidth::U64 => {
                let a = arr.as_any().downcast_ref::<UInt64Array>().ok_or_else(|| {
                    StorageError::Invalid(format!("graph '{what}' column type mismatch"))
                })?;
                Ok((0..a.len()).map(|i| a.value(i)).collect())
            }
        }
    };
    let src = ids(0, "src")?;
    let dst = ids(1, "dst")?;
    let weights: Option<Vec<f64>> = if weighted {
        let w = batch.column(2);
        if w.null_count() != 0 {
            return Err(StorageError::Invalid(
                "graph 'weight' column contains nulls".into(),
            ));
        }
        match weight_type {
            WeightType::F32 => {
                // lossless upcast at the API boundary
                let a = w.as_any().downcast_ref::<Float32Array>().ok_or_else(|| {
                    StorageError::Invalid("graph 'weight' column type mismatch".into())
                })?;
                Some((0..a.len()).map(|i| f64::from(a.value(i))).collect())
            }
            WeightType::F64 => {
                let a = w.as_any().downcast_ref::<Float64Array>().ok_or_else(|| {
                    StorageError::Invalid("graph 'weight' column type mismatch".into())
                })?;
                Some((0..a.len()).map(|i| a.value(i)).collect())
            }
        }
    } else {
        None
    };

    // Node count: dataset-level metadata if present (isolated vertices
    // included), otherwise the highest id + 1.
    let max_id = src
        .iter()
        .zip(dst.iter())
        .map(|(s, d)| (*s).max(*d))
        .max()
        .unwrap_or(0);
    let num_nodes = schema
        .metadata()
        .get("num_nodes")
        .and_then(|v| v.parse::<u64>().ok())
        .unwrap_or(max_id + 1);
    let edges: Vec<GraphEdge> = match &weights {
        Some(weights) => src
            .iter()
            .zip(dst.iter())
            .zip(weights.iter())
            .map(|((s, d), w)| GraphEdge::weighted(*s, *d, *w))
            .collect(),
        None => src
            .iter()
            .zip(dst.iter())
            .map(|(s, d)| GraphEdge::unweighted(*s, *d))
            .collect(),
    };
    if let Some(e) = edges
        .iter()
        .find(|e| e.src >= num_nodes || e.dst >= num_nodes)
    {
        return Err(StorageError::Invalid(format!(
            "edge ({}, {}) out of bounds for {}-node graph",
            e.src, e.dst, num_nodes
        )));
    }

    Ok(StoredGraph {
        edges,
        node_id_width: width,
        weight_type,
        num_nodes,
        weighted,
    })
}

pub trait LanceStorage {
    /// Async helper: write a RecordBatch to a Lance dataset.
    ///
    /// Runs the in-house v2.1 implementation (`lancefmt`) on the blocking
    /// pool (#75 M5; the `official-lance` transition flag was removed after
    /// one release cycle).
    async fn write_lance_batch_async(&self, uri: String, batch: RecordBatch) -> StorageResult<()> {
        info!("Writing Lance dataset (in-house v2.1) to {}", uri);
        let path = uri_to_path(&uri)?;
        tokio::task::spawn_blocking(move || crate::lancefmt::write_dataset(&batch, &path))
            .await
            .map_err(|e| StorageError::Io(format!("lancefmt writer task failed: {e}")))?
    }

    /// Async helper: read and concatenate all RecordBatches from a Lance dataset.
    async fn read_lance_all_batches_async(&self, uri: String) -> StorageResult<RecordBatch> {
        info!("Reading Lance dataset (in-house v2.1) from {}", uri);
        let path = uri_to_path(&uri)?;
        let combined = tokio::task::spawn_blocking(move || crate::lancefmt::scan_all(&path))
            .await
            .map_err(|e| StorageError::Io(format!("lancefmt reader task failed: {e}")))??;
        debug!(
            "Combined Lance batch for {:?} has {} rows",
            uri,
            combined.num_rows()
        );
        Ok(combined)
    }

    /// Writes a single-column primitive vector as a Lance dataset named
    /// `<name>_<key>.lance` and registers it in the metadata files map under
    /// `key` with filetype "vector" and shape (`<len>`, 1).
    ///
    /// Shared implementation for the scalar `save_*` methods (lambdas,
    /// vectors, indices, norms, centroid maps, cluster assignments).
    async fn save_primitive_column<T: ArrowPrimitiveType>(
        &self,
        key: &str,
        field_name: &str,
        values: Vec<T::Native>,
        md_path: &Path,
    ) -> StorageResult<()>
    where
        Self: StorageBackend,
    {
        self.validate_initialized(md_path)?;
        let path = self.file_path(key);
        let len = values.len();
        info!("Saving {} values for {} (field {})", len, key, field_name);

        // Dataset-level kind (RFC #81-P1): scalar artifacts are
        // vector-space collections, mirroring the registry-level
        // `CollectionKind::for_filetype("vector")` shim, so logical-kind
        // readers (`load_scalars`) can enforce the kind (#106 review).
        let schema = Schema::new(vec![Field::new(field_name, T::DATA_TYPE, false)]).with_metadata(
            std::collections::HashMap::from([("kind".to_string(), "vector-space".to_string())]),
        );
        let batch = RecordBatch::try_new(
            Arc::new(schema),
            vec![Arc::new(PrimitiveArray::<T>::from_iter_values(values)) as ArrayRef],
        )
        .map_err(|e| StorageError::Lance(e.to_string()))?;

        crate::commit::with_commit_actor(&self.metadata_path(), || async {
            let mut metadata = self.load_metadata().await?;
            metadata = metadata.add_file(
                key,
                FileInfo::new(
                    format!("{}_{}.lance", self.get_name(), key),
                    "vector",
                    (len, 1),
                    None,
                    None,
                )?,
            );
            self.save_metadata(&metadata).await
        })
        .await?;

        let uri = Self::path_to_uri(&path)?;
        self.write_lance_batch_async(uri, batch).await?;
        info!("Vector {} saved successfully", key);
        Ok(())
    }
}
