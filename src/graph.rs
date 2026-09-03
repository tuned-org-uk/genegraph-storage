//! First-class graph storage types (RFC #81-P3).
//!
//! The canonical on-disk layout is an edge-list Lance dataset:
//! `src: UInt32|UInt64`, `dst: UInt32|UInt64` and an optional
//! `weight: Float32|Float64` column (all non-null). Node-id and weight
//! widths are schema-chosen — `u32` node-id and `f64` weight defaults,
//! `u64`/`f32` opt-ins (#106) — and declare the storage limits
//! explicitly rather than narrowing silently (consistent with #51).
//! Weights are persisted faithfully — the storage layer makes no domain
//! assumptions, so normalization transforms belong to the producer; the
//! opt-in [`GraphWriteOptions::weight_range`] asserts compliance with a
//! declared interval instead. `CsMat` CSR conversion stays at the API
//! boundary ([`StoredGraph::to_csr`]); sprs never reaches the format
//! layer.

use std::collections::BTreeMap;

use sprs::{CsMat, TriMat};

use crate::{StorageError, StorageResult};

/// Node-id storage width of a graph collection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum NodeIdWidth {
    /// `UInt32` ids (default); caps the graph at ~4.29 billion nodes.
    U32,
    /// `UInt64` ids (opt-in via schema).
    U64,
}

impl NodeIdWidth {
    pub const fn as_str(&self) -> &'static str {
        match self {
            NodeIdWidth::U32 => "u32",
            NodeIdWidth::U64 => "u64",
        }
    }

    pub fn parse(s: &str) -> StorageResult<Self> {
        match s {
            "u32" => Ok(NodeIdWidth::U32),
            "u64" => Ok(NodeIdWidth::U64),
            other => Err(StorageError::Invalid(format!(
                "unknown node-id width '{other}' (expected u32 or u64)"
            ))),
        }
    }

    /// The `DataType` used for `src`/`dst` columns at this width.
    pub(crate) const fn data_type(self) -> arrow::datatypes::DataType {
        match self {
            NodeIdWidth::U32 => arrow::datatypes::DataType::UInt32,
            NodeIdWidth::U64 => arrow::datatypes::DataType::UInt64,
        }
    }
}

/// Edge-weight storage width of a graph collection, mirroring
/// [`NodeIdWidth`] (#106): the width is schema-declared through
/// [`GraphWriteOptions::weight_type`], resolving the RFC #81 open
/// question ("Float64 only, or allow Float32?") with "both,
/// schema-declared".
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum WeightType {
    /// `Float64` weights (default): the same width and exactness as the
    /// `value` column of the legacy sparse-matrix artifacts, so an f64
    /// CSR (laplacian, adjacency) round-trips bit-identically through
    /// the graph API just like through `save_sparse`.
    F64,
    /// `Float32` weights (opt-in): half the storage bytes for
    /// memory-bound consumers. Weights that cannot be represented exactly
    /// at this width are rejected on save instead of being silently
    /// narrowed; finite values beyond the f32 range surface
    /// [`StorageError::Overflow`] (#51). Infinities narrow bit-exactly;
    /// NaN is preserved as NaN (classification semantics only — the
    /// payload is not guaranteed bit-exact, unlike the
    /// [`WeightType::F64`] width).
    F32,
}

impl WeightType {
    pub const fn as_str(&self) -> &'static str {
        match self {
            WeightType::F64 => "f64",
            WeightType::F32 => "f32",
        }
    }

    pub fn parse(s: &str) -> StorageResult<Self> {
        match s {
            "f64" => Ok(WeightType::F64),
            "f32" => Ok(WeightType::F32),
            other => Err(StorageError::Invalid(format!(
                "unknown weight width '{other}' (expected f32 or f64)"
            ))),
        }
    }

    /// The `DataType` used for the `weight` column at this width.
    pub(crate) const fn data_type(self) -> arrow::datatypes::DataType {
        match self {
            WeightType::F64 => arrow::datatypes::DataType::Float64,
            WeightType::F32 => arrow::datatypes::DataType::Float32,
        }
    }
}

/// A directed edge with an optional weight (RFC #81-P3).
///
/// `weight` is `f64` at the API boundary (the widest width, like node ids
/// are `u64`) and is persisted faithfully: the storage layer makes no
/// domain assumptions, so normalization transforms belong to the producer
/// (declare [`GraphWriteOptions::weight_range`] for an opt-in bounds
/// assertion on already-normalized values). The storage width is
/// schema-declared via [`GraphWriteOptions::weight_type`] — `Float64` by
/// default, mirroring the `value` column of the legacy sparse-matrix
/// artifacts — and weights that cannot be stored exactly at a declared
/// `Float32` width are rejected, never silently narrowed. A collection is
/// either fully weighted or topology-only; mixed edges are rejected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphEdge {
    pub src: u64,
    pub dst: u64,
    pub weight: Option<f64>,
}

impl GraphEdge {
    pub fn weighted(src: u64, dst: u64, weight: f64) -> Self {
        Self {
            src,
            dst,
            weight: Some(weight),
        }
    }

    pub fn unweighted(src: u64, dst: u64) -> Self {
        Self {
            src,
            dst,
            weight: None,
        }
    }
}

/// Options for [`crate::traits::backend::StorageBackend::save_graph_with`].
#[derive(Debug, Clone)]
pub struct GraphWriteOptions {
    /// Storage width for node ids (default `U32`); ids above the limit
    /// surface [`StorageError::Overflow`] instead of being truncated.
    pub node_id_width: NodeIdWidth,
    /// Storage width for edge weights (default `F64`, mirroring
    /// `node_id_width`, #106): weights are `f64` at the API boundary.
    /// The default matches the sparse-matrix `value` column (bit-exact
    /// f64 round trips); an explicit `F32` declaration halves the storage
    /// bytes but rejects values that cannot be stored exactly instead of
    /// silently narrowing them.
    pub weight_type: WeightType,
    /// Opt-in weight-range compliance check (#106): when `Some((low,
    /// high))`, every weight must already lie in the closed interval — a
    /// bounds assertion that catches a forgotten producer transform,
    /// never a transform the storage layer performs (normalization
    /// belongs upstream; values are persisted faithfully). NaN lies in
    /// no interval and is rejected; an inverted interval is a rejected
    /// configuration.
    pub weight_range: Option<(f64, f64)>,
    /// Explicit node count (e.g. to keep isolated vertices). Must be at
    /// least `max(src, dst) + 1` over all edges; defaults to exactly that.
    pub num_nodes: Option<u64>,
    /// User properties stored on the collection (registry level) and
    /// stamped into the dataset schema metadata (dataset level).
    pub properties: BTreeMap<String, String>,
}

impl Default for GraphWriteOptions {
    fn default() -> Self {
        Self {
            node_id_width: NodeIdWidth::U32,
            weight_type: WeightType::F64,
            weight_range: None,
            num_nodes: None,
            properties: BTreeMap::new(),
        }
    }
}

impl GraphWriteOptions {
    pub fn with_width(node_id_width: NodeIdWidth) -> Self {
        Self {
            node_id_width,
            ..Default::default()
        }
    }
}

/// Options for the graph read path (#108).
///
/// The tolerant default is the compat shim: an unstamped dataset loads with
/// `num_nodes = max_id + 1` and columns are read positionally (name-agnostic),
/// so pre-collections triplet artifacts still load. Strict mode makes the
/// hazard detectable at the API instead of silently resizing.
#[derive(Debug, Clone, Copy, Default)]
pub struct GraphReadOptions {
    /// When `true`, reject datasets without a `num_nodes` stamp with a typed
    /// [`StorageError::Invalid`] instead of silently resizing to `max_id + 1`
    /// (a graph whose trailing vertices are isolated would otherwise load
    /// with a smaller node count than was written). Also validates the
    /// `src`/`dst`/`weight` column names, positively identifying
    /// pre-collections triplet artifacts (`row`/`col`/`value`) rather than
    /// positionally coercing them.
    pub strict: bool,
}

impl GraphReadOptions {
    /// Strict read mode: reject unstamped `num_nodes` and validate column
    /// names (#108).
    pub fn strict() -> Self {
        Self { strict: true }
    }
}

/// A graph collection loaded back from storage (RFC #81-P3).
#[derive(Debug, Clone, PartialEq)]
pub struct StoredGraph {
    /// Edges in stored order; `weight` is `None` for topology-only
    /// collections. `f32`-stored weights are upcast to `f64` losslessly
    /// at load time.
    pub edges: Vec<GraphEdge>,
    /// Storage width used for node ids.
    pub node_id_width: NodeIdWidth,
    /// Storage width used for the `weight` column (mirrors
    /// `node_id_width`, #106); for topology-only collections it reports
    /// the declared width from the dataset metadata (the `F64` default
    /// when unstamped) and is meaningful only when `weighted`.
    pub weight_type: WeightType,
    /// Node count (from the dataset metadata; isolated vertices included).
    pub num_nodes: u64,
    /// Whether the edge list carries weights.
    pub weighted: bool,
}

impl StoredGraph {
    /// Converts the edge list to a CSR matrix at the API boundary.
    ///
    /// Topology-only graphs get weight `1.0` (unweighted convention); the
    /// result is `CsMat<f64>` — weights are `f64` at the API boundary, so
    /// `WeightType::F64` collections round-trip exactly.
    pub fn to_csr(&self) -> StorageResult<CsMat<f64>> {
        let n = self.num_nodes;
        if n > usize::MAX as u64 {
            return Err(StorageError::Overflow(format!(
                "node count {n} exceeds the addressable size"
            )));
        }
        let n = n as usize;
        let mut trimat = TriMat::new((n, n));
        trimat.reserve(self.edges.len());
        for edge in &self.edges {
            let (src, dst) = (edge.src as usize, edge.dst as usize);
            if src >= n || dst >= n {
                return Err(StorageError::Invalid(format!(
                    "edge ({}, {}) out of bounds for {}-node graph",
                    edge.src, edge.dst, n
                )));
            }
            let weight = edge.weight.unwrap_or(1.0);
            trimat.add_triplet(src, dst, weight);
        }
        let csr = trimat.to_csr();
        if csr.rows() != n || csr.cols() != n {
            return Err(StorageError::Invalid(format!(
                "dimension mismatch after CSR conversion: expected {n}x{n}, got {}x{}",
                csr.rows(),
                csr.cols()
            )));
        }
        Ok(csr)
    }
}

/// Dataset-level schema-metadata keys stamped by the `save_graph` /
/// `save_vectors` writers (RFC #81-P1: dataset-level kinds alongside
/// registry-level kinds). User properties may not shadow these.
pub(crate) const RESERVED_METADATA_KEYS: [&str; 5] = [
    "kind",
    "node_id_width",
    "weighted",
    "num_nodes",
    "weight_type",
];
