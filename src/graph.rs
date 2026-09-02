//! First-class graph storage types (RFC #81-P3).
//!
//! The canonical on-disk layout is an edge-list Lance dataset:
//! `src: UInt32|UInt64`, `dst: UInt32|UInt64` and an optional
//! `weight: Float32` column (all non-null). Node-id width is schema-chosen
//! (`u32` default, `u64` opt-in) and declares the storage-type limit
//! explicitly rather than truncating (consistent with #51). `CsMat` CSR
//! conversion stays at the API boundary ([`StoredGraph::to_csr`]); sprs
//! never reaches the format layer.

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

/// A directed edge with an optional weight (RFC #81-P3).
///
/// `weight` is `f32`, per the RFC decision that graph weights live in
/// `[-1.0, 1.0]`. A collection is either fully weighted or topology-only;
/// mixed edges are rejected.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct GraphEdge {
    pub src: u64,
    pub dst: u64,
    pub weight: Option<f32>,
}

impl GraphEdge {
    pub fn weighted(src: u64, dst: u64, weight: f32) -> Self {
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

/// A graph collection loaded back from storage (RFC #81-P3).
#[derive(Debug, Clone, PartialEq)]
pub struct StoredGraph {
    /// Edges in stored order; `weight` is `None` for topology-only
    /// collections.
    pub edges: Vec<GraphEdge>,
    /// Storage width used for node ids.
    pub node_id_width: NodeIdWidth,
    /// Node count (from the dataset metadata; isolated vertices included).
    pub num_nodes: u64,
    /// Whether the edge list carries weights.
    pub weighted: bool,
}

impl StoredGraph {
    /// Converts the edge list to a CSR matrix at the API boundary.
    ///
    /// Topology-only graphs get weight `1.0` (unweighted convention); the
    /// result is `CsMat<f64>` with weights upcast from `f32` losslessly.
    pub fn to_csr(&self) -> StorageResult<CsMat<f64>> {
        let n = self.num_nodes;
        if n > usize::MAX as u64 {
            return Err(StorageError::Overflow(format!(
                "node count {n} exceeds the addressable size"
            )));
        }
        let n = n as usize;
        let mut trimat = TriMat::new((n, n));
        trimat.reserve(nnz_hint(self.edges.len()));
        for edge in &self.edges {
            let (src, dst) = (edge.src as usize, edge.dst as usize);
            if src >= n || dst >= n {
                return Err(StorageError::Invalid(format!(
                    "edge ({}, {}) out of bounds for {}-node graph",
                    edge.src, edge.dst, n
                )));
            }
            let weight = match edge.weight {
                Some(w) => w as f64,
                None => 1.0,
            };
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

fn nnz_hint(len: usize) -> usize {
    // TriMat::reserve clamps to capacity; keep the hint modest.
    len.min(1 << 20)
}

/// Dataset-level schema-metadata keys stamped by the `save_graph` /
/// `save_vectors` writers (RFC #81-P1: dataset-level kinds alongside
/// registry-level kinds). User properties may not shadow these.
pub(crate) const RESERVED_METADATA_KEYS: [&str; 4] =
    ["kind", "node_id_width", "weighted", "num_nodes"];
