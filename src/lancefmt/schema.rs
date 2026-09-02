//! Arrow schema <-> Lance field conversion for the supported type subset.
//!
//! Supported: `Float64` ("double"), `UInt32` ("uint32") and
//! `FixedSizeList<Float64>` ("fixed_size_list:double:N"), all non-nullable
//! at the top level. Anything else is rejected with
//! [`StorageError::UnsupportedFormat`] (never guessed, see #75).

use std::sync::Arc;

use arrow::datatypes::{DataType, Field, Schema};
use prost::Message;

use crate::StorageError;
use crate::StorageResult;
use crate::lancefmt::pb::file;

pub(crate) const PLAIN_ENCODING: i32 = 1;

pub(crate) fn logical_type_name(dt: &DataType) -> StorageResult<String> {
    match dt {
        DataType::Float64 => Ok("double".to_string()),
        DataType::UInt32 => Ok("uint32".to_string()),
        DataType::Int64 => Ok("int64".to_string()),
        DataType::FixedSizeList(child, n) if matches!(child.data_type(), DataType::Float64) => {
            Ok(format!("fixed_size_list:double:{n}"))
        }
        other => Err(StorageError::UnsupportedFormat(format!(
            "unsupported Arrow type for lancefmt: {other:?}"
        ))),
    }
}

fn parse_logical_type(name: &str) -> StorageResult<DataType> {
    match name {
        "double" => Ok(DataType::Float64),
        "uint32" => Ok(DataType::UInt32),
        "int64" => Ok(DataType::Int64),
        _ => {
            if let Some(rest) = name.strip_prefix("fixed_size_list:double:") {
                let dim: i32 = rest.parse().map_err(|_| {
                    StorageError::UnsupportedFormat(format!("bad fixed_size_list dim: {name}"))
                })?;
                return Ok(DataType::FixedSizeList(
                    Arc::new(Field::new("item", DataType::Float64, true)),
                    dim,
                ));
            }
            Err(StorageError::UnsupportedFormat(format!(
                "unsupported Lance logical type: {name}"
            )))
        }
    }
}

/// Encodes an Arrow schema into the flat Lance field list (+ schema metadata).
pub(crate) fn to_lance_schema(schema: &Schema) -> StorageResult<(Vec<file::Field>, SchemaMeta)> {
    let mut fields = Vec::with_capacity(schema.fields().len());
    for (id, f) in schema.fields().iter().enumerate() {
        if f.is_nullable() {
            return Err(StorageError::UnsupportedFormat(format!(
                "lancefmt writer requires non-nullable top-level fields; '{}' is nullable",
                f.name()
            )));
        }
        fields.push(file::Field {
            r#type: 0,
            name: f.name().clone(),
            id: id as i32,
            parent_id: -1,
            logical_type: logical_type_name(f.data_type())?,
            nullable: false,
            metadata: Default::default(),
            unenforced_primary_key: false,
            unenforced_primary_key_position: 0,
            unenforced_clustering_key: false,
            unenforced_clustering_key_position: 0,
            encoding: PLAIN_ENCODING,
            dictionary: None,
            extension_name: String::new(),
        });
    }
    let meta: SchemaMeta = schema
        .metadata()
        .iter()
        .map(|(k, v)| (k.clone(), v.clone().into_bytes()))
        .collect();
    Ok((fields, meta))
}

pub(crate) type SchemaMeta = std::collections::HashMap<String, Vec<u8>>;

/// Rebuilds an Arrow schema from the flat Lance field list.
///
/// Mirrors official-reader behavior: FSL child fields are nullable.
pub(crate) fn from_lance_fields(
    fields: &[file::Field],
    metadata: &SchemaMeta,
) -> StorageResult<Schema> {
    let mut arrow_fields = Vec::with_capacity(fields.len());
    for f in fields {
        if f.parent_id != -1 {
            return Err(StorageError::UnsupportedFormat(format!(
                "lancefmt reader only supports top-level fields; '{}' has parent_id {}",
                f.name, f.parent_id
            )));
        }
        let dt = parse_logical_type(&f.logical_type)?;
        arrow_fields.push(Field::new(&f.name, dt, f.nullable));
    }
    let meta = metadata
        .iter()
        .map(|(k, v)| {
            let s = String::from_utf8(v.clone())
                .map_err(|_| StorageError::Invalid("schema metadata is not UTF-8".into()))?;
            Ok((k.clone(), s))
        })
        .collect::<StorageResult<std::collections::HashMap<_, _>>>()?;
    Ok(Schema::new_with_metadata(arrow_fields, meta))
}

pub(crate) fn encode_schema_global(
    fields: &[file::Field],
    meta: &SchemaMeta,
    num_rows: u64,
) -> Vec<u8> {
    let descriptor = file::FileDescriptor {
        schema: Some(file::Schema {
            fields: fields.to_vec(),
            metadata: meta.clone(),
        }),
        length: num_rows,
    };
    descriptor.encode_to_vec()
}
