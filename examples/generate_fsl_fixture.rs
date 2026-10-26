//! Generate the in-house non-nullable FSL fixture for issue #147.

use std::path::Path;
use std::sync::Arc;

use arrow::array::{Array, FixedSizeListArray, Float64Array};
use arrow::datatypes::{DataType, Field, Schema};
use arrow::record_batch::RecordBatch;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let output = std::env::args().nth(1).expect("provide a new output path");
    let path = Path::new(&output);
    assert!(!path.exists(), "output path must not exist");
    let child = Arc::new(Field::new("item", DataType::Float64, false));
    let values = (0..400).map(|i| i as f64 * 0.125).collect::<Vec<_>>();
    let list = FixedSizeListArray::new(child, 8, Arc::new(Float64Array::from(values)), None);
    let schema = Schema::new(vec![Field::new("vector", list.data_type().clone(), false)]);
    let batch = RecordBatch::try_new(Arc::new(schema), vec![Arc::new(list)])?;
    genegraph_storage::lancefmt::write_dataset(&batch, path)?;
    Ok(())
}
