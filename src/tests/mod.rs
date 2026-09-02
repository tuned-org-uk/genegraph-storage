mod lancefmt_common;
mod test_catalog;
mod test_collections;
mod test_commit;
mod test_data;
mod test_generations;
mod test_lance_layer;
mod test_lancefmt_impl;
mod test_metadata;
mod test_parquet_io;

use std::fs;
use std::path::PathBuf;

pub(crate) async fn tmp_dir(test_name: &str) -> PathBuf {
    let mut d = std::env::temp_dir();
    let unique_name = format!(
        "{}_{}",
        test_name,
        uuid::Uuid::new_v4().to_string().replace("-", "")
    );
    d.push(format!(
        "{}_{}",
        unique_name,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap()
            .as_millis()
    ));
    fs::create_dir_all(&d).unwrap();
    d.canonicalize().unwrap_or(d)
}

// Logging harness: delegate to the crate-level initializer instead of
// keeping a duplicate INIT static (#95).
pub fn init() {
    crate::init();
}
