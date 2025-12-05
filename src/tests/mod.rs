mod test_data;
mod test_lance_layer;
mod test_metadata;

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

// Logging harness
use std::sync::Once;

static INIT: Once = Once::new();

pub fn init() {
    INIT.call_once(|| {
        // Read RUST_LOG env variable, default to "info" if not set
        let env = env_logger::Env::default().default_filter_or("debug");

        // don't panic if called multiple times across binaries
        let _ = env_logger::Builder::from_env(env).try_init();
    });
}
