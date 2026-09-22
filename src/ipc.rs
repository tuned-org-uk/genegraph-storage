//! Arrow-IPC interop codec (#142).
//!
//! Format codec only — like `zzarr/`, this module holds no locks and no
//! registry logic: backends compose matrix/graph batch conversion and the
//! metadata publish cycle around it.
//!
//! Two artifacts are supported:
//!
//! - the Arrow IPC *file* format (`.arrow`): magic, message blocks and a
//!   footer (`ARROW1\0\0` head, `[footer][i32 footer_len][ARROW1]` tail);
//! - the Arrow IPC *stream* format (`.arrows`): messages ending in an EOS
//!   marker — the incremental-flush surface of [`IpcStreamWriterHandle`].
//!
//! Errors surface typed (`StorageError::IPC`), never guessed at.

use std::fs::File;
use std::io::BufWriter;
use std::path::{Path, PathBuf};

use arrow::ipc::reader::{FileReader, StreamReader};
use arrow::ipc::writer::{FileWriter, StreamWriter};
use arrow::record_batch::RecordBatch;
use log::{debug, info};

use crate::{StorageError, StorageResult};

/// Extension of Arrow IPC *file* artifacts (`.arrow`).
pub const FILE_EXT: &str = "arrow";
/// Extension of Arrow IPC *stream* artifacts (`.arrows`).
pub const STREAM_EXT: &str = "arrows";

/// Registry storage-format token for IPC file artifacts (#142).
pub const FILE_STORAGE_FORMAT: &str = "arrow-ipc";
/// Registry storage-format token for IPC stream artifacts (#142).
pub const STREAM_STORAGE_FORMAT: &str = "arrow-ipc-stream";

/// Maps an `arrow-ipc` codec failure onto [`StorageError::IPC`].
fn ipc_err(what: &str, e: arrow::error::ArrowError) -> StorageError {
    StorageError::IPC(format!("{what}: {e}"))
}

/// Writes `batches` as one Arrow IPC *file* at `path` (the writer flushes
/// the footer on finish, closing the spec trailer). The file is truncated
/// on overwrite — interop exports follow write-on-save semantics.
pub fn write_ipc_file(path: &Path, batches: &[RecordBatch]) -> StorageResult<()> {
    let batch = batches
        .first()
        .ok_or_else(|| StorageError::Invalid("cannot write an empty IPC file".into()))?;
    let file = File::create(path).map_err(|e| {
        StorageError::Io(format!("Failed to create IPC file {}: {e}", path.display()))
    })?;
    let mut writer = FileWriter::try_new_buffered(file, batch.schema().as_ref()).map_err(|e| {
        ipc_err(
            &format!("Failed to create IPC file writer at {}", path.display()),
            e,
        )
    })?;
    for batch in batches {
        writer.write(batch).map_err(|e| {
            ipc_err(
                &format!("Failed to write IPC batch to {}", path.display()),
                e,
            )
        })?;
    }
    writer
        .finish()
        .map_err(|e| ipc_err(&format!("Failed to finish IPC file {}", path.display()), e))?;
    info!(
        "Wrote Arrow IPC file {} ({} batches)",
        path.display(),
        batches.len()
    );
    Ok(())
}

/// Reads all record batches of an Arrow IPC *file* and concatenates them
/// in file order. An artifact without batches is rejected.
pub fn read_ipc_file(path: &Path) -> StorageResult<RecordBatch> {
    let file = File::open(path).map_err(|e| {
        StorageError::Io(format!("Failed to open IPC file {}: {e}", path.display()))
    })?;
    let reader = FileReader::try_new(file, None).map_err(|e| {
        ipc_err(
            &format!("Failed to open IPC file reader for {}", path.display()),
            e,
        )
    })?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().map_err(|e| {
        ipc_err(
            &format!("Failed to read IPC file {}: {e}", path.display()),
            e,
        )
    })?;
    combined_batches(path, batches)
}

/// Reads all record batches of an Arrow IPC *stream* and concatenates
/// them in flush order. A stream without batches is rejected.
pub fn read_ipc_stream(path: &Path) -> StorageResult<RecordBatch> {
    let file = File::open(path).map_err(|e| {
        StorageError::Io(format!("Failed to open IPC stream {}: {e}", path.display()))
    })?;
    let reader = StreamReader::try_new(file, None).map_err(|e| {
        ipc_err(
            &format!(
                "Failed to open IPC stream reader for {}: {e}",
                path.display()
            ),
            e,
        )
    })?;
    let batches: Vec<RecordBatch> = reader.collect::<Result<Vec<_>, _>>().map_err(|e| {
        ipc_err(
            &format!("Failed to read IPC stream {}: {e}", path.display()),
            e,
        )
    })?;
    combined_batches(path, batches)
}

/// Reads an Arrow IPC artifact at `path`, dispatching on the file
/// extension: `.arrow` opens the file-format reader, `.arrows` the
/// stream-format reader. Any other extension is a typed rejection.
pub fn read_ipc_artifact(path: &Path) -> StorageResult<RecordBatch> {
    match path.extension().and_then(|e| e.to_str()) {
        Some(FILE_EXT) => read_ipc_file(path),
        Some(STREAM_EXT) => read_ipc_stream(path),
        other => Err(StorageError::Invalid(format!(
            "not an Arrow IPC artifact extension '{other:?}': {}",
            path.display()
        ))),
    }
}

/// Concatenates the collected batches of an artifact in read order; an
/// empty artifact is a typed rejection, never a silently empty matrix.
fn combined_batches(path: &Path, batches: Vec<RecordBatch>) -> StorageResult<RecordBatch> {
    if batches.is_empty() {
        return Err(StorageError::Invalid(format!(
            "Empty Arrow IPC artifact at {}",
            path.display()
        )));
    }
    let schema = batches[0].schema();
    let combined = arrow::compute::concat_batches(&schema, &batches).map_err(|e| {
        ipc_err(
            &format!("Failed to concatenate IPC batches at {}", path.display()),
            e,
        )
    })?;
    debug!(
        "IPC artifact {}: {} batches -> {} rows",
        path.display(),
        batches.len(),
        combined.num_rows()
    );
    Ok(combined)
}

/// Streaming Arrow IPC writer handle (`*.arrows`): the incremental-flush
/// surface of [`crate::traits::backend::StorageBackend::open_ipc_stream_writer`].
///
/// The handle owns the open file. `write_batch` flushes one record batch;
/// [`finish`](Self::finish) writes the EOS marker, flushes and closes the
/// file, returning the artifact path. Batches must match the schema the
/// stream was opened with (the codec validates and surfaces a typed
/// error). Dropping the handle without `finish` closes the file without
/// an EOS marker: a stream artifact is only complete after `finish`.
pub struct IpcStreamWriterHandle {
    writer: Option<StreamWriter<BufWriter<File>>>,
    path: PathBuf,
}

impl IpcStreamWriterHandle {
    /// Creates the stream artifact at `path` and writes the schema
    /// message. The parent directory must exist.
    pub fn create(path: PathBuf, schema: &arrow::datatypes::Schema) -> StorageResult<Self> {
        let file = File::create(&path).map_err(|e| {
            StorageError::Io(format!(
                "Failed to create IPC stream {}: {e}",
                path.display()
            ))
        })?;
        let writer = StreamWriter::try_new_buffered(file, schema).map_err(|e| {
            ipc_err(
                &format!("Failed to create IPC stream writer at {}", path.display()),
                e,
            )
        })?;
        info!("Opened Arrow IPC stream writer at {}", path.display());
        Ok(Self {
            writer: Some(writer),
            path,
        })
    }

    /// The artifact path this handle flushes to.
    pub fn path(&self) -> &Path {
        &self.path
    }

    /// Flushes one record batch. The batch schema must match the schema
    /// the stream was opened with.
    pub fn write_batch(&mut self, batch: &RecordBatch) -> StorageResult<()> {
        let writer = self
            .writer
            .as_mut()
            .ok_or_else(|| StorageError::Invalid("IPC stream writer is already finished".into()))?;
        writer.write(batch).map_err(|e| {
            ipc_err(
                &format!(
                    "Failed to write IPC stream batch to {}",
                    self.path.display()
                ),
                e,
            )
        })
    }

    /// Writes the EOS marker, flushes and closes the file. Returns the
    /// artifact path.
    pub fn finish(mut self) -> StorageResult<PathBuf> {
        let mut writer = self
            .writer
            .take()
            .ok_or_else(|| StorageError::Invalid("IPC stream writer is already finished".into()))?;
        writer.finish().map_err(|e| {
            ipc_err(
                &format!("Failed to finish IPC stream at {}", self.path.display()),
                e,
            )
        })?;
        info!("Closed Arrow IPC stream at {}", self.path.display());
        Ok(self.path)
    }
}

// =========
// Async wrappers: the codec is synchronous; backends share these so the
// blocking dispatch exists once, not per backend (#145 review).
// =========

/// [`write_ipc_file`] off the async executor thread.
pub async fn write_ipc_file_async(path: PathBuf, batches: Vec<RecordBatch>) -> StorageResult<()> {
    tokio::task::spawn_blocking(move || write_ipc_file(&path, &batches))
        .await
        .map_err(|e| StorageError::Io(format!("ipc writer task failed: {e}")))?
}

/// [`read_ipc_artifact`] off the async executor thread.
pub async fn read_ipc_artifact_async(path: PathBuf) -> StorageResult<RecordBatch> {
    tokio::task::spawn_blocking(move || read_ipc_artifact(&path))
        .await
        .map_err(|e| StorageError::Io(format!("ipc reader task failed: {e}")))?
}

/// [`IpcStreamWriterHandle::create`] off the async executor thread.
pub async fn open_stream_writer_async(
    path: PathBuf,
    schema: arrow::datatypes::SchemaRef,
) -> StorageResult<IpcStreamWriterHandle> {
    tokio::task::spawn_blocking(move || IpcStreamWriterHandle::create(path, schema.as_ref()))
        .await
        .map_err(|e| StorageError::Io(format!("ipc stream writer task failed: {e}")))?
}
