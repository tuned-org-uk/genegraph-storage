//! Write deterministic NPY data for the external NumPy interoperability test.

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args().nth(1).expect("provide output path");
    genegraph_storage::npy::write_f64(
        std::path::Path::new(&path),
        &[2, 2],
        &[0.25, -1.5, 3.0, 9.25],
    )?;
    Ok(())
}
