"""Verify Rust-written zzarr arrays with Python zarr (round-trip leg).

Run after the ignored Rust conformance test:
    cargo test --release --lib -- --ignored zzarr_conformance_output
    uv run --with "zarr>=3.0" --with numpy python tests/fixtures/zarr/verify_rust_output.py

Exits non-zero on any mismatch.
"""

import sys
from pathlib import Path

import numpy as np
import zarr

HERE = Path(__file__).parent
OUT = HERE / "out_rust"

failures = []


def check(label: str, got: np.ndarray, want: np.ndarray) -> None:
    if got.dtype != want.dtype or got.shape != want.shape or not np.array_equal(got, want):
        failures.append(f"{label}: got {got!r}, want {want!r}")
    else:
        print(f"ok {label}")


def read(path: Path) -> np.ndarray:
    return np.asarray(zarr.open(store=zarr.storage.LocalStore(path), mode="r"))


check("f32_1d.zarr", read(OUT / "f32_1d.zarr"), np.arange(10, dtype=np.float32) * np.float32(0.5))
check("i64_1d.zarr", read(OUT / "i64_1d.zarr"), np.arange(10, dtype=np.int64))

slug = OUT / "main--matrix"
check("csr data.zarr", read(slug / "data.zarr"), np.array([0.5, 1.5, 2.5], dtype=np.float32))
check("csr indices.zarr", read(slug / "indices.zarr"), np.array([0, 1, 1], dtype=np.int64))
check("csr indptr.zarr", read(slug / "indptr.zarr"), np.array([0, 1, 2, 3], dtype=np.int64))

meta = __import__("json").loads((slug / "meta.json").read_text())
want_meta = {"nitems": 3, "nfeatures": 2, "nclusters": 1, "csr_shape": [3, 2]}
if meta != want_meta:
    failures.append(f"meta.json: got {meta}, want {want_meta}")
else:
    print("ok meta.json")

if failures:
    print("\nFAILURES:")
    for f in failures:
        print(" -", f)
    sys.exit(1)
print("\nall Rust-written arrays read back by Python zarr")
