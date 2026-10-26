"""Generate NumPy-originated NPY fixtures and validate Rust output."""

from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).parent


def generate() -> None:
    np.save(
        ROOT / "numpy_f64_c_order.npy",
        np.array([[0.25, -1.5], [3.0, 9.25]], dtype="<f8"),
    )
    np.save(ROOT / "numpy_u64_indices.npy", np.array([0, 2**32 + 5, 2**64 - 1], dtype="<u8"))


def verify_rust_output(path: Path) -> None:
    actual = np.load(path, allow_pickle=False)
    expected = np.array([[0.25, -1.5], [3.0, 9.25]], dtype="<f8")
    assert actual.dtype == expected.dtype
    assert actual.shape == expected.shape
    np.testing.assert_array_equal(actual, expected)
    assert actual.flags.c_contiguous


if __name__ == "__main__":
    if len(sys.argv) == 1:
        generate()
    elif len(sys.argv) == 3 and sys.argv[1] == "verify":
        verify_rust_output(Path(sys.argv[2]))
    else:
        raise SystemExit("usage: generate_fixtures.py [verify PATH]")
