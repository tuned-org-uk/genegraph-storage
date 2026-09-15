"""Generate golden Zarr v3 fixtures for the genegraph-storage zzarr suite.

Run from the kernel root:
    uv run --with "zarr>=3.0" --with numpy python tests/fixtures/zarr/generate_fixtures.py

All values are deterministic (arange patterns, no RNG). Arrays use the
zzarr contract codecs: bytes (little endian) + zstd, default chunk key
encoding. One array is uncompressed to pin the codec variation.
"""

from pathlib import Path

import numpy as np
import zarr
from zarr.codecs import BytesCodec, ZstdCodec

HERE = Path(__file__).parent


def write_array(name: str, data: np.ndarray, chunks: tuple, compressors) -> None:
    path = HERE / name
    zarr.create_array(
        store=zarr.storage.LocalStore(path),
        data=data,
        chunks=chunks,
        compressors=compressors,
    )
    print(f"wrote {path} shape={data.shape} dtype={data.dtype}")


def main() -> None:
    # zarr-python 3: `compressors` takes bytes-to-bytes codecs only; the
    # array-to-bytes codec (bytes, little endian) is implicit.
    zz = (ZstdCodec(level=3),)
    zz_plain: tuple = ()

    # 1-D float32, three chunks (multi-chunk read conformance).
    write_array(
        "f32_1d.zarr",
        np.arange(10, dtype=np.float32) * np.float32(0.5),
        chunks=(4,),
        compressors=zz,
    )

    # 1-D int64, single chunk (indices/indptr analog).
    write_array(
        "i64_1d.zarr",
        np.arange(10, dtype=np.int64),
        chunks=(10,),
        compressors=zz,
    )

    # 2-D float64, four 2x2 chunks.
    write_array(
        "f64_2d.zarr",
        (np.arange(16, dtype=np.float64) / 10.0).reshape(4, 4),
        chunks=(2, 2),
        compressors=zz,
    )

    # 1-D float32, bytes codec only (no compression).
    write_array(
        "f32_1d_uncompressed.zarr",
        np.arange(6, dtype=np.float32),
        chunks=(3,),
        compressors=zz_plain,
    )

    # zzarr CSR meta sample (mirrors _persist_csr meta.json).
    meta = {"nitems": 50, "nfeatures": 4, "nclusters": 2, "csr_shape": [50, 4]}
    (HERE / "meta.json").write_text(
        __import__("json").dumps(meta, separators=(",", ":"))
    )
    print(f"wrote {HERE / 'meta.json'}")


if __name__ == "__main__":
    main()
