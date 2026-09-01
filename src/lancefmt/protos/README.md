# Vendored Lance v2.1 protobuf definitions

This directory pins the **Lance v2.1 format spec** that `genegraph-storage`'s
in-house implementation (#75, milestone M1+) targets.

## Provenance

- Source: [`lancedb/lance`](https://github.com/lancedb/lance) as shipped in
  the `lance` crate **v11.0.0** (`protos/` directory, crates.io package).
- License: **Apache-2.0** — each file carries
  `SPDX-License-Identifier: Apache-2.0` /
  `SPDX-FileCopyrightText: Copyright The Lance Authors` headers, preserved
  unmodified. See the root `NOTICE` file.
- Pinned: 2026-09-01, from the lance 11.0.0 crates.io release
  (`file.proto`, `file2.proto`, `table.proto`, `encodings_v2_0.proto`,
  `encodings_v2_1.proto`).

## Encoding policy (per #75, "never guess")

Our artifact schemas only require the **Flat** encoding with no compression
plus validity handling (none / all-set / bitmap). Files written by the
official crate that use any other encoding (bitpacking, FSST, dictionary,
full-zip, blob, general, ...) are **rejected with a typed error**
(`StorageError::UnsupportedFormat`), never guessed at.

## Regeneration

These files are vendored verbatim and must not be hand-edited. To update the
pin, copy the `protos/` directory from the target `lance` crate release,
update the pinned version above and in the root `NOTICE`, and re-run the
conformance suite (`cargo test --release lancefmt`).
