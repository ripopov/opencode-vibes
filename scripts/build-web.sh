#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" >/dev/null 2>&1 && pwd)"
REPO_ROOT="$(cd -- "${SCRIPT_DIR}/.." >/dev/null 2>&1 && pwd)"
CRATE_NAME="cornell_egui_pathtracer"
DIST_DIR="${REPO_ROOT}/dist"
WASM_PATH="${REPO_ROOT}/target/wasm32-unknown-unknown/release/${CRATE_NAME}.wasm"

cargo build --manifest-path "${REPO_ROOT}/Cargo.toml" --target wasm32-unknown-unknown --release

rm -rf "${DIST_DIR}"
mkdir -p "${DIST_DIR}/pkg"
cp "${REPO_ROOT}/web/index.html" "${DIST_DIR}/index.html"
touch "${DIST_DIR}/.nojekyll"

wasm-bindgen \
  --target web \
  --out-dir "${DIST_DIR}/pkg" \
  "${WASM_PATH}"
