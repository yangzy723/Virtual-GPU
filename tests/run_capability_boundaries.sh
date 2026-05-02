#!/usr/bin/env bash
set -euo pipefail

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
build_dir="$repo_root/build"
python_bin="${VGPU_PYTHON:-${PYTHON:-python3}}"

require_file() {
    if [[ ! -e "$1" ]]; then
        echo "missing required file: $1" >&2
        exit 1
    fi
}

require_file "$build_dir/vgpu_driver_capability_boundary_test"
require_file "$build_dir/vgpu_runtime_capability_boundary_test"
require_file "$build_dir/vgpu_runtime_not_supported_semantics_test"
require_file "$build_dir/vgpu_server"
require_file "$build_dir/libvgpu_preload_init.so"
require_file "$build_dir/libcuda.so"
require_file "$build_dir/libcudart.so"

echo "[capability-boundaries] driver"
"$build_dir/vgpu_driver_capability_boundary_test"

echo "[capability-boundaries] runtime-success-and-boundary"
"$build_dir/vgpu_runtime_capability_boundary_test"

echo "[capability-boundaries] runtime-not-supported"
"$build_dir/vgpu_runtime_not_supported_semantics_test"

echo "[capability-boundaries] python-known-incomplete"
export LD_PRELOAD="$build_dir/libvgpu_preload_init.so:$build_dir/libcuda.so:$build_dir/libcudart.so"
export LD_LIBRARY_PATH="$build_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
cd "$repo_root"
"$python_bin" -m unittest tests.python.test_capability_boundaries -v