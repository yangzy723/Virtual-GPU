#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
SOCKET_PATH="${GPU_SCHEDULER_SOCKET:-$BUILD_DIR/vgpu_demo.sock}"
DAEMON_LOG="${DAEMON_LOG:-$BUILD_DIR/vgpu_daemon.log}"
CLIENT_LOG="${CLIENT_LOG:-$BUILD_DIR/vgpu_client.log}"
PYTHON_BIN="${PYTHON_BIN:-/home/yzy/miniconda3/envs/vgpu/bin/python}"

if [[ ! -x "$PYTHON_BIN" ]]; then
  PYTHON_BIN="python3"
fi

mkdir -p "$BUILD_DIR"
rm -f "$SOCKET_PATH" "$DAEMON_LOG" "$CLIENT_LOG"

cmake -S "$ROOT_DIR" -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="${CMAKE_BUILD_TYPE:-Release}" >/dev/null
cmake --build "$BUILD_DIR" -j >/dev/null

cleanup() {
  if [[ -n "${DAEMON_PID:-}" ]]; then
    kill "$DAEMON_PID" >/dev/null 2>&1 || true
    wait "$DAEMON_PID" 2>/dev/null || true
  fi
}
trap cleanup EXIT

GPU_SCHEDULER_SOCKET="$SOCKET_PATH" \
GPU_SCHEDULER_VERBOSE="${GPU_SCHEDULER_VERBOSE:-1}" \
GPU_SCHEDULER_MAX_KERNELS="${GPU_SCHEDULER_MAX_KERNELS:-2}" \
GPU_SCHEDULER_MAX_MEMCPY="${GPU_SCHEDULER_MAX_MEMCPY:-2}" \
GPU_SCHEDULER_POLL_US="${GPU_SCHEDULER_POLL_US:-100}" \
"$BUILD_DIR/gpu_scheduler" >"$DAEMON_LOG" 2>&1 &
DAEMON_PID=$!

GPU_SCHEDULER_SOCKET="$SOCKET_PATH" \
LD_PRELOAD="$BUILD_DIR/libcuda.so" \
LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" \
VGPU_TRACE="${VGPU_TRACE:-1}" \
"$PYTHON_BIN" - <<'PY' >"$CLIENT_LOG" 2>&1
import torch

print("cuda", torch.cuda.is_available())
h = torch.randn(128, 128)
d = h.to("cuda")
_ = d.cpu()
a = torch.randn(256, 256, device="cuda")
b = torch.randn(256, 256, device="cuda")
c = torch.matmul(a, b)
torch.cuda.synchronize()
print("matmul ok", tuple(c.shape), float(c[0, 0]))
PY

echo "== Client Summary =="
grep -E 'cuda |matmul ok|daemon request|daemon approved|daemon rejected|cuMemAllocAsync|cuMemAllocFromPoolAsync|cuLaunchKernelEx|cuLaunchKernel\(|cuMemcpy' "$CLIENT_LOG" | tail -n 100 || true

echo
echo "== Daemon Summary =="
grep -E 'registered client|pid=.*op=|APPROVED|REJECTED|processed request' "$DAEMON_LOG" | tail -n 80 || true

echo
echo "logs:"
echo "  client: $CLIENT_LOG"
echo "  daemon: $DAEMON_LOG"
