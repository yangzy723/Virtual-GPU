#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
BUILD_DIR="${BUILD_DIR:-$ROOT_DIR/build}"
SOCKET_PATH="${GPU_SCHEDULER_SOCKET:-$BUILD_DIR/vgpu_demo.sock}"
DAEMON_LOG="${DAEMON_LOG:-$BUILD_DIR/vgpu_daemon.log}"
CLIENT_LOG="${CLIENT_LOG:-$BUILD_DIR/vgpu_client.log}"
PYTHON_BIN="${PYTHON_BIN:-${VGPU_TEST_PYTHON:-}}"

if [[ -z "$PYTHON_BIN" ]]; then
  for candidate in \
    "/home/yzy/miniconda3/envs/agent/bin/python" \
    "/home/yzy/miniconda3/envs/cogpu/bin/python" \
    "python3"; do
    if command -v "$candidate" >/dev/null 2>&1 || [[ -x "$candidate" ]]; then
      PYTHON_BIN="$candidate"
      break
    fi
  done
fi

if [[ -z "$PYTHON_BIN" ]]; then
  echo "error: no python interpreter found; set PYTHON_BIN=/path/to/python" >&2
  exit 1
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
GPU_SCHEDULER_COMPLETION_POLL_US="${GPU_SCHEDULER_COMPLETION_POLL_US:-100}" \
"$BUILD_DIR/gpu_scheduler" >"$DAEMON_LOG" 2>&1 &
DAEMON_PID=$!

for _ in {1..200}; do
  if [[ -S "$SOCKET_PATH" ]]; then
    break
  fi
  sleep 0.05
done

if [[ ! -S "$SOCKET_PATH" ]]; then
  echo "error: daemon socket did not appear: $SOCKET_PATH" >&2
  tail -n 80 "$DAEMON_LOG" >&2 || true
  exit 1
fi

GPU_SCHEDULER_SOCKET="$SOCKET_PATH" \
LD_PRELOAD="$BUILD_DIR/libcuda.so" \
LD_LIBRARY_PATH="$BUILD_DIR:${LD_LIBRARY_PATH:-}" \
VGPU_TRACE="${VGPU_TRACE:-1}" \
"$PYTHON_BIN" - <<'PY' >"$CLIENT_LOG" 2>&1
import sys
import time

import torch

print("cuda", torch.cuda.is_available())
if not torch.cuda.is_available():
  print("SKIP: CUDA not available")
  sys.exit(0)

h = torch.randn(128, 128)
d = h.to("cuda")
_ = d.cpu()
a = torch.randn(256, 256, device="cuda")
b = torch.randn(256, 256, device="cuda")
c = torch.matmul(a, b)
torch.cuda.synchronize()
time.sleep(0.2)
print("matmul ok", tuple(c.shape), float(c[0, 0]))
PY

if grep -q 'SKIP: CUDA not available' "$CLIENT_LOG"; then
  cat "$CLIENT_LOG"
  echo "matmul intercept demo skipped: CUDA is not available" >&2
  exit 0
fi

required_client_patterns=(
  'matmul ok'
  'daemon request op=3'
  'daemon approved op=3'
)

for pattern in "${required_client_patterns[@]}"; do
  if ! grep -q "$pattern" "$CLIENT_LOG"; then
    echo "error: missing client evidence: $pattern" >&2
    tail -n 120 "$CLIENT_LOG" >&2 || true
    exit 1
  fi
done

required_daemon_patterns=(
  'op=KERNEL_REQUEST'
  'op=KERNEL_COMPLETE'
)

for pattern in "${required_daemon_patterns[@]}"; do
  if ! grep -q "$pattern" "$DAEMON_LOG"; then
    echo "error: missing daemon evidence: $pattern" >&2
    tail -n 120 "$DAEMON_LOG" >&2 || true
    exit 1
  fi
done

echo "== Client Summary =="
grep -E 'cuda |matmul ok|daemon request|daemon approved|daemon rejected|cuMemAllocAsync|cuMemAllocFromPoolAsync|cuLaunchKernelEx|cuLaunchKernel\(|cuMemcpy' "$CLIENT_LOG" | tail -n 140 || true

echo
echo "== Daemon Summary =="
grep -E 'registered client|pid=.*op=|APPROVED|REJECTED|processed request' "$DAEMON_LOG" | tail -n 80 || true

echo
echo "matmul intercept verified: torch.matmul reached scheduled CUDA Driver kernel path"

echo
echo "logs:"
echo "  client: $CLIENT_LOG"
echo "  daemon: $DAEMON_LOG"
