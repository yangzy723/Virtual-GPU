"""Tests for the GPU scheduler daemon and libcuda.so shim.

Tests:
1. Daemon starts and accepts connections
2. Dynamic symbol routing covers scheduled Driver API paths
3. Fail-open warning is visible when daemon is unavailable
4. PyTorch matmul and Driver memset paths reach the daemon
5. Multiple clients run concurrently
"""

import os
import ctypes
import socket
import subprocess
import sys
import time
import unittest
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
BUILD_DIR = REPO_ROOT / "build"
DAEMON_BIN = BUILD_DIR / "gpu_scheduler"
SHIM_LIB = BUILD_DIR / "libcuda.so"
PYTHON = os.environ.get("VGPU_TEST_PYTHON", sys.executable)

SCHEDULED_SYMBOLS = [
    "cuMemAlloc", "cuMemAlloc_v2",
    "cuMemAllocAsync", "cuMemAllocFromPoolAsync",
    "cuMemFree", "cuMemFree_v2", "cuMemFreeAsync",
    "cuMemcpy", "cuMemcpyAsync",
    "cuMemcpyPeer", "cuMemcpyPeerAsync",
    "cuMemcpyHtoDAsync_v2", "cuMemcpyDtoHAsync_v2", "cuMemcpyDtoDAsync_v2",
    "cuMemcpy2D", "cuMemcpy2D_v2", "cuMemcpy2DUnaligned", "cuMemcpy2DUnaligned_v2",
    "cuMemcpy2DAsync", "cuMemcpy2DAsync_v2",
    "cuMemcpy3D", "cuMemcpy3D_v2", "cuMemcpy3DAsync", "cuMemcpy3DAsync_v2",
    "cuMemcpy3DPeer", "cuMemcpy3DPeerAsync", "cuMemcpyBatchAsync_v2",
    "cuMemsetD8", "cuMemsetD8_v2", "cuMemsetD16", "cuMemsetD16_v2",
    "cuMemsetD32", "cuMemsetD32_v2",
    "cuMemsetD2D8", "cuMemsetD2D8_v2", "cuMemsetD2D16", "cuMemsetD2D16_v2",
    "cuMemsetD2D32", "cuMemsetD2D32_v2",
    "cuMemsetD8Async", "cuMemsetD16Async", "cuMemsetD32Async",
    "cuMemsetD2D8Async", "cuMemsetD2D16Async", "cuMemsetD2D32Async",
]

LEGACY_STRUCT_MEMCPY_SYMBOLS = [
    "cuMemcpy2D", "cuMemcpy2DUnaligned", "cuMemcpy2DAsync",
    "cuMemcpy3D", "cuMemcpy3DAsync",
]

LEGACY_V1_MEMSET_SYMBOLS = [
    "cuMemsetD8", "cuMemsetD16", "cuMemsetD32",
    "cuMemsetD2D8", "cuMemsetD2D16", "cuMemsetD2D32",
]


def _wait_for_socket(sock_path: Path, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if sock_path.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                    s.connect(str(sock_path))
                    return True
            except OSError:
                pass
        time.sleep(0.05)
    return sock_path.exists()


def _require_env() -> None:
    if not DAEMON_BIN.exists():
        raise unittest.SkipTest("gpu_scheduler is not built")
    if not SHIM_LIB.exists():
        raise unittest.SkipTest("libcuda.so shim is not built")


def _start_daemon(sock_path: Path, extra_env: dict | None = None) -> subprocess.Popen:
    env = {
        "PATH": os.environ.get("PATH", ""),
        "HOME": os.environ.get("HOME", ""),
        "GPU_SCHEDULER_SOCKET": str(sock_path),
        "GPU_SCHEDULER_MAX_KERNELS": "4",
    }
    if extra_env:
        env.update(extra_env)
    proc = subprocess.Popen(
        [str(DAEMON_BIN)],
        cwd=str(REPO_ROOT),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
    )
    assert _wait_for_socket(sock_path), "daemon socket did not appear"
    return proc


def _stop_daemon(proc: subprocess.Popen) -> tuple[str, str]:
    proc.terminate()
    try:
        stdout, stderr = proc.communicate(timeout=5)
    except subprocess.TimeoutExpired:
        proc.kill()
        stdout, stderr = proc.communicate(timeout=5)

    def _decode(data: bytes | str | None) -> str:
        if data is None:
            return ""
        if isinstance(data, bytes):
            return data.decode("utf-8", errors="replace")
        return data

    return _decode(stdout), _decode(stderr)


def _run_with_shim(
    script: str,
    sock_path: Path,
    timeout: float = 60,
    extra_env: dict | None = None,
) -> subprocess.CompletedProcess:
    env = os.environ.copy()
    env["GPU_SCHEDULER_SOCKET"] = str(sock_path)
    env["LD_PRELOAD"] = str(SHIM_LIB)
    env["LD_LIBRARY_PATH"] = f"{BUILD_DIR}:{env.get('LD_LIBRARY_PATH', '')}"
    if extra_env:
        env.update(extra_env)
    return subprocess.run(
        [PYTHON, "-c", script],
        cwd=str(REPO_ROOT),
        env=env,
        capture_output=True,
        text=True,
        timeout=timeout,
    )


TORCH_SCRIPT = """
import sys
try:
    import torch
except Exception as exc:
    print(f"SKIP: torch import failed: {exc}")
    sys.exit(0)

print(f"torch {torch.__version__}")
print(f"cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("SKIP: CUDA not available")
    sys.exit(0)

a = torch.randn(64, 64, device="cuda")
b = torch.randn(64, 64, device="cuda")
c = torch.matmul(a, b)
print(f"matmul shape: {c.shape}")
assert c.shape == (64, 64)

c_ref = torch.matmul(a.cpu(), b.cpu())
assert torch.allclose(c.cpu(), c_ref, atol=1e-4), "matmul mismatch"
print("TORCH_TEST_PASSED")
"""


class TestDaemonBasic(unittest.TestCase):

    def test_cugetprocaddress_routes_scheduled_symbols(self):
        """Scheduled symbols resolve to shim implementations for modern ABI requests."""
        _require_env()
        shim = ctypes.CDLL(str(SHIM_LIB), mode=ctypes.RTLD_GLOBAL)
        get_proc = shim.cuGetProcAddress
        get_proc.argtypes = [
            ctypes.c_char_p,
            ctypes.POINTER(ctypes.c_void_p),
            ctypes.c_int,
            ctypes.c_ulonglong,
        ]
        get_proc.restype = ctypes.c_int

        for symbol in SCHEDULED_SYMBOLS:
            with self.subTest(symbol=symbol):
                pfn = ctypes.c_void_p()
                rc = get_proc(symbol.encode("ascii"), ctypes.byref(pfn), 7000, 0)
                shim_addr = ctypes.cast(getattr(shim, symbol), ctypes.c_void_p).value
                self.assertEqual(rc, 0)
                self.assertEqual(pfn.value, shim_addr)

        for symbol in LEGACY_STRUCT_MEMCPY_SYMBOLS:
            with self.subTest(symbol=f"legacy-{symbol}"):
                pfn = ctypes.c_void_p()
                rc = get_proc(symbol.encode("ascii"), ctypes.byref(pfn), 3020, 0)
                shim_addr = ctypes.cast(getattr(shim, symbol), ctypes.c_void_p).value
                if rc == 0:
                    self.assertNotEqual(pfn.value, shim_addr)

        for symbol in LEGACY_V1_MEMSET_SYMBOLS:
            with self.subTest(symbol=f"legacy-{symbol}"):
                pfn = ctypes.c_void_p()
                rc = get_proc(symbol.encode("ascii"), ctypes.byref(pfn), 2000, 0)
                shim_addr = ctypes.cast(getattr(shim, symbol), ctypes.c_void_p).value
                if rc == 0:
                    self.assertNotEqual(pfn.value, shim_addr)

    def test_daemon_start_stop(self):
        """Daemon starts, accepts connection, and shuts down cleanly."""
        _require_env()
        sock = Path("/tmp/vgpu_test_basic.sock")
        if sock.exists():
            sock.unlink()

        proc = _start_daemon(sock)
        try:
            with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM) as s:
                s.connect(str(sock))
        finally:
            _stop_daemon(proc)
            if sock.exists():
                sock.unlink()

    def test_torch_matmul_with_daemon(self):
        """torch.matmul works through the shim with daemon scheduling."""
        _require_env()
        sock = Path("/tmp/vgpu_test_torch.sock")
        if sock.exists():
            sock.unlink()

        proc = _start_daemon(sock)
        try:
            result = _run_with_shim(TORCH_SCRIPT, sock, timeout=120)
        finally:
            _stop_daemon(proc)
            if sock.exists():
                sock.unlink()

        combined = result.stdout + result.stderr
        if "SKIP:" in combined:
            self.skipTest(combined.split("SKIP:")[1].strip())
        self.assertEqual(result.returncode, 0, f"Torch test failed:\n{combined}")
        self.assertIn("TORCH_TEST_PASSED", combined)

    def test_torch_matmul_without_daemon(self):
        """torch.matmul works via graceful degradation (no daemon)."""
        _require_env()
        sock = Path("/tmp/vgpu_test_nodeamon.sock")
        if sock.exists():
            sock.unlink()

        result = _run_with_shim(TORCH_SCRIPT, sock, timeout=120)
        combined = result.stdout + result.stderr
        if "SKIP:" in combined:
            self.skipTest(combined.split("SKIP:")[1].strip())
        self.assertEqual(result.returncode, 0, f"Graceful degradation failed:\n{combined}")
        self.assertIn("TORCH_TEST_PASSED", combined)

    def test_fail_open_without_daemon_prints_warning(self):
        """Fail-open passthrough must be visible when the daemon is unavailable."""
        _require_env()
        sock = Path("/tmp/vgpu_test_fail_open_warning.sock")
        if sock.exists():
            sock.unlink()

        script = """
import ctypes

shim = ctypes.CDLL("build/libcuda.so")
device_ptr = ctypes.c_ulonglong()
shim.cuMemAlloc(ctypes.byref(device_ptr), ctypes.c_size_t(1024))
print("FAIL_OPEN_WARNING_TEST_DONE")
"""
        result = _run_with_shim(
            script,
            sock,
            timeout=30,
            extra_env={"VGPU_TRACE": "0"},
        )

        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0, f"Fail-open warning test failed:\n{combined}")
        self.assertIn("FAIL_OPEN_WARNING_TEST_DONE", combined)
        self.assertIn("[vGPU warning] scheduler unavailable, fail-open passthrough", result.stderr)
        self.assertIn("cannot connect to daemon", result.stderr)

    def test_multi_client_concurrent(self):
        """Daemon serves multiple clients concurrently without deadlock."""
        _require_env()
        sock = Path("/tmp/vgpu_test_multi.sock")
        if sock.exists():
            sock.unlink()

        proc = _start_daemon(sock)
        try:
            num_clients = 3
            with ThreadPoolExecutor(max_workers=num_clients) as pool:
                futures = {}
                for i in range(num_clients):
                    script = f"""
import sys
try:
    import torch
except Exception as exc:
    print(f"SKIP: torch import failed: {{exc}}")
    sys.exit(0)
client_id = {i}
a = torch.randn(32, 32, device='cuda')
b = torch.randn(32, 32, device='cuda')
c = torch.matmul(a, b)
assert c.shape == (32, 32), f"client {{client_id}}: wrong shape {{c.shape}}"
c_ref = torch.matmul(a.cpu(), b.cpu())
assert torch.allclose(c.cpu(), c_ref, atol=1e-4), f"client {{client_id}}: matmul mismatch"
print(f"CLIENT_{{client_id}}_PASSED")
"""
                    futures[pool.submit(_run_with_shim, script, sock, timeout=120)] = i

                errors = []
                for future in as_completed(futures):
                    client_id = futures[future]
                    result = future.result()
                    combined = result.stdout + result.stderr
                    if "SKIP:" in combined:
                        continue
                    if result.returncode != 0:
                        errors.append(f"Client {client_id} failed (rc={result.returncode}):\n{combined}")
                    elif f"CLIENT_{client_id}_PASSED" not in combined:
                        errors.append(f"Client {client_id}: missing success marker in output:\n{combined}")

                self.assertEqual(errors, [], "Multi-client test had failures:\n" + "\n---\n".join(errors))
        finally:
            _stop_daemon(proc)
            if sock.exists():
                sock.unlink()

    def test_kernel_launch_and_memcpy_paths_reach_daemon(self):
        """With controls enabled, daemon log should include KERNEL_REQUEST and MEMCPY_REQUEST."""
        _require_env()
        sock = Path("/tmp/vgpu_test_kernel_memcpy.sock")
        if sock.exists():
            sock.unlink()

        proc = _start_daemon(
            sock,
            extra_env={
                "GPU_SCHEDULER_VERBOSE": "1",
                "GPU_SCHEDULER_MAX_KERNELS": "2",
                "GPU_SCHEDULER_MAX_MEMCPY": "2",
            },
        )

        script = """
import sys
import time
try:
    import torch
except Exception as exc:
    print(f"SKIP: torch import failed: {exc}")
    sys.exit(0)

print(f"cuda available: {torch.cuda.is_available()}")
if not torch.cuda.is_available():
    print("SKIP: CUDA not available")
    sys.exit(0)

# Host->Device / Device->Host copy path
h = torch.randn(128, 128)
d = h.to("cuda")
_ = d.cpu()

# Kernel launch path
a = torch.randn(128, 128, device="cuda")
b = torch.randn(128, 128, device="cuda")
c = torch.matmul(a, b)
torch.cuda.synchronize()
time.sleep(0.2)
print("KERNEL_MEMCPY_DAEMON_TEST_PASSED", tuple(c.shape))
"""

        try:
            result = _run_with_shim(
                script,
                sock,
                timeout=120,
                extra_env={
                    "VGPU_TRACE": "1",
                },
            )
        finally:
            _, daemon_stderr = _stop_daemon(proc)
            if sock.exists():
                sock.unlink()

        combined = result.stdout + result.stderr
        if "SKIP:" in combined:
            self.skipTest(combined.split("SKIP:")[1].strip())

        self.assertEqual(result.returncode, 0, f"Kernel/memcpy path test failed:\n{combined}")
        self.assertIn("KERNEL_MEMCPY_DAEMON_TEST_PASSED", combined)
        self.assertIn("op=KERNEL_REQUEST", daemon_stderr)
        self.assertIn("op=KERNEL_COMPLETE", daemon_stderr)
        self.assertIn("op=MEMCPY_REQUEST", daemon_stderr)
        self.assertIn("op=MEMCPY_COMPLETE", daemon_stderr)

    def test_memset_path_reaches_daemon(self):
        """cuMemsetD8 should be routed through the daemon and write the expected bytes."""
        _require_env()
        sock = Path("/tmp/vgpu_test_memset.sock")
        if sock.exists():
            sock.unlink()

        proc = _start_daemon(
            sock,
            extra_env={
                "GPU_SCHEDULER_VERBOSE": "1",
                "GPU_SCHEDULER_MAX_MEMCPY": "4",
            },
        )

        script = """
import ctypes
import sys

CUDA_SUCCESS = 0

shim = ctypes.CDLL("build/libcuda.so")

shim.cuInit.argtypes = [ctypes.c_uint]
shim.cuInit.restype = ctypes.c_int
shim.cuDeviceGetCount.argtypes = [ctypes.POINTER(ctypes.c_int)]
shim.cuDeviceGetCount.restype = ctypes.c_int
shim.cuDeviceGet.argtypes = [ctypes.POINTER(ctypes.c_int), ctypes.c_int]
shim.cuDeviceGet.restype = ctypes.c_int
shim.cuCtxCreate.argtypes = [ctypes.POINTER(ctypes.c_void_p), ctypes.c_uint, ctypes.c_int]
shim.cuCtxCreate.restype = ctypes.c_int
shim.cuCtxDestroy.argtypes = [ctypes.c_void_p]
shim.cuCtxDestroy.restype = ctypes.c_int
shim.cuMemAlloc.argtypes = [ctypes.POINTER(ctypes.c_ulonglong), ctypes.c_size_t]
shim.cuMemAlloc.restype = ctypes.c_int
shim.cuMemFree.argtypes = [ctypes.c_ulonglong]
shim.cuMemFree.restype = ctypes.c_int
shim.cuMemsetD8.argtypes = [ctypes.c_ulonglong, ctypes.c_ubyte, ctypes.c_size_t]
shim.cuMemsetD8.restype = ctypes.c_int
shim.cuMemcpyDtoH.argtypes = [ctypes.c_void_p, ctypes.c_ulonglong, ctypes.c_size_t]
shim.cuMemcpyDtoH.restype = ctypes.c_int
shim.cuCtxSynchronize.argtypes = []
shim.cuCtxSynchronize.restype = ctypes.c_int

def check(rc, label):
    if rc != CUDA_SUCCESS:
        raise RuntimeError(f"{label} failed rc={rc}")

rc = shim.cuInit(0)
if rc != CUDA_SUCCESS:
    print(f"SKIP: cuInit failed rc={rc}")
    sys.exit(0)

count = ctypes.c_int()
rc = shim.cuDeviceGetCount(ctypes.byref(count))
if rc != CUDA_SUCCESS or count.value <= 0:
    print(f"SKIP: no CUDA device rc={rc} count={count.value}")
    sys.exit(0)

device = ctypes.c_int()
check(shim.cuDeviceGet(ctypes.byref(device), 0), "cuDeviceGet")

ctx = ctypes.c_void_p()
check(shim.cuCtxCreate(ctypes.byref(ctx), 0, device.value), "cuCtxCreate")

ptr = ctypes.c_ulonglong()
try:
    check(shim.cuMemAlloc(ctypes.byref(ptr), 64), "cuMemAlloc")
    check(shim.cuMemsetD8(ptr.value, 0x5A, 64), "cuMemsetD8")
    check(shim.cuCtxSynchronize(), "cuCtxSynchronize")

    out = (ctypes.c_ubyte * 64)()
    check(shim.cuMemcpyDtoH(out, ptr.value, 64), "cuMemcpyDtoH")
    data = bytes(out)
    assert data == bytes([0x5A]) * 64, data[:8]
    print("MEMSET_DAEMON_TEST_PASSED")
finally:
    if ptr.value:
        shim.cuMemFree(ptr.value)
    if ctx.value:
        shim.cuCtxDestroy(ctx)
"""

        try:
            result = _run_with_shim(
                script,
                sock,
                timeout=120,
                extra_env={"VGPU_TRACE": "1"},
            )
        finally:
            _, daemon_stderr = _stop_daemon(proc)
            if sock.exists():
                sock.unlink()

        combined = result.stdout + result.stderr
        if "SKIP:" in combined:
            self.skipTest(combined.split("SKIP:")[1].strip())

        self.assertEqual(result.returncode, 0, f"Memset path test failed:\n{combined}")
        self.assertIn("MEMSET_DAEMON_TEST_PASSED", combined)
        self.assertIn("cuMemsetD8", combined)
        self.assertIn("op=MEMCPY_REQUEST", daemon_stderr)
        self.assertIn("op=MEMCPY_COMPLETE", daemon_stderr)


if __name__ == "__main__":
    unittest.main()
