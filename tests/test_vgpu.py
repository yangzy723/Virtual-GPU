"""Tests for the GPU scheduler daemon and libcuda.so shim.

Tests:
1. Daemon starts and accepts connections
2. Basic CUDA ops work with daemon (scheduling)
3. Graceful degradation: works without daemon
4. PyTorch matmul through shim stack
5. Multiple clients concurrent operation
"""

import os
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
PYTHON = "/home/yzy/miniconda3/envs/vgpu/bin/python"


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
import torch

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
            proc.terminate()
            proc.wait(timeout=5)
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
            proc.terminate()
            proc.wait(timeout=5)
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
import torch, sys
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
            proc.terminate()
            proc.wait(timeout=5)
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
import torch

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
            proc.terminate()
            _, daemon_stderr = proc.communicate(timeout=5)
            if isinstance(daemon_stderr, bytes):
                daemon_stderr = daemon_stderr.decode("utf-8", errors="replace")
            if sock.exists():
                sock.unlink()

        combined = result.stdout + result.stderr
        if "SKIP:" in combined:
            self.skipTest(combined.split("SKIP:")[1].strip())

        self.assertEqual(result.returncode, 0, f"Kernel/memcpy path test failed:\n{combined}")
        self.assertIn("KERNEL_MEMCPY_DAEMON_TEST_PASSED", combined)
        self.assertIn("op=KERNEL_REQUEST", daemon_stderr)
        self.assertIn("op=MEMCPY_REQUEST", daemon_stderr)


if __name__ == "__main__":
    unittest.main()
