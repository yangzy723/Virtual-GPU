"""End-to-end test: torch.matmul through the vGPU shim stack.

Requires:
- vgpu_server running with VGPU_SERVER_SOCK set
- LD_PRELOAD pointing to vGPU shim libraries
- PyTorch installed with CUDA support
"""

import os
import socket
import subprocess
import sys
import time
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
BUILD_DIR = REPO_ROOT / "build"
SERVER_BIN = BUILD_DIR / "vgpu_server"


def _wait_for_socket(sock_path: Path, timeout_s: float = 10.0) -> bool:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        if sock_path.exists():
            try:
                with socket.socket(socket.AF_UNIX, socket.SOCK_STREAM):
                    return True
            except OSError:
                pass
        time.sleep(0.05)
    return sock_path.exists()


def _require_env() -> None:
    required = ["LD_PRELOAD", "LD_LIBRARY_PATH"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise unittest.SkipTest(
            f"torch.matmul test requires environment variables: {', '.join(missing)}")
    if not SERVER_BIN.exists():
        raise unittest.SkipTest("vgpu_server is not built")


MATMUL_SCRIPT = '''
import sys
import torch

print("torch", torch.__version__)
print("cuda available", torch.cuda.is_available())

if not torch.cuda.is_available():
    print("CUDA not available")
    sys.exit(1)

# Basic matmul
a = torch.randn(3, 4, device="cuda")
b = torch.randn(4, 5, device="cuda")
c = torch.matmul(a, b)
print("matmul result shape:", c.shape)
assert c.shape == (3, 5), f"Expected (3, 5), got {c.shape}"

# Verify against CPU reference
c_ref = torch.matmul(a.cpu(), b.cpu())
if torch.allclose(c.cpu(), c_ref, atol=1e-4):
    print("matmul correctness: PASS")
else:
    print("matmul correctness: FAIL")
    sys.exit(1)

# Larger matmul
a2 = torch.randn(128, 256, device="cuda")
b2 = torch.randn(256, 64, device="cuda")
c2 = torch.matmul(a2, b2)
print("large matmul shape:", c2.shape)
assert c2.shape == (128, 64)

# Batched matmul
a3 = torch.randn(8, 16, 32, device="cuda")
b3 = torch.randn(8, 32, 24, device="cuda")
c3 = torch.matmul(a3, b3)
print("batched matmul shape:", c3.shape)
assert c3.shape == (8, 16, 24)

print("ALL TESTS PASSED")
'''


class TestTorchMatmul(unittest.TestCase):

    def test_torch_matmul(self):
        """Verify torch.matmul works end-to-end through the vGPU shim stack."""
        _require_env()

        sock_path = Path("/tmp/vgpu_matmul_test.sock")
        if sock_path.exists():
            sock_path.unlink()

        server_env = {
            "PATH": os.environ.get("PATH", ""),
            "HOME": os.environ.get("HOME", ""),
            "USER": os.environ.get("USER", ""),
            "LOGNAME": os.environ.get("LOGNAME", ""),
            "SHELL": os.environ.get("SHELL", ""),
            "LANG": os.environ.get("LANG", "C"),
            "VGPU_SERVER_SOCK": str(sock_path),
        }

        server = subprocess.Popen(
            [str(SERVER_BIN)],
            cwd=str(REPO_ROOT),
            env=server_env,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )

        try:
            self.assertTrue(_wait_for_socket(sock_path), "vgpu_server socket did not appear")

            test_env = os.environ.copy()
            test_env["VGPU_SERVER_SOCK"] = str(sock_path)
            result = subprocess.run(
                [sys.executable, "-c", MATMUL_SCRIPT],
                cwd=str(REPO_ROOT),
                env=test_env,
                capture_output=True,
                text=True,
                timeout=120,
            )
        finally:
            server.terminate()
            try:
                server.wait(timeout=5)
            except subprocess.TimeoutExpired:
                server.kill()
                server.wait(timeout=5)
            if sock_path.exists():
                sock_path.unlink()

        combined = result.stdout + result.stderr
        self.assertEqual(result.returncode, 0,
                         f"torch.matmul test failed:\n{combined}")
        self.assertIn("ALL TESTS PASSED", combined)


if __name__ == "__main__":
    unittest.main()
