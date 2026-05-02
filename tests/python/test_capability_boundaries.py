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
PROBE_SCRIPT = Path(__file__).with_name("cublas_probe.py")


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


def _require_boundary_env() -> None:
    required = ["LD_PRELOAD", "LD_LIBRARY_PATH"]
    missing = [name for name in required if not os.environ.get(name)]
    if missing:
        raise unittest.SkipTest(
            f"capability boundary test requires environment variables: {', '.join(missing)}")
    if not SERVER_BIN.exists():
        raise unittest.SkipTest("vgpu_server is not built")


class CapabilityBoundaryTests(unittest.TestCase):

    def test_cublas_probe_known_incomplete_path(self):
        _require_boundary_env()

        sock_path = Path("/tmp/vgpu_capability_boundary.sock")
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

            probe_env = os.environ.copy()
            probe_env["VGPU_SERVER_SOCK"] = str(sock_path)
            result = subprocess.run(
                [sys.executable, str(PROBE_SCRIPT)],
                cwd=str(REPO_ROOT),
                env=probe_env,
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
        self.assertEqual(result.returncode, 1, combined)
        self.assertIn("cuda available True", combined)
        self.assertNotIn("cublasDestroy_v2", combined)

        # The current boundary is that the PyTorch + cuBLAS path still does not
        # complete successfully under the shim stack. Depending on the exact
        # interception state, the failure can happen during torch.cuda.init()
        # or later at cublasCreate_v2. Keep both failure shapes explicit here.
        saw_init_failure = "cudaErrorNotSupported" in combined
        saw_cublas_failure = "cuda initialized" in combined and "cublasCreate_v2 rc= 1" in combined
        self.assertTrue(
            saw_init_failure or saw_cublas_failure,
            combined,
        )


if __name__ == "__main__":
    unittest.main()