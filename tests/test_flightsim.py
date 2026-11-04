"""GPU regressions: xvfb-run -a python3 -B -m unittest discover -s tests -v."""
import os
from pathlib import Path
import shlex
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


@unittest.skipUnless(os.environ.get("DISPLAY") and
                     (ROOT / "flightsim/proto-layer-shell.h").exists(),
                     "flight render checks need an X display and make -C flightsim")
class FlightsimRenderTests(unittest.TestCase):
    def test_gpu_history_reprojection_resize_and_water(self):
        with tempfile.TemporaryDirectory() as tmp:
            binary = Path(tmp) / "flightsim-render-test"
            command = shlex.split(os.environ.get("CC", "cc")) + [
                "-O1", "-Wall", "-Wextra", "-o", str(binary),
                "tests/flightsim_render.c", "flightsim/proto-layer-shell.c",
                "flightsim/proto-xdg-shell.c", "-lGL", "-lX11", "-lEGL",
                "-lwayland-client", "-lwayland-egl", "-lm", "-lpthread",
            ]
            compiled = subprocess.run(command, cwd=ROOT, capture_output=True, text=True)
            self.assertEqual(compiled.returncode, 0, compiled.stdout + compiled.stderr)
            result = subprocess.run([str(binary)], cwd=ROOT, capture_output=True,
                                    text=True, timeout=120)
            self.assertEqual(result.returncode, 0, result.stdout + result.stderr)


if __name__ == "__main__":
    unittest.main()
