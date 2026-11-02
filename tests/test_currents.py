"""Run with xvfb-run -a python3 -B -m unittest discover -s tests -v."""
import os
from pathlib import Path
import subprocess
import tempfile
import unittest

try:
    from PIL import Image, ImageChops, ImageStat
except ImportError:
    Image = None


BINARY = Path(__file__).resolve().parents[1] / "currents/currents"


@unittest.skipUnless(os.environ.get("DISPLAY") and Image and BINARY.exists(),
                     "render checks need a display, Pillow and a built Currents binary")
class CurrentsRenderTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.counter = 0

    def render(self, at=14, seed=42, output=None):
        self.counter += 1
        path = Path(self.tmp.name) / f"frame-{self.counter}.ppm"
        args = [str(BINARY), "--shot", str(path), "--size", "320x240",
                "--at", str(at), "--seed", str(seed)]
        if output is not None:
            args += ["--layout", "0,0,320,240;160,0,320,240", "--output", str(output)]
        subprocess.run(args, check=True, capture_output=True)
        with Image.open(path) as img:
            return img.copy()

    def test_same_time_and_seed_reproduce_identical_frame(self):
        a, b = self.render(), self.render()
        self.assertIsNone(ImageChops.difference(a, b).getbbox())
        self.assertGreater(max(ImageStat.Stat(a).stddev), 20)

    def test_animation_and_seed_change_the_picture(self):
        a = self.render()
        for b in (self.render(at=15), self.render(seed=12345)):
            self.assertGreater(sum(ImageStat.Stat(ImageChops.difference(a, b)).mean), 3)

    def test_overlapping_outputs_draw_the_same_world_pixels(self):
        left, right = self.render(output=0), self.render(output=1)
        a = left.crop((160, 0, 320, 240))
        b = right.crop((0, 0, 160, 240))
        error = ImageStat.Stat(ImageChops.difference(a, b)).mean
        self.assertLess(max(error), 0.5, error)


if __name__ == "__main__":
    unittest.main()
