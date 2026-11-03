"""Exercise the launch scripts without locking the desktop or changing user state."""
import concurrent.futures
import os
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest


ROOT = Path(__file__).resolve().parents[1]


class RotationTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        for name in ("savers.sh", "pick-saver.sh", "lock-bg.sh", "location.py"):
            shutil.copy2(ROOT / name, self.root / name)
        self.savers = subprocess.check_output(
            ["bash", "-c", '. "$1"; printf "%s\\n" "${SAVERS[@]}"',
             "bash", str(self.root / "savers.sh")], text=True,
        ).splitlines()
        for saver in self.savers:
            binary = {"pipes": "pipes/bin/pipes", "maze": "maze3d/bin/maze3d"}.get(
                saver, f"{saver}/{saver}")
            self.executable(binary, '#!/bin/sh\nprintf "%s\\n" "$@"\n')
        self.executable("build/windowtolayer/target/release/windowtolayer", "#!/bin/sh\n")
        wrapper = self.root / "build/swaylock-plugin/example_xwayland_wrapper.py"
        wrapper.parent.mkdir(parents=True)
        wrapper.touch()
        self.env = dict(os.environ, XDG_STATE_HOME=str(self.root / "state"),
                        XDG_CONFIG_HOME=str(self.root / "config"),
                        XDG_CACHE_HOME=str(self.root / "cache"))
        self.env.pop("SWAYLOCK_PLUGIN_OUTPUT_NAME", None)

    def executable(self, name, content):
        path = self.root / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
        path.chmod(0o755)

    def pick(self):
        return subprocess.check_output(
            [str(self.root / "pick-saver.sh")], env=self.env, text=True,
        ).strip()

    def assert_cycles(self, picks):
        n = len(self.savers)
        for start in range(0, len(picks), n):
            self.assertCountEqual(picks[start:start + n], self.savers)
            if start:
                self.assertNotEqual(picks[start - 1], picks[start])

    def test_sequential_cycles_and_new_state_directory(self):
        self.assert_cycles([self.pick() for _ in range(3 * len(self.savers))])

    def test_overlapping_picks_preserve_cycles(self):
        state = Path(self.env["XDG_STATE_HOME"])
        state.mkdir()
        # Slow the refill to reliably exercise simultaneous readers of an empty bag.
        shuf = shutil.which("shuf")
        self.executable("tools/shuf", f'#!/bin/sh\nsleep 0.05\nexec "{shuf}" "$@"\n')
        self.env["PATH"] = str(self.root / "tools") + os.pathsep + self.env["PATH"]
        with concurrent.futures.ThreadPoolExecutor(max_workers=12) as pool:
            list(pool.map(lambda _: self.pick(), range(3 * len(self.savers))))
        picks = [line.split()[-1] for line in (state / "lock-saver.log").read_text().splitlines()]
        self.assertEqual(len(picks), 3 * len(self.savers))
        self.assert_cycles(picks)

    def test_unavailable_savers_are_removed_from_saved_bag(self):
        state = Path(self.env["XDG_STATE_HOME"])
        state.mkdir()
        (state / "lock-saver-bag").write_text("retired soapfilm currents\n")
        self.assertEqual(self.pick(), "soapfilm")
        self.assertEqual(self.pick(), "currents")

    def launch(self, saver, output=None):
        env = self.env.copy()
        if output:
            env["SWAYLOCK_PLUGIN_OUTPUT_NAME"] = output
        return subprocess.check_output(
            [str(self.root / "lock-bg.sh"), saver, "12345", "1790000000"],
            env=env, text=True,
        ).splitlines()

    def test_independent_savers_have_repeatable_monitor_seeds(self):
        for saver in ("soapfilm", "aquarium"):
            with self.subTest(saver=saver):
                seeds = []
                for output in ("eDP-1", "DP-1", "HDMI-A-1"):
                    args = self.launch(saver, output)
                    self.assertIn("--layer", args)
                    for option in ("--layout", "--output", "--epoch"):
                        self.assertNotIn(option, args)
                    seed = args[args.index("--seed") + 1]
                    self.assertGreater(int(seed), 0)
                    self.assertEqual(args, self.launch(saver, output))
                    seeds.append(seed)
                self.assertEqual(len(set(seeds)), 3)

    def test_spanning_saver_keeps_shared_seed_and_epoch(self):
        args = self.launch("nightsky")
        self.assertEqual(args, ["--layer", "--seed", "12345", "--epoch", "1790000000"])

    def test_detected_coordinates_only_go_to_geographic_savers(self):
        (self.root / "location.py").write_text('print("55.95 -3.19")\n')
        for saver in ("nightsky", "atc"):
            args = self.launch(saver)
            self.assertEqual(args[-4:], ["--lat", "55.95", "--lon", "-3.19"])
        self.assertNotIn("--lat", self.launch("stir"))


if __name__ == "__main__":
    unittest.main()
