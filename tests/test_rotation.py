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
        (state / "lock-saver-bag").write_text("retired aquarium gauges algorithms soapfilm currents\n")
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
        # Temporarily promote Aquarium in this fixture to keep its per-output
        # behavior covered while it is benched in the real rotation.
        with (self.root / "savers.sh").open("a") as savers:
            savers.write("\nSAVERS+=(aquarium)\n")
        self.executable("aquarium/aquarium", '#!/bin/sh\nprintf "%s\\n" "$@"\n')
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

    def test_render_scale_comes_from_the_host_config(self):
        conf = self.root / "config/wallpapers/render-scale"
        conf.parent.mkdir(parents=True)
        conf.write_text("# heavy here\nstir 0.5\n# voronoi 0.1\n* .75\nnightsky 1\natc banana\n")
        self.assertEqual(self.launch("stir")[-2:], ["--render-scale", "0.5"])
        self.assertEqual(self.launch("voronoi")[-2:], ["--render-scale", ".75"])
        self.assertNotIn("--render-scale", self.launch("nightsky"))   # 1 = full resolution
        self.assertNotIn("--render-scale", self.launch("atc"))        # malformed: ignored
        self.assertNotIn("--render-scale", self.launch("pipes"))      # X11 hacks: not ours to scale

    def test_no_render_scale_without_a_config(self):
        self.assertNotIn("--render-scale", self.launch("stir"))

    def skip_saver(self, *args, cur=None):
        """Run lock-next.sh with a stand-in locked session; returns lock-bg.cur."""
        shutil.copy2(ROOT / "lock-next.sh", self.root / "lock-next.sh")
        run = self.root / "run"
        run.mkdir(exist_ok=True)
        if cur is not None:
            (run / "lock-bg.cur").write_text(cur)
        for tool in ("pgrep", "pkill"):   # "locked", and nothing to kill
            self.executable(f"stubs/{tool}", "#!/bin/sh\nexit 0\n")
        env = dict(self.env, XDG_RUNTIME_DIR=str(run),
                   PATH=f"{self.root / 'stubs'}:{os.environ['PATH']}")
        subprocess.check_call([str(self.root / "lock-next.sh"), *args], env=env)
        return (run / "lock-bg.cur").read_text().split()

    def test_reroll_keeps_the_saver_with_a_new_seed_and_epoch(self):
        saver, seed, epoch = self.skip_saver("--reroll", cur="tiedye 12345 1790000000.5\n")
        self.assertEqual(saver, "tiedye")
        self.assertNotEqual(seed, "12345")
        self.assertNotEqual(epoch, "1790000000.5")

    def test_reroll_with_nothing_running_picks_a_saver(self):
        saver, _, _ = self.skip_saver("--reroll")
        self.assertIn(saver, self.savers)

    def test_next_moves_to_another_pick(self):
        picks = {self.skip_saver(cur="tiedye 1 1\n")[0] for _ in range(len(self.savers))}
        self.assertTrue(picks <= set(self.savers))
        self.assertGreater(len(picks), 1)


if __name__ == "__main__":
    unittest.main()
