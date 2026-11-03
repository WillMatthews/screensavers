import importlib.util
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch, MagicMock

spec = importlib.util.spec_from_file_location("location", Path(__file__).resolve().parents[1] / "location.py")
location = importlib.util.module_from_spec(spec)
spec.loader.exec_module(location)


class LocationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.config = Path(self.temp.name) / "config"
        self.cache = Path(self.temp.name) / "cache"
        self.config.mkdir()
        (self.config / "location").write_text("51.517501 -0.140344\n")
        self.network = patch.object(location.urllib.request, "urlopen")
        self.urlopen = self.network.start()
        self.addCleanup(self.network.stop)
        response = MagicMock()
        response.read.return_value = b'{"latitude": 55.95, "longitude": -3.19}'
        self.urlopen.return_value.__enter__.return_value = response

    def auto(self):
        (self.config / "location-mode").write_text("auto\n")

    def resolve(self):
        return location.resolve(self.config, self.cache)

    def test_default_never_uses_network(self):
        self.assertIsNone(self.resolve())
        self.urlopen.assert_not_called()

    def test_auto_caches_without_overwriting_manual(self):
        self.auto()
        self.assertEqual(self.resolve(), (55.95, -3.19))
        self.assertEqual(self.resolve(), (55.95, -3.19))
        self.urlopen.assert_called_once()
        self.assertEqual((self.config / "location").read_text(), "51.517501 -0.140344\n")

    def test_failure_falls_back_and_backs_off(self):
        self.auto()
        self.urlopen.side_effect = OSError("offline")
        self.assertIsNone(self.resolve())
        self.assertIsNone(self.resolve())
        self.urlopen.assert_called_once()

    def test_disable_ignores_cached_automatic_location(self):
        self.auto()
        self.resolve()
        (self.config / "location-mode").write_text("manual\n")
        self.assertIsNone(self.resolve())
        self.urlopen.assert_called_once()

    def test_expired_cache_refreshes(self):
        self.auto()
        self.cache.mkdir()
        (self.cache / "location.json").write_text(json.dumps({"checked": 0, "coords": [0, 0]}))
        self.assertEqual(self.resolve(), (55.95, -3.19))
        self.urlopen.assert_called_once()

    def test_corrupt_cache_is_replaced(self):
        self.auto()
        self.cache.mkdir()
        (self.cache / "location.json").write_text("[]")
        self.assertEqual(self.resolve(), (55.95, -3.19))

    def test_invalid_coordinates_are_rejected(self):
        for lat, lon in [(91, 0), (0, 181), (float("nan"), 0), (True, 0), (None, 0)]:
            with self.subTest(lat=lat, lon=lon):
                with self.assertRaises((ValueError, TypeError)):
                    location.coordinates(lat, lon)

    def test_concurrent_output_does_not_wait_or_fetch(self):
        self.auto()
        self.cache.mkdir()
        with (self.cache / "location.lock").open("a") as lock:
            location.fcntl.flock(lock, location.fcntl.LOCK_EX)
            self.assertIsNone(self.resolve())
        self.urlopen.assert_not_called()


if __name__ == "__main__":
    unittest.main()
