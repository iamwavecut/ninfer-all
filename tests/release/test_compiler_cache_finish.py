"""Check how a finished build bounds an object cache restored from an HF snapshot."""
import importlib.util
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import unittest

ROOT = Path(__file__).resolve().parents[2]
SCRIPT = ROOT / ".github/scripts/finish-compiler-cache.py"
SPEC = importlib.util.spec_from_file_location("hf_build_cache", ROOT / ".github/scripts/hf-build-cache.py")
SNAPSHOT = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SNAPSHOT)


@unittest.skipUnless(shutil.which("ccache") and shutil.which("cc"), "needs ccache and a C compiler")
class CompilerCacheFinishTest(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.cache = self.root / "objects"
        self.environment = os.environ | {
            "CCACHE_DIR": str(self.cache), "CCACHE_MAXSIZE": "100Gi", "CCACHE_COMPRESSLEVEL": "5",
            "CCACHE_CONFIGPATH": str(self.root / "ccache.conf")}
        for index in range(7):
            (self.root / f"unit{index}.c").write_text(f"int unit{index}(void) {{ return {index}; }}\n")

    def ccache(self, *arguments):
        return subprocess.run(["ccache", *arguments], env=self.environment, check=True,
                              capture_output=True, text=True, timeout=60).stdout

    def compile(self, index):
        """Compile one unit and report whether ccache answered it from the cache."""
        self.ccache("--zero-stats")
        self.ccache("cc", "-c", str(self.root / f"unit{index}.c"), "-o", str(self.root / f"unit{index}.o"))
        counters = dict(line.split("\t") for line in self.ccache("--print-stats").splitlines())
        return int(counters["direct_cache_hit"]) + int(counters["preprocessed_cache_hit"]) == 1

    def restore_snapshot(self):
        """Replace the cache with what a job restores: the archive omits ccache's counters."""
        archive = self.root / "cache.tar"
        self.assertGreater(SNAPSHOT.pack(self.cache, archive), 0)
        shutil.rmtree(self.cache)
        SNAPSHOT.unpack(archive, self.cache)
        archive.unlink()

    def finish(self, *options):
        result = subprocess.run(
            [sys.executable, str(SCRIPT), "--stats", str(self.root / "stats.json"), *options],
            env=self.environment, capture_output=True, text=True, timeout=120)
        self.assertEqual(result.returncode, 0, result.stdout + result.stderr)

    def build_after_restore(self):
        """Restore six units' entries, then use three of them and add a seventh."""
        for index in range(6):
            self.assertFalse(self.compile(index))
        self.restore_snapshot()
        # Archived modification times are whole seconds; leave a clear gap before the build.
        time.sleep(2.5)
        started = time.time()
        for index in range(3):
            self.assertTrue(self.compile(index))
        self.assertFalse(self.compile(6))
        return started

    def test_finished_build_keeps_only_the_entries_it_used(self):
        started = self.build_after_restore()
        self.finish("--evict-unused-since", str(started))
        for index in (0, 1, 2, 6):
            self.assertTrue(self.compile(index), f"unit{index} was used by the build")
        for index in (3, 4, 5):
            self.assertFalse(self.compile(index), f"unit{index} was not used by the build")

    def test_unfinished_build_keeps_the_restored_entries(self):
        self.build_after_restore()
        self.finish()
        for index in range(7):
            self.assertTrue(self.compile(index), f"unit{index} stays cached")

if __name__ == "__main__":
    unittest.main()
