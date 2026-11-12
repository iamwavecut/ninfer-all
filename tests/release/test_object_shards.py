"""Verify that object sharding covers the release graph exactly once."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import tempfile
import unittest

SPEC = importlib.util.spec_from_file_location(
    "object_shards", Path(__file__).resolve().parents[2] / ".github/scripts/build-object-shard.py")
SHARDS = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(SHARDS)


class ObjectShardsTest(unittest.TestCase):
    def test_complete_disjoint_partition_independent_of_order_and_separator(self):
        objects = [f"src/ops/CMakeFiles/ops.dir/kernel{i}.cu.obj" for i in range(600)]
        commands = [{"output": path} for path in objects] + [{"output": "apps/ninfer.exe"}]
        parts = [SHARDS.select_objects(commands, index, 8) for index in range(8)]
        self.assertEqual(sorted(output for part in parts for output in part), sorted(objects))
        self.assertTrue(all(parts))
        changed = [{"output": item["output"].replace("/", "\\")} for item in reversed(commands)]
        for index, part in enumerate(parts):
            self.assertEqual(SHARDS.select_objects(changed, index, 8), part)

    def test_duplicate_rules_do_not_compile_twice_and_invalid_shards_fail(self):
        self.assertEqual(SHARDS.select_objects([{"output": "x.o"}] * 2, 0, 1), ["x.o"])
        for index, count in ((-1, 8), (8, 8), (0, 0)):
            with self.assertRaises(ValueError):
                SHARDS.select_objects([], index, count)

    def test_foreign_objects_are_compiled_objects_of_other_shards(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / ".ninja_log").write_text(
                "# ninja log v6\n"
                "0\t9\t1\tsrc\\ops\\a.cu.obj\th\n"
                "0\t9\t1\tsrc/ops/b.cu.obj\th\n"
                "0\t9\t1\tsrc/serve/CMakeFiles/ninfer_webui_embed.dir/webui/embed.cpp.obj\th\n"
                "0\t9\t1\tsrc/ops/libninfer_ops.a\th\n")
            self.assertEqual(SHARDS.foreign_objects(build, ["src/ops/a.cu.obj"]),
                             ["src/ops/b.cu.obj"])
            self.assertEqual(SHARDS.foreign_objects(build, ["src/ops/a.cu.obj",
                                                            "src/ops/b.cu.obj"]), [])


@unittest.skipUnless(shutil.which("ninja"), "needs ninja")
class ReleaseCommandsTest(unittest.TestCase):
    def test_release_commands_are_the_edges_the_targets_need(self):
        with tempfile.TemporaryDirectory() as directory:
            build = Path(directory)
            (build / "build.ninja").write_text(
                "rule cc\n  command = cc -c $in -o $out\n"
                "rule link\n  command = ld $in -o $out\n"
                "build a.o: cc a.c\nbuild b.o: cc b.c\nbuild test.o: cc test.c\n"
                "build ninfer: link a.o\nbuild ninfer-serve: link a.o b.o\n"
                "build ninfer_bench: link b.o\nbuild tests: link test.o\n")
            outputs = sorted(item["output"] for item in SHARDS.release_commands(build))
            self.assertEqual(outputs, ["a.o", "b.o", "ninfer", "ninfer-serve", "ninfer_bench"])
            self.assertEqual(SHARDS.select_objects(SHARDS.release_commands(build), 0, 1),
                             ["a.o", "b.o"])
            version = subprocess.run(["ninja", "--version"], capture_output=True, text=True).stdout
            if tuple(int(part) for part in version.split(".")[:2]) >= (1, 13):
                targets = json.loads(subprocess.check_output(
                    ["ninja", "-C", str(build), "-t", "compdb-targets", *SHARDS.RELEASE_TARGETS],
                    text=True))
                self.assertEqual(sorted(item["output"] for item in targets), outputs)
