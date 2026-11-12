"""Verify that object sharding covers the release graph exactly once."""
import importlib.util
import json
from pathlib import Path
import shutil
import subprocess
import sys
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

    def test_weighed_objects_spread_longest_first_and_others_keep_their_hash_owner(self):
        light = [f"src/ops/CMakeFiles/ops.dir/light{i}.cu.obj" for i in range(200)]
        heavy = [f"src/ops/CMakeFiles/ops.dir/heavy{i}.cu.obj" for i in range(6)]
        weights = {f"src/ops/CMakeFiles/ops.dir/heavy{i}.cu": seconds
                   for i, seconds in enumerate((3600, 3000, 2900, 2800, 600, 500))}
        commands = [{"output": path} for path in light + heavy]
        parts = [SHARDS.select_objects(commands, index, 4, weights) for index in range(4)]
        self.assertEqual(sorted(output for part in parts for output in part),
                         sorted(light + heavy))
        loads = [sum(weights[SHARDS.object_key(o)] for o in part if "heavy" in o) for part in parts]
        self.assertEqual(sorted(loads), [3000, 3400, 3400, 3600])
        unweighed = [SHARDS.select_objects(commands, index, 4) for index in range(4)]
        for part, plain in zip(parts, unweighed):
            self.assertEqual([o for o in part if "light" in o], [o for o in plain if "light" in o])
        linux = [{"output": path.replace(".obj", ".o")} for path in heavy]
        self.assertEqual(SHARDS.select_objects(linux, 0, 4, weights),
                         ["src/ops/CMakeFiles/ops.dir/heavy0.cu.o"])

    def test_checked_in_weights_name_both_platforms_in_whole_seconds(self):
        weights = json.loads(SHARDS.WEIGHTS.read_text())
        self.assertEqual(sorted(weights), ["linux", "windows"])
        for times in weights.values():
            self.assertTrue(times)
            self.assertTrue(all(isinstance(value, int) and value >= 120 for value in times.values()))
            self.assertFalse([key for key in times if key.endswith((".o", ".obj"))])

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


class ObjectWeightsTest(unittest.TestCase):
    def test_longest_cold_time_over_toolkits_without_suffix_or_short_objects(self):
        spec = importlib.util.spec_from_file_location(
            "object_weights", Path(__file__).resolve().parents[2] / ".github/scripts/object-weights.py")
        weights_tool = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(weights_tool)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            logs = {"linux-logs-12.9.2-objects-0": "0\t300000\t1\tsrc/a.cu.o\th\n"
                                                   "0\t5000\t1\tsrc/b.cu.o\th\n",
                    "linux-logs-13.4.3-objects-3": "0\t420000\t1\tsrc/a.cu.o\th\n",
                    "windows-logs-13.4.3-objects-1": "# ninja log v7\n"
                                                     "0\t3600000\t1\tsrc\\a.cu.obj\th\n"
                                                     "0\t9000\t1\tsrc\\lib.lib\th\n"}
            for name, text in logs.items():
                (root / "logs" / name / "build-release").mkdir(parents=True)
                (root / "logs" / name / "build-release" / ".ninja_log").write_text(text)
            output = root / "weights.json"
            subprocess.run([sys.executable, spec.origin, str(output), str(root / "logs")],
                           check=True, capture_output=True)
            self.assertEqual(json.loads(output.read_text()),
                             {"linux": {"src/a.cu": 420}, "windows": {"src/a.cu": 3600}})
