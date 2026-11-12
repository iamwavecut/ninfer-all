"""Check the CUDA 12.9 glibc 2.41 header patch on synthetic headers."""
import importlib.util
from pathlib import Path
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[2]
SPEC = importlib.util.spec_from_file_location("patch_cuda12_glibc", ROOT / ".github/scripts/patch-cuda12-glibc.py")
PATCH = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCH)

FUNCTIONS = (("double", "sinpi", ""), ("double", "cospi", ""), ("float", "sinpif", "const "),
             ("float", "cospif", ""))
HEADER = "".join(f"extern __DEVICE_FUNCTIONS_DECL__ __device_builtin__ {kind}                 {name}({kind} x);\n"
                 f"__func__({kind} {name}({const}{kind} a));\n" for kind, name, const in FUNCTIONS)
INLINE = "".join(f"__func__({kind} {name}({const}{kind} a))\n{{\n  return a;\n}}\n"
                 for kind, name, const in FUNCTIONS)


class Cuda12GlibcPatchTest(unittest.TestCase):
    def test_patches_once_and_a_second_run_on_the_same_toolkit_keeps_it(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            include = root / "targets/x86_64-linux/include/crt"
            include.mkdir(parents=True)
            header, inline = include / "math_functions.h", include / "math_functions.hpp"
            header.write_text(HEADER)
            inline.write_text(INLINE)
            PATCH.patch_headers(root)
            patched = header.read_text(), inline.read_text()
            self.assertEqual(patched[0].count("x) noexcept (true);"), 4)
            self.assertEqual(patched[0].count("a) noexcept (true))"), 4)
            self.assertEqual(patched[1].count("a) noexcept (true))"), 4)
            self.assertIn("return a;", patched[1])
            PATCH.patch_headers(root)
            self.assertEqual((header.read_text(), inline.read_text()), patched)

    def test_unexpected_headers_fail_without_a_marker(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            include = root / "targets/x86_64-linux/include/crt"
            include.mkdir(parents=True)
            (include / "math_functions.h").write_text(HEADER.replace("sinpi(double x)", "sinpi(double y)"))
            (include / "math_functions.hpp").write_text(INLINE)
            with self.assertRaises(RuntimeError):
                PATCH.patch_headers(root)
            self.assertFalse((root / "ninfer-toolchain-patches.txt").exists())


if __name__ == "__main__":
    unittest.main()
