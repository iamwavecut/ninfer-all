#!/usr/bin/env python3
"""Include boundaries of the public interface.

- Only include/ninfer/engine.h includes the umbrella ninfer/types.h. Other code includes the
  header that defines the types it uses, so a change to one domain recompiles only its users.
- CUDA translation units under src/ do not reach the Engine-level headers: a change to the Engine
  options, requests or statistics would otherwise recompile the kernels. The kernels take what
  they need from the small headers (ninfer/base.h, ninfer/model_enums.h,
  ninfer/kv_cache_storage.h, ninfer/startup.h).
- CUDA translation units under src/ops/ do not reach the host API headers core/device.h,
  ninfer/ops/linear.h, ninfer/ops/softmax_attention.h, core/paged_kv_cache.h and
  ops/common/device_route.h, which most host code includes and changes often. Kernels take their
  launch types from core/device_launch.h, ninfer/ops/linear_policy.h,
  ninfer/ops/softmax_attention_envelope.h and core/paged_kv_view.h, and a route lookup from
  ops/common/device_route_lookup.h. A unit that implements or calls the host API includes that
  header itself and is listed in HOST_API_UNITS.
- The host launchers of the causal-cache attention (causal_cache/**/*.cpp) reach no kernel header
  other than geometry.cuh, and the kernel units' headers (causal_cache/**/*_launch.cuh, which the
  generated units of kernel_unit.cu.in include) and the reducers and merges there (*_reduce.cu,
  *_merge.cu) reach none of the launchers' routing headers. An edit of a kernel then recompiles
  only its own units, and an edit of the routing compiles no kernel.
- The GGUF bridge's kernel units (ggml_vec_*.cu, ggml_mmq_*.cu, ggml_fattn.cu) reach its types
  (ggml_bridge_args.h) but not its entry points (ggml_bridge.h), so a new entry point or a new
  block type's value compiles none of them; the vector units reach no matrix launch
  (ggml_bridge_mmq_launch.cuh), the matrix units no vector launch (ggml_bridge_vec_launch.cuh).
- ninfer/types_fwd.h declares every struct, class and enum the headers listed in ninfer/types.h
  define, with the same keyword, and nothing else.

The check reads #include lines as text and resolves them against the include directories the build
uses, so an include inside an #if counts.
"""
import os
import re
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
INCLUDE_DIRS = ['include', 'src', 'tests', 'bench', 'third_party']
SOURCE_DIRS = ['include', 'src', 'apps', 'tests', 'bench', 'tools']
SOURCE_SUFFIXES = ('.h', '.hpp', '.cuh', '.inl', '.cpp', '.cc', '.cu')
UMBRELLA = 'include/ninfer/types.h'
UMBRELLA_OWNER = 'include/ninfer/engine.h'
DECLARATIONS = 'include/ninfer/types_fwd.h'
ENGINE_LEVEL = {
    'include/ninfer/engine.h',
    'include/ninfer/engine_options.h',
    'include/ninfer/observability/execution.h',
    'include/ninfer/observability/resources.h',
    'include/ninfer/observability/snapshot.h',
    'include/ninfer/request.h',
    'include/ninfer/types.h',
}
HOST_API = (
    'src/core/device.h',
    'include/ninfer/ops/linear.h',
    'include/ninfer/ops/softmax_attention.h',
    'src/core/paged_kv_cache.h',
    'src/ops/common/device_route.h',
)
HOST_API_UNITS = {
    'src/ops/linear/t2/t2_a8.cu': {'src/ops/common/device_route.h'},  # reads the installed profile
    'src/ops/linear_add/q4/q4_linear_add.cu': {'src/ops/common/device_route.h'},  # routed_candidate
    'src/ops/linear_topk/t2.cu': {'include/ninfer/ops/linear.h'},  # calls ops::linear
    'src/ops/softmax_attention/dense/causal_cache/merge.cu': {'include/ninfer/ops/softmax_attention.h'},
}
GGUF = 'src/ops/linear/gguf'
GGUF_ENTRY_POINTS = f'{GGUF}/ggml_bridge.h'
GGUF_VECTOR_LAUNCHES = f'{GGUF}/ggml_bridge_vec_launch.cuh'
GGUF_MATRIX_LAUNCHES = f'{GGUF}/ggml_bridge_mmq_launch.cuh'
CAUSAL_CACHE = 'src/ops/softmax_attention/dense/causal_cache'
CAUSAL_CACHE_HOST_KERNEL_HEADERS = {f'{CAUSAL_CACHE}/geometry.cuh'}
CAUSAL_CACHE_ROUTING = {
    'include/ninfer/ops/softmax_attention_envelope.h',
    'src/core/paged_kv_view.h',
    f'{CAUSAL_CACHE}/int8/prompt_fast_plan.h',
    f'{CAUSAL_CACHE}/int8/small_t_route.h',
    f'{CAUSAL_CACHE}/prompt_launch.h',
    f'{CAUSAL_CACHE}/prompt_policy.h',
    f'{CAUSAL_CACHE}/route.h',
    f'{CAUSAL_CACHE}/small_t_invocation.h',
    f'{CAUSAL_CACHE}/small_t_launch.h',
}
INCLUDE = re.compile(r'^\s*#\s*include\s*[<"]([^>"]+)[>"]', re.M)
TYPE = re.compile(r'^(struct|class|enum class)\s+(\w+)\b[^;]*?(\{|;)', re.M)


def resolve(source, name, cache):
    for directory in [os.path.dirname(source)] + INCLUDE_DIRS:
        candidate = os.path.normpath(os.path.join(directory, name))
        if candidate in cache or os.path.isfile(os.path.join(ROOT, candidate)):
            cache.add(candidate)
            return candidate
    return None


def includes(path, memo, cache):
    if path not in memo:
        try:
            with open(os.path.join(ROOT, path), encoding='utf-8', errors='replace') as file:
                text = file.read()
        except OSError:
            text = ''
        memo[path] = [r for r in (resolve(path, n, cache) for n in INCLUDE.findall(text)) if r]
    return memo[path]


def chain_to(start, targets, memo, cache):
    parent = {start: None}
    queue = [start]
    while queue:
        current = queue.pop(0)
        if current in targets:
            path = []
            while current is not None:
                path.append(current)
                current = parent[current]
            return path[::-1]
        for child in includes(current, memo, cache):
            if child not in parent:
                parent[child] = current
                queue.append(child)
    return None


def read(path):
    with open(os.path.join(ROOT, path), encoding='utf-8') as file:
        return file.read()


def declaration_failures(memo, cache):
    defined = {}
    for header in includes(UMBRELLA, memo, cache):
        if header != DECLARATIONS:
            defined.update({m[1]: m[0] for m in TYPE.findall(read(header)) if m[2] == '{'})
    declared = {m[1]: m[0] for m in TYPE.findall(read(DECLARATIONS)) if m[2] == ';'}
    failures = [f'{DECLARATIONS} lacks {kind} {name}' for name, kind in sorted(defined.items())
                if declared.get(name) != kind]
    failures += [f'{DECLARATIONS} declares {name}, which no listed header defines'
                 for name in sorted(set(declared) - set(defined))]
    return failures


def main():
    memo, cache, failures = {}, set(), []
    failures += declaration_failures(memo, cache)
    for top in SOURCE_DIRS:
        for directory, _, names in os.walk(os.path.join(ROOT, top)):
            for name in sorted(names):
                if not name.endswith(SOURCE_SUFFIXES):
                    continue
                source = os.path.relpath(os.path.join(directory, name), ROOT)
                if source != UMBRELLA_OWNER and UMBRELLA in includes(source, memo, cache):
                    failures.append(f'{source} includes {UMBRELLA}; include the defining headers')
    for directory, _, names in os.walk(os.path.join(ROOT, 'src')):
        for name in sorted(names):
            if not name.endswith('.cu'):
                continue
            source = os.path.relpath(os.path.join(directory, name), ROOT)
            chain = chain_to(source, ENGINE_LEVEL, memo, cache)
            if chain:
                failures.append(f'{source} reaches {chain[-1]}: ' + ' -> '.join(chain))
    for directory, _, names in os.walk(os.path.join(ROOT, 'src', 'ops')):
        for name in sorted(names):
            if not name.endswith('.cu'):
                continue
            source = os.path.relpath(os.path.join(directory, name), ROOT)
            allowed = HOST_API_UNITS.get(source, set())
            for header in HOST_API:
                chain = chain_to(source, {header}, memo, cache)
                if chain and not (header in allowed and len(chain) == 2):
                    failures.append(f'{source} reaches {header}: ' + ' -> '.join(chain))
    names = sorted(os.path.relpath(os.path.join(directory, name), os.path.join(ROOT, CAUSAL_CACHE))
                   for directory, _, files in os.walk(os.path.join(ROOT, CAUSAL_CACHE))
                   for name in files)
    kernel_headers = {f'{CAUSAL_CACHE}/{name}' for name in names if name.endswith('.cuh')}
    for name in names:
        source = f'{CAUSAL_CACHE}/{name}'
        if name.endswith('.cpp'):
            chain = chain_to(source, kernel_headers - CAUSAL_CACHE_HOST_KERNEL_HEADERS, memo, cache)
            if chain:
                failures.append(f'{source} reaches kernel header {chain[-1]}: '
                                + ' -> '.join(chain))
        elif name.endswith(('_launch.cuh', '_reduce.cu', '_merge.cu')):
            chain = chain_to(source, CAUSAL_CACHE_ROUTING, memo, cache)
            if chain:
                failures.append(f'{source} reaches routing header {chain[-1]}: '
                                + ' -> '.join(chain))
    for name in sorted(os.listdir(os.path.join(ROOT, GGUF))):
        forbidden = {GGUF_ENTRY_POINTS, GGUF_VECTOR_LAUNCHES, GGUF_MATRIX_LAUNCHES}
        if name.startswith('ggml_vec_'):
            forbidden.discard(GGUF_VECTOR_LAUNCHES)
        elif name.startswith('ggml_mmq_'):
            forbidden.discard(GGUF_MATRIX_LAUNCHES)
        elif name != 'ggml_fattn.cu':
            continue
        source = f'{GGUF}/{name}'
        chain = chain_to(source, forbidden, memo, cache)
        if chain:
            failures.append(f'{source} reaches {chain[-1]}: ' + ' -> '.join(chain))
    for source, headers in sorted(HOST_API_UNITS.items()):
        for header in sorted(headers - set(includes(source, memo, cache))):
            failures.append(f'{source} no longer includes {header}; remove it from HOST_API_UNITS')
    for failure in failures:
        print(failure)
    if failures:
        print(f'{len(failures)} include boundary violation(s)', file=sys.stderr)
        return 1
    return 0


if __name__ == '__main__':
    sys.exit(main())
