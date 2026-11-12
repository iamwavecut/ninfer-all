"""Compile one stable subset of the release objects, preserving their full CUDA target set."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess

# Cold compile times, in seconds, of the release objects that take at least two minutes, per
# platform, as object-weights.py measures them. A few attention units take 40-60 minutes each; by
# path hash alone four of them met in one Windows shard of two compilers and set its time.
WEIGHTS = Path(__file__).with_name("object-weights.json")


def object_key(output):
    """The object's build-tree path without the platform's object suffix."""
    return output[:-len(".obj")] if output.endswith(".obj") else output[:-len(".o")]


def select_objects(commands, index, count, weights=None):
    """The shard's objects. A weighed object goes, longest first, to the shard with the least weight
    so far; every other object to the shard of its path hash, so no other object's weight or
    presence moves it. Object paths are relative to the build tree: host checkout paths and
    compilation database order do not change ownership."""
    if count < 1 or not 0 <= index < count:
        raise ValueError("Invalid object shard")
    weights = weights or {}
    outputs = {item["output"].replace("\\", "/") for item in commands}
    outputs = {output for output in outputs if output.endswith((".o", ".obj"))}
    weighed = sorted((output for output in outputs if object_key(output) in weights),
                     key=lambda output: (-weights[object_key(output)], output))
    loads, owners = [0] * count, {}
    for output in weighed:
        owner = min(range(count), key=lambda shard: (loads[shard], shard))
        owners[output] = owner
        loads[owner] += weights[object_key(output)]
    for output in outputs - owners.keys():
        owners[output] = int.from_bytes(hashlib.sha256(output.encode()).digest()[:8], "big") % count
    return sorted(output for output, owner in owners.items() if owner == index)


def platform_weights(path=WEIGHTS):
    return json.loads(path.read_text())["windows" if os.name == "nt" else "linux"]


RELEASE_TARGETS = ("ninfer", "ninfer-serve", "ninfer_bench")


def release_commands(build, targets=RELEASE_TARGETS):
    """The compile database entries of the edges that `targets` need. `ninja -t compdb-targets`
    lists them only from Ninja 1.13; Debian 13 ships 1.12. The whole graph's database, kept to the
    commands `ninja -t commands` prints for the targets, is the same set on both."""
    needed = set(subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "commands", *targets], text=True).splitlines())
    database = json.loads(subprocess.check_output(
        ["ninja", "-C", str(build), "-t", "compdb"], text=True))
    return [item for item in database if item.get("command") in needed]


# A generator tool that a custom command runs compiles in every shard that needs the command's output.
GENERATOR_TOOLS = ("/ninfer_webui_embed.dir/",)


def foreign_objects(build, objects):
    """Objects the build compiled outside the shard, from its Ninja log. Ninja runs a target's
    custom command after every library the target links and starts the target's objects after the
    command, so a generated source in a library target makes a shard compile that whole closure."""
    owned = set(objects)
    compiled = set()
    for line in (build / ".ninja_log").read_text().splitlines():
        parts = line.split("\t")
        if line.startswith("#") or len(parts) < 4:
            continue
        output = parts[3].replace("\\", "/")
        if output.endswith((".o", ".obj")):
            compiled.add(output)
    return sorted(output for output in compiled - owned
                  if not any(tool in output for tool in GENERATOR_TOOLS))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", type=Path, default=Path("build-release"))
    parser.add_argument("--shard", type=int, required=True)
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--jobs", type=int, default=2)
    args = parser.parse_args()
    objects = select_objects(release_commands(args.build), args.shard, args.count,
                             platform_weights())
    if not objects:
        raise ValueError("The release object shard is empty")
    (args.build / "object-shard.json").write_text(json.dumps(objects, indent=2) + "\n")
    print(f"Compiling {len(objects)} release objects in shard {args.shard + 1}/{args.count}", flush=True)
    # Keep commands below Windows' command-line limit. Ninja still builds each
    # object's generated dependencies through the original project graph.
    batches = [[]]
    length = 0
    for output in objects:
        if length + len(output) + 3 > 24000:
            batches.append([])
            length = 0
        batches[-1].append(output)
        length += len(output) + 3
    for batch in batches:
        subprocess.run(["cmake", "--build", str(args.build), "-j", str(args.jobs),
                        "--target", *batch], check=True)
    foreign = foreign_objects(args.build, objects)
    if foreign:
        raise SystemExit(f"The shard compiled {len(foreign)} objects of other shards, among them "
                         + ", ".join(foreign[:5]))


if __name__ == "__main__":
    main()
