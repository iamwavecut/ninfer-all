"""Compile one stable subset of the release objects, preserving their full CUDA target set."""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess


def select_objects(commands, index, count):
    if count < 1 or not 0 <= index < count:
        raise ValueError("Invalid object shard")
    outputs = set()
    for item in commands:
        output = item["output"].replace("\\", "/")
        if not output.endswith((".o", ".obj")):
            continue
        # Object paths are relative to the build tree. Host checkout paths and
        # compilation database order must not change shard ownership.
        owner = int.from_bytes(hashlib.sha256(output.encode()).digest()[:8], "big") % count
        if owner == index:
            outputs.add(output)
    return sorted(outputs)


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
    objects = select_objects(release_commands(args.build), args.shard, args.count)
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
