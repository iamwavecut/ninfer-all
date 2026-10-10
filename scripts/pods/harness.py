#!/usr/bin/env python3
"""Small SSH/tmux harness. Run with Python 3.11; provider credentials stay local."""
import argparse
import ast
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time

ROOT = Path(__file__).resolve().parents[2]
STATE = Path(os.environ.get("NINFER_POD_STATE", ROOT / ".local/pods/pod.json")).expanduser().resolve()
REMOTE = "/workspace/ninfer-work"
SSH_KEY = Path(os.environ.get("NINFER_POD_SSH_KEY", str(Path.home() / ".ssh/id_ed25519")))


def run(argv, **kwargs):
    return subprocess.run(argv, check=True, text=True, **kwargs)


def provider_rejection(action, message, status=0):
    # Classify the CLI's rejection without printing provider text, which can contain secrets.
    lower = str(message).lower()
    reason = "provider rejection"
    if any(word in lower for word in ("credit", "balance", "funds", "payment")):
        reason = "billing or balance"
    elif status in (401, 403):
        reason = "authentication or permission"
    elif any(word in lower for word in ("resource", "available", "capacity", "occupied")):
        reason = "resources unavailable"
    elif any(word in lower for word in ("expired", "deleted", "not found")):
        reason = "instance unavailable"
    code = f" (HTTP {status})" if type(status) is int and status else ""
    return RuntimeError(f"vastai {action} rejected{code}: {reason}")


def vast(*args):
    result = subprocess.run(["vastai", *map(str, args), "--raw"],
                            capture_output=True, text=True)
    # Current CLI versions report HTTP failures as JSON on stderr and still exit zero.
    # Interpret that result before empty stdout can be mistaken for a successful mutation.
    for line in reversed(result.stderr.splitlines()):
        try:
            diagnostic = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(diagnostic, dict) and diagnostic.get("error") is True:
            raise provider_rejection(args[0], diagnostic.get("msg", ""), diagnostic.get("status_code", 0))
    if result.returncode:
        # Provider diagnostics can contain instance API keys. Never print raw responses.
        raise RuntimeError(f"vastai {args[0]} failed, exit {result.returncode}")
    if not result.stdout.strip():
        if args[0] == "start":
            raise provider_rejection("start", "")
        if args[0] in ("destroy", "stop"):
            return {}
        raise RuntimeError(f"vastai {args[0]} returned no machine-readable result")
    try:
        parsed = json.loads(result.stdout)
    except json.JSONDecodeError:
        # Some installed CLI mutation commands print a Python dict even with --raw.
        try:
            parsed = ast.literal_eval(result.stdout)
        except (SyntaxError, ValueError):
            if args[0] == "start":
                if result.stdout.strip() == f"starting instance {args[2]}.":
                    return {}
                raise provider_rejection("start", result.stdout) from None
            if args[0] in ("stop", "destroy"):
                # These CLI versions print prose after success. Callers verify provider state.
                return {}
            raise RuntimeError(f"vastai {args[0]} returned an unrecognized response") from None
    if args[0] == "start" and isinstance(parsed, dict) and parsed.get("success") is False:
        raise provider_rejection("start", parsed.get("msg", ""))
    return parsed


def save(state):
    STATE.parent.mkdir(parents=True, exist_ok=True)
    temporary = STATE.with_name(f"{STATE.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(state, indent=2) + "\n")
    os.replace(temporary, STATE)


def state():
    return json.loads(STATE.read_text())


def arm_guard(pod, hours):
    if pod["provider"] != "vast" or not 0 < hours <= 6:
        raise ValueError("guard requires a Vast rental and a deadline of zero to six hours")
    pod["deadline"] = time.time() + hours * 3600
    command = [sys.executable, str(Path(__file__).resolve()), "_guard",
               str(pod["id"]), str(pod["deadline"])]
    log = STATE.parent / "guard.log"
    if sys.platform == "darwin":
        # launchd owns the guard independently of an interrupted app tool process. caffeinate
        # prevents idle sleep; provider credentials remain in their local CLI configuration.
        label = f"ninfer-rental-{pod['id']}"
        subprocess.run(["launchctl", "remove", label], capture_output=True)
        run(["launchctl", "submit", "-l", label, "-o", str(log), "-e", str(log), "--",
             "/usr/bin/env", f"PATH={os.environ['PATH']}", f"NINFER_POD_STATE={STATE}",
             "/usr/bin/caffeinate", "-i", "-s",
             *command])
        pod["guard_label"] = label
        pod.pop("guard_pid", None)
    else:
        with log.open("a") as output:
            process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output,
                                       stderr=output, start_new_session=True,
                                       env={**os.environ, "NINFER_POD_STATE": str(STATE)})
        pod["guard_pid"] = process.pid
    save(pod)
    print(json.dumps({k: pod[k] for k in ("id", "deadline", "guard_label", "guard_pid") if k in pod}))


def endpoint(pod):
    if pod["provider"] == "vast":
        info = vast("show", "instance", pod["id"])
        if isinstance(info, list):
            info = info[0]
        pod.update(host=info["ssh_host"], port=info["ssh_port"],
                   status=info.get("actual_status"), intended_status=info.get("intended_status"),
                   hourly=info.get("dph_total"))
        mappings = (info.get("ports") or {}).get("22/tcp", [])
        if info.get("public_ipaddr") and mappings:
            pod.update(direct_host=info["public_ipaddr"], direct_port=int(mappings[0]["HostPort"]))
        if pod.get("ssh_direct"):
            pod.update(host=pod["direct_host"], port=pod["direct_port"])
        save(pod)
    return pod


def rental_stopped(actual, intended):
    # A queued start can still report exited. Cancel the desired running state before releasing
    # the guard, or the provider may start compute later when capacity becomes available.
    return actual in ("exited", "stopped") and intended == "stopped"


def ssh_args(pod):
    return ["-i", str(SSH_KEY), "-o", "IdentitiesOnly=yes", "-o", "BatchMode=yes",
            "-o", "ConnectTimeout=15", "-o", "StrictHostKeyChecking=accept-new",
            "-p", str(pod["port"]), f"root@{pod['host']}"]


def ssh(pod, command, **kwargs):
    return run(["ssh", *ssh_args(pod), command], **kwargs)


def copy(pod, source, destination, download=False):
    opts = ssh_args(pod)[:-1]
    opts[opts.index("-p")] = "-P"
    remote = f"root@{pod['host']}:{destination}"
    paths = [remote, str(source)] if download else [str(source), remote]
    run(["scp", "-O", *opts, *paths])


def valid_name(name):
    if not re.fullmatch(r"[a-z0-9][a-z0-9-]{0,47}", name):
        raise ValueError("job name must contain lowercase letters, digits or hyphens")
    return name


def launch(pod, name, body, after=None, retry=False, inputs=(), gpu=False):
    name = valid_name(name)
    after = [valid_name(dependency) for dependency in (after or [])]
    inputs = [path.resolve(strict=True) for path in inputs]
    if len({path.name for path in inputs}) != len(inputs):
        raise ValueError("job input filenames must be unique")
    pod.pop("collected", None)
    save(pod)
    local = STATE.parent / "runners" / f"{name}.sh"
    local.parent.mkdir(exist_ok=True)
    job = f"{REMOTE}/jobs/{name}"
    dependency = ""
    for required in after:
        dependency += f"""
while [ ! -f ../{required}/exit ]; do sleep 2; done
[ "$(cat ../{required}/exit)" = 0 ] || exit 125
cmp -s source.json ../{required}/source.json || exit 125
"""
    script = f"""#!/usr/bin/env bash
set -Eeuo pipefail
cd {shlex.quote(job)}
trap 'rc=$?; date -u +%FT%TZ > {job}/finished; echo "$rc" > {job}/exit' EXIT
# A job ended by stop-job (tmux sends SIGHUP) or a signal must not record success: jobs that
# depend on it read this exit code.
trap 'exit 129' HUP; trap 'exit 130' INT; trap 'exit 143' TERM
date -u +%FT%TZ > started
cp {REMOTE}/source.json source.json
{dependency}
{'exec 9>' + REMOTE + '/gpu.lock; flock 9' if gpu else ''}
cd {REMOTE}/src
export NINFER_JOB_DIR={job}
date -u +%FT%TZ > {job}/executing
bash {job}/routine.sh
"""
    local.write_text(script)
    routine = local.with_name(f"{name}-routine.sh")
    routine.write_text("#!/usr/bin/env bash\nset -Eeuo pipefail\n" + body + "\n")
    if retry:
        ssh(pod, f"test -f {job}/exit && mkdir -p {REMOTE}/previous-jobs && "
            f"mv {job} {REMOTE}/previous-jobs/{name}-{int(time.time())}")
    ssh(pod, f"test ! -e {job} && mkdir -p {job}")
    copy(pod, local, f"{job}/run.sh")
    copy(pod, routine, f"{job}/routine.sh")
    if inputs:
        ssh(pod, f"mkdir -p {job}/inputs")
        hashes = {}
        for path in inputs:
            copy(pod, path, f"{job}/inputs/{shlex.quote(path.name)}")
            hashes[path.name] = hashlib.sha256(path.read_bytes()).hexdigest()
        manifest = local.with_suffix(".inputs.json")
        manifest.write_text(json.dumps(hashes, indent=2) + "\n")
        copy(pod, manifest, f"{job}/inputs.json")
    ssh(pod, f"tmux new-session -d -s ninfer-{name} " +
        shlex.quote(f"bash {job}/run.sh >{job}/output.log 2>&1"))
    print(json.dumps({"job": name, "session": f"ninfer-{name}", "after": after, "gpu": gpu}))


def create_snapshot():
    # A source snapshot includes local edits and the harness, without publishing commits.
    files = run(["git", "ls-files", "-z", "--cached", "--others", "--exclude-standard"],
                cwd=ROOT, capture_output=True).stdout.split("\0")
    digest = hashlib.sha256()
    archive = STATE.parent / "source.tar.gz"
    with tarfile.open(archive, "w:gz") as tar:
        for name in sorted(filter(None, files)):
            file = ROOT / name
            if not file.is_file():
                continue
            digest.update(name.encode() + b"\0" + file.read_bytes())
            tar.add(file, arcname=name, recursive=False)
    manifest = {"commit": run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                              capture_output=True).stdout.strip(),
                "source_sha256": digest.hexdigest()}
    local = STATE.parent / "source.json"
    local.write_text(json.dumps(manifest, indent=2) + "\n")
    return archive, local, manifest


def deploy(pod, snapshot=None):
    if snapshot is None:
        archive, local, manifest = create_snapshot()
    else:
        archive, local = snapshot / "source.tar.gz", snapshot / "source.json"
        manifest = json.loads(local.read_text())
        if not archive.is_file():
            raise ValueError("saved source snapshot is missing its archive")
    # Never modify sources while a job could be compiling them.
    ssh(pod, f"mkdir -p {REMOTE}/jobs; "
        f"for job in {REMOTE}/jobs/*; do "
        '[ ! -d "$job" ] || [ -f "$job/exit" ] || exit 1; done')
    copy(pod, archive, f"{REMOTE}/source.tar.gz")
    copy(pod, local, f"{REMOTE}/source.json.next")
    copy(pod, ROOT / "scripts/pods/deploy_snapshot.py", f"{REMOTE}/deploy_snapshot.py")
    # Preserve unchanged files, and timestamp changed contents when they arrive on the build host.
    # The first deployment precedes bootstrap; there is no old build to invalidate then.
    ssh(pod, f"set -eu; if [ -x {REMOTE}/py311/bin/python ]; then "
        f"{REMOTE}/py311/bin/python {REMOTE}/deploy_snapshot.py --root {REMOTE}; else "
        f"test ! -f {REMOTE}/build/CMakeCache.txt; mkdir -p {REMOTE}/src; cd {REMOTE}/src; "
        "if [ -f ../files ]; then while IFS= read -r file; do rm -f -- \"$file\"; done < ../files; fi; "
        f"tar xzf ../source.tar.gz; tar tzf ../source.tar.gz > ../files; fi; "
        f"mv {REMOTE}/source.json.next {REMOTE}/source.json")
    print(json.dumps(manifest))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("inventory")
    offers = sub.add_parser("offers")
    offers.add_argument("query")
    rent = sub.add_parser("rent")
    rent.add_argument("offer", type=int)
    rent.add_argument("--hours", required=True, type=float)
    rent.add_argument("--disk", default=180, type=int)
    rent.add_argument("--image", required=True)
    attach = sub.add_parser("attach")
    attach.add_argument("--host", required=True)
    attach.add_argument("--port", required=True, type=int)
    sub.add_parser("probe")
    sub.add_parser("direct", help="verify and select this rental's direct SSH endpoint")
    sub.add_parser("wait")
    sub.add_parser("stop")
    resume = sub.add_parser("resume")
    resume.add_argument("--hours", required=True, type=float)
    sub.add_parser("key")
    deploy_command = sub.add_parser("deploy")
    deploy_command.add_argument("--snapshot", type=Path,
                                help="deploy a saved source.tar.gz/source.json pair unchanged")
    start = sub.add_parser("start")
    start.add_argument("name")
    start.add_argument("script", type=Path)
    start.add_argument("--after", action="append", help="required job; repeat for multiple jobs")
    start.add_argument("--retry", action="store_true")
    start.add_argument("--gpu", action="store_true", help="serialize GPU jobs on this host")
    start.add_argument("--input", type=Path, action="append", default=[])
    stop = sub.add_parser("stop-job")
    stop.add_argument("name")
    watch = sub.add_parser("watch")
    watch.add_argument("name")
    status = sub.add_parser("status")
    status.add_argument("name", nargs="?")
    status.add_argument("--lines", type=int, default=12)
    status.add_argument("--file", default="output.log", help="read a named file inside the job")
    status.add_argument("--match", help="show only log lines matching this extended regular expression")
    collect = sub.add_parser("collect")
    collect.add_argument("--job", help="download one terminal job without marking the pod collected")
    migrate = sub.add_parser("copy-from", help="request an asynchronous provider copy from a retained rental")
    migrate.add_argument("source_state", type=Path)
    migrate.add_argument("path", choices=("build", "models", "jobs/runtime-package"))
    sub.add_parser("copy-status", help="inspect provider transfer markers and delivered file sizes")
    sub.add_parser("cancel-copy", help="cancel this rental's pending provider copies, retaining source files")
    sub.add_parser("reboot", help="restart a rental container after all harness jobs finish")
    sub.add_parser("destroy")
    sub.add_parser("discard-empty")
    guard_public = sub.add_parser("guard")
    guard_public.add_argument("--hours", required=True, type=float)
    guard = sub.add_parser("_guard")
    guard.add_argument("instance", type=int)
    guard.add_argument("deadline", type=float)
    args = parser.parse_args()
    if args.command == "inventory":
        keys = ("id", "actual_status", "intended_status", "gpu_name", "num_gpus", "dph_total", "label")
        print(json.dumps([{k: p.get(k) for k in keys} for p in vast("show", "instances")], indent=2))
        return
    if args.command == "offers":
        offers = vast("search", "offers", args.query, "--storage", 180,
                      "--limit", 10, "-o", "dph_total")
        keys = ("id", "gpu_name", "num_gpus", "cpu_name", "cpu_cores_effective", "cpu_ram",
                "dph_total", "disk_space", "disk_bw", "inet_down", "cuda_max_good", "geolocation")
        print(json.dumps([{k: p.get(k) for k in keys} for p in offers], indent=2))
        return
    if args.command == "_guard":
        print(json.dumps({"guard": "armed", "instance": args.instance,
                          "deadline": args.deadline}), flush=True)
        while time.time() < args.deadline:
            current = state()
            if (current.get("id") != args.instance or current.get("destroyed")
                    or current.get("deadline") != args.deadline):
                print(json.dumps({"guard": "released", "instance": args.instance}), flush=True)
                return
            time.sleep(min(30, args.deadline - time.time()))
        # Stop preserves model files/results. No credential is carried in argv or logs.
        for attempt in range(6):
            try:
                current = state()
                if (current.get("id") != args.instance or current.get("destroyed")
                        or current.get("deadline") != args.deadline):
                    print(json.dumps({"guard": "released", "instance": args.instance}), flush=True)
                    return
                pods = vast("show", "instances")
                pod = next((p for p in pods if p["id"] == args.instance), None)
                if pod is None or rental_stopped(pod.get("actual_status"), pod.get("intended_status")):
                    print(json.dumps({"guard": "stop_confirmed", "instance": args.instance}), flush=True)
                    return
                vast("stop", "instance", args.instance)
                print(json.dumps({"guard": "stop_requested", "instance": args.instance,
                                  "attempt": attempt + 1}), flush=True)
            except RuntimeError as error:
                print(json.dumps({"guard": "retry", "instance": args.instance,
                                  "error": str(error)}), flush=True)
            time.sleep(10)
        raise RuntimeError("rental guard could not confirm the provider's stopped state")
    if args.command == "guard":
        arm_guard(state(), args.hours)
        return
    if args.command in ("stop", "resume"):
        pod = endpoint(state())
        if pod["provider"] != "vast" or pod.get("destroyed"):
            raise ValueError("stop/resume requires this harness's retained Vast rental")
        if args.command == "resume":
            if not 0 < args.hours <= 6:
                raise ValueError("resume requires a deadline of zero to six hours")
            arm_guard(pod, args.hours)
            if pod.get("status") != "running":
                vast("start", "instance", pod["id"])
            pod["resume_requested_at"] = time.time()
            save(pod)
            print(json.dumps({"resume_requested": pod["id"]}))
            return
        if pod.get("status") not in ("exited", "stopped"):
            ssh(pod, f"for job in {REMOTE}/jobs/*; do "
                '[ ! -d "$job" ] || [ -f "$job/exit" ] || exit 1; done')
        for attempt in range(6):
            pod = endpoint(pod)
            if rental_stopped(pod.get("status"), pod.get("intended_status")):
                if sys.platform == "darwin" and pod.get("guard_label"):
                    subprocess.run(["launchctl", "remove", pod["guard_label"]], capture_output=True)
                print(json.dumps({"stop_confirmed": pod["id"], "files": "retained"}))
                return
            vast("stop", "instance", pod["id"])
            time.sleep(10)
        raise RuntimeError("could not confirm the retained rental's stopped state")
    if args.command == "rent":
        if args.hours <= 0 or args.hours > 6:
            raise ValueError("rental deadline must be between zero and six hours")
        if STATE.exists() and not state().get("destroyed"):
            raise ValueError("a pod is already attached; finish it before renting another")
        vast("show", "instances")  # Fresh inventory before creation.
        public = SSH_KEY.with_suffix(".pub").read_text().strip()
        keys = vast("show", "ssh-keys")
        if not any(public.split()[1] in str(key) for key in keys):
            vast("create", "ssh-key", str(SSH_KEY.with_suffix(".pub")))
        onstart = STATE.parent / "onstart.sh"
        onstart.parent.mkdir(parents=True, exist_ok=True)
        onstart.write_text("#!/bin/bash\nset -eu\nmkdir -p /root/.ssh\nprintf '%s\\n' " +
                          shlex.quote(public) + " >> /root/.ssh/authorized_keys\n" +
                          # Some hosts hand /root to another owner; sshd then refuses every key.
                          "chown -R root:root /root/.ssh\nchown root:root /root\nchmod 755 /root\n" +
                          "chmod 700 /root/.ssh\nchmod 600 /root/.ssh/authorized_keys\n")
        result = vast("create", "instance", args.offer, "--image", args.image,
                      "--disk", args.disk, "--ssh", "--direct", "--cancel-unavail",
                      "--label", "ninfer-correctness", "--onstart", str(onstart))
        if not result.get("success"):
            raise RuntimeError("provider did not create an instance")
        pod = {"provider": "vast", "id": result["new_contract"],
               "deadline": time.time() + args.hours * 3600}
        save(pod)
        arm_guard(pod, args.hours)
        return
    if args.command == "attach":
        save({"provider": "ssh", "host": args.host, "port": args.port})
        return
    if args.command == "wait":
        last = None
        for _ in range(40):
            pod = endpoint(state())
            if pod.get("status") != last:
                print(json.dumps({k: pod.get(k) for k in ("id", "status", "hourly")}), flush=True)
                last = pod.get("status")
            resuming = time.time() - pod.get("resume_requested_at", 0) < 90
            if (pod.get("status") in ("exited", "offline", "unknown", "stopped") and
                    not (resuming and pod.get("status") in ("exited", "stopped"))):
                raise RuntimeError(f"instance cannot become ready: {pod['status']}")
            result = subprocess.run(["ssh", *ssh_args(pod), "echo SSH_OK; command -v tmux"],
                                    capture_output=True, text=True)
            if result.stdout.startswith("SSH_OK"):
                if result.returncode != 0:
                    ssh(pod, "apt-get update -qq && apt-get install -y -qq tmux")
                print(result.stdout.strip())
                return
            time.sleep(15)
        raise RuntimeError("SSH/tmux readiness timed out after ten minutes")
    if args.command == "discard-empty":
        pod = state()
        if pod["provider"] != "vast" or pod.get("deployed"):
            raise ValueError("only this harness's rental before source deployment can be discarded")
        (STATE.parent / f"failed-{pod['id']}.json").write_text(json.dumps(pod, indent=2) + "\n")
        vast("destroy", "instance", pod["id"], "-y")
        if any(p["id"] == pod["id"] for p in vast("show", "instances")):
            raise RuntimeError("instance destruction is not yet confirmed")
        pod["destroyed"] = True
        save(pod)
        print(json.dumps({"discarded_empty": pod["id"]}))
        return
    pod = endpoint(state())
    if args.command == "direct":
        if not pod.get("direct_host") or not pod.get("direct_port"):
            raise ValueError("provider did not publish a direct SSH endpoint")
        direct = {**pod, "host": pod["direct_host"], "port": pod["direct_port"]}
        ssh(direct, "true")
        direct["ssh_direct"] = True
        save(direct)
        print(json.dumps({"direct_ssh_verified": pod["id"],
                          "host": direct["host"], "port": direct["port"]}))
        return
    if args.command == "reboot":
        if pod.get("provider") != "vast" or pod.get("destroyed"):
            raise ValueError("reboot requires this harness's retained Vast rental")
        ssh(pod, f"for job in {REMOTE}/jobs/*; do "
            '[ ! -d "$job" ] || [ -f "$job/exit" ] || exit 1; done')
        result = subprocess.run(["vastai", "reboot", "instance", str(pod["id"]), "--raw"],
                                capture_output=True, text=True)
        if result.returncode or result.stdout.strip() != f"Rebooting instance {pod['id']}.":
            raise RuntimeError("provider reboot was not acknowledged")
        pod["resume_requested_at"] = time.time()
        save(pod)
        print(json.dumps({"reboot_requested": pod["id"]}))
        return
    if args.command == "cancel-copy":
        if pod.get("provider") != "vast" or not pod.get("copies"):
            raise ValueError("cancel-copy requires copies requested by this rental record")
        result = subprocess.run(["vastai", "cancel", "copy", str(pod["id"]), "--raw"],
                                capture_output=True, text=True)
        if result.returncode or "Remote copy canceled" not in result.stdout:
            raise RuntimeError("provider copy cancellation was not acknowledged")
        pod["copy_cancel_requested_at"] = time.time()
        save(pod)
        print(json.dumps({"copy_cancel_requested": pod["id"]}))
        return
    if args.command == "copy-status":
        info = vast("show", "instance", pod["id"])
        if isinstance(info, list):
            info = info[0]
        message = str(info.get("status_msg", "")).lower()
        print(json.dumps({"transfer_markers": [word for word in
              ("copy", "rsync", "progress", "complete", "success", "error", "failed", "denied",
               "start", "finish", "queue", "prepare", "wait", "connect", "timeout", "refused",
               "not found", "no such", "unavailable", "transfer", "stopped", "invalid",
               "offline", "cannot", "abort", "source", "destination", "mkdir", "busy")
              if word in message], "percentages": re.findall(r"\b\d{1,3}(?:\.\d+)?%", message),
              "rsync_exit_codes": re.findall(r"\bcode (\d+)\b", message),
              "status_length": len(message), "ports": info.get("ports")}), flush=True)
        if message in ("receiving copy...", "error during copying"):
            print(message, flush=True)
        ssh(pod, f"date -u +%FT%TZ; df -h {REMOTE}; "
            f"for dir in {REMOTE}/build {REMOTE}/models {REMOTE}/jobs/runtime-package; do "
            '[ ! -e "$dir" ] || stat -c "%F %n %s bytes" "$dir"; '
            '[ ! -d "$dir" ] || du -sh "$dir"; done; '
            f"find {REMOTE}/models -maxdepth 1 -type f -printf '%f %s bytes\\n' 2>/dev/null || true; "
            f"for file in {REMOTE}/build/CMakeCache.txt {REMOTE}/jobs/runtime-package/runtime.tar.gz; do "
            '[ ! -f "$file" ] || stat -c "%n %s bytes" "$file"; done')
        return
    if args.command == "copy-from":
        source = json.loads(args.source_state.read_text())
        owned = {p["id"] for p in vast("show", "instances")}
        if (source.get("provider") != "vast" or pod.get("provider") != "vast" or
                source.get("id") not in owned or pod["id"] not in owned or source["id"] == pod["id"]):
            raise ValueError("copy-from requires two distinct owned Vast rentals")
        destination = f"{REMOTE}/{args.path}"
        ssh(pod, f"test ! -e {shlex.quote(destination)}")
        result = subprocess.run([
            "vastai", "copy", f"{source['id']}:{destination}/", f"{pod['id']}:{destination}/", "--raw",
        ], capture_output=True, text=True)
        if result.returncode or "Remote to Remote copy initiated" not in result.stdout:
            raise RuntimeError("provider copy was not acknowledged; raw diagnostics withheld")
        pod.setdefault("copies", {})[args.path] = {"source": source["id"], "requested_at": time.time()}
        save(pod)
        print(json.dumps({"copy_requested": args.path, "source": source["id"], "destination": pod["id"]}))
        return
    if args.command == "key":
        result = vast("attach", "ssh", pod["id"], SSH_KEY.with_suffix(".pub").read_text().strip())
        if result.get("success") is False and result.get("msg") != "SSH key already associated with instance.":
            message = str(result.get("msg", "no reason"))
            if "http" in message or "key=" in message:
                message = "provider rejected key attachment"
            raise RuntimeError(message)
        print("SSH key attached")
    elif args.command == "probe":
        ssh(pod, "echo SSH_OK; nvidia-smi --query-gpu=name,driver_version,memory.free "
            "--format=csv,noheader; command -v nvcc; command -v tmux; df -h /workspace")
    elif args.command == "deploy":
        deploy(pod, args.snapshot)
        pod["deployed"] = True
        save(pod)
    elif args.command == "start":
        launch(pod, args.name, args.script.read_text(), args.after, args.retry, args.input, args.gpu)
    elif args.command == "stop-job":
        name = valid_name(args.name)
        job = f"{REMOTE}/jobs/{name}"
        ssh(pod, f"test -d {job} && {{ tmux kill-session -t ninfer-{name} 2>/dev/null || true; }}; "
            f"if [ ! -f {job}/exit ]; then echo 130 > {job}/exit; date -u +%FT%TZ > {job}/finished; fi")
    elif args.command == "watch":
        launch(pod, args.name, (Path(__file__).parent / "satellite.sh").read_text())
    elif args.command == "status":
        if not 1 <= args.lines <= 200:
            raise ValueError("status --lines must be between 1 and 200")
        if args.file in (".", "..") or not re.fullmatch(r"[a-zA-Z0-9_.-]+", args.file):
            raise ValueError("status --file must be a job-local filename")
        pattern = valid_name(args.name) if args.name else "*"
        log = '"$job/"' + shlex.quote(args.file)
        tail = (f"grep -nE -- {shlex.quote(args.match)} {log} | tail -n {args.lines} || true"
                if args.match else f"tail -n {args.lines} {log}")
        ssh(pod, f"for job in {REMOTE}/jobs/{pattern}; do [ -d \"$job\" ] || continue; "
            'printf "%s exit=" "${job##*/}"; '
            'if [ -f "$job/exit" ]; then cat "$job/exit"; '
            'elif ! tmux has-session -t "ninfer-${job##*/}" 2>/dev/null; then echo orphaned; '
            'elif [ -f "$job/executing" ]; then echo running; else echo waiting; fi; '
            f'{tail}; done')
    elif args.command == "collect":
        collection = f"results-{time.time_ns()}"
        target = STATE.parent / collection
        target.mkdir()
        remote_archive = f"{REMOTE}/{collection}.tar.gz"
        if args.job:
            name = valid_name(args.job)
            ssh(pod, f"test -f {REMOTE}/jobs/{name}/exit")
            archive_command = f"tar czf {remote_archive} jobs/{name}"
        else:
            ssh(pod, f"for job in {REMOTE}/jobs/*; do "
                '[ ! -d "$job" ] || [ -f "$job/exit" ] || exit 1; done')
            archive_command = (
                "if [ -f models/flash-next-q2_0-mtp.ninfer.conversion.json ]; then "
                "mkdir -p jobs/models; "
                "cp models/flash-next-q2_0-mtp.ninfer.conversion.json jobs/models/conversion.json; fi; "
                'set -- jobs source.json source.tar.gz; '
                'for input in previous-jobs models/technical.hot models/cpu-q2-layer-12.bin; do '
                '[ ! -e "$input" ] || set -- "$@" "$input"; done; '
                f'tar czf {remote_archive} "$@"')
        # Compression outlives the observing SSH command, like every other long server routine.
        session = f"ninfer-{collection}"
        program = (f"set -Eeuo pipefail; cd {REMOTE}; "
                   f"trap 'echo $? > {remote_archive}.exit' EXIT; " + archive_command)
        ssh(pod, f"tmux new-session -d -s {session} " +
            shlex.quote("bash -c " + shlex.quote(program) + f" > {remote_archive}.log 2>&1"))
        ssh(pod, f"while [ ! -f {remote_archive}.exit ]; do "
            f"tmux has-session -t {session} 2>/dev/null || exit 1; sleep 2; done; "
            f"test \"$(cat {remote_archive}.exit)\" = 0 || "
            f"{{ cat {remote_archive}.log; exit 1; }}")
        copy(pod, target / "results.tar.gz", remote_archive, download=True)
        if args.job:
            print(target)
            return
        pod["collected"] = str(target)
        save(pod)
        print(target)
    elif args.command == "destroy":
        if pod["provider"] != "vast" or not pod.get("collected"):
            raise ValueError("destroy requires a Vast pod and locally collected results")
        # Only this harness's ephemeral rental; never an arbitrary attached host.
        ssh(pod, f"for job in {REMOTE}/jobs/*; do "
            '[ ! -d "$job" ] || [ -f "$job/exit" ] || exit 1; done')
        vast("destroy", "instance", pod["id"], "-y")
        if any(p["id"] == pod["id"] for p in vast("show", "instances")):
            raise RuntimeError("instance destruction is not yet confirmed")
        pod["destroyed"] = True
        save(pod)
        print(json.dumps({"destroyed": pod["id"]}))


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, ValueError, KeyError, subprocess.CalledProcessError) as error:
        print(str(error), file=sys.stderr)
        sys.exit(1)
