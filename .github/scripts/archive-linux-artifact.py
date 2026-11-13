"""Preserve verified Linux release volumes in the private HF bucket before runner cleanup.

Each branch keeps the newest complete backup of each artifact kind; hf-build-cache.py prune
removes a branch's backups after it is merged or deleted.
"""
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import tempfile
import threading
import time

import hf_client

BUCKET = "WaveCut/ninfer-cache"
ROOT = "native-artifacts/v2/iamwavecut/ninfer-all"


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def retire_older(api, kind):
    """Delete every backup of this kind older than its newest complete one."""
    runs = {}
    for entry in api.list_bucket_tree(BUCKET, prefix=kind + "/", recursive=True):
        if entry.type == "file":
            runs.setdefault(entry.path[len(kind) + 1:].partition("/")[0], []).append(entry.path)
    complete = [int(run) for run, paths in runs.items()
                if run.isdecimal() and f"{kind}/{run}/COMPLETE.json" in paths]
    stale = [path for run, paths in runs.items() if complete and
             (not run.isdecimal() or int(run) < max(complete)) for path in paths]
    if stale:
        api.batch_bucket_files(BUCKET, delete=stale)
    return stale


def archive(api, manifest, revision, run_id, ref):
    name = manifest.name.removeprefix("SHA256SUMS-").removesuffix(".txt")
    match = re.fullmatch(r"(ninfer-linux-x64-cuda(?:[0-9.]+-sm[0-9a-]+|12\.9-13\.4))-[0-9a-f]{12}", name)
    if not match:
        raise ValueError("Unexpected Linux artifact name")
    metadata = manifest.parent / name / "BUILD-INFO.json"
    info = json.loads(metadata.read_text())
    if info["source_revision"] != revision or info["platform"] != "linux-x64":
        raise ValueError("Artifact does not belong to this source revision")
    files = [(manifest, manifest.name), (metadata, "BUILD-INFO.json")]
    for line in manifest.read_text().splitlines():
        expected, filename = line.split("  ", 1)
        if not re.fullmatch(re.escape(name) + r"\.tar\.gz\.[0-9]{3}", filename):
            raise ValueError("Unexpected Linux archive volume")
        path = manifest.parent / filename
        if digest(path) != expected:
            raise ValueError("Local archive checksum mismatch")
        files.append((path, filename))
    if len(files) < 3:
        raise ValueError("Artifact contains no archive volumes")
    kind = f"{ROOT}/{hashlib.sha256(ref.encode()).hexdigest()[:20]}/{match[1]}"
    base = f"{kind}/{run_id}"
    records = [{"name": name, "bytes": path.stat().st_size, "sha256": digest(path)}
               for path, name in files]
    api.batch_bucket_files(BUCKET, add=[(path, f"{base}/{name}") for path, name in files])
    with tempfile.TemporaryDirectory(dir=manifest.parent) as temporary:
        check = Path(temporary) / "verify"
        for record in records:
            api.download_bucket_files(BUCKET, files=[(f"{base}/{record['name']}", check)],
                                      raise_on_missing_files=True)
            if check.stat().st_size != record["bytes"] or digest(check) != record["sha256"]:
                raise ValueError("HF artifact round-trip checksum mismatch")
            check.unlink()
    receipt = manifest.parent / f"HF-{name}.json"
    receipt.write_text(json.dumps({"schema": 2, "run_id": run_id, "ref": ref, "bucket": BUCKET,
                                  "prefix": base, "verified_at": time.time(), "files": records}, indent=2) + "\n")
    api.batch_bucket_files(BUCKET, add=[(receipt, f"{base}/COMPLETE.json")])
    print(json.dumps({"state": "verified", "prefix": base}), flush=True)
    try:
        print(json.dumps({"state": "retired", "files": len(retire_older(api, kind))}), flush=True)
    except Exception as error:
        # The next complete backup of this kind retires these too.
        print(f"::warning::Older Linux artifact backups remain ({type(error).__name__})", flush=True)


def failure(error):
    """Describe a failed backup by its HTTP status and the Hub's message, without signed URLs."""
    text = type(error).__name__
    response = getattr(error, "response", None)
    if response is None:
        return text
    try:
        body = response.json()
    except ValueError:
        body = {}
    message = body.get("error") if isinstance(body, dict) else None
    text += f", HTTP {response.status_code}"
    if isinstance(message, str):
        text += ": " + re.sub(r"https?://\S+", "<url>", message)[:300]
    return text


def main():
    workflow = os.environ.get("GITHUB_WORKFLOW_REF", "").split("@", 1)
    allowed = ("iamwavecut/ninfer-all/.github/workflows/native-release.yml",
               "iamwavecut/ninfer-all/.github/workflows/linux-cache-bootstrap.yml")
    if (len(workflow) != 2 or workflow[0] not in allowed or not workflow[1].startswith("refs/heads/")
            or os.environ.get("GITHUB_JOB") not in ("linux", "universal-linux")):
        raise ValueError("Only a native Linux artifact job may archive release volumes")
    api = hf_client.open_api()
    if not api.bucket_info(BUCKET).private:
        raise ValueError("Artifact bucket must be private")
    manifest, = Path("dist").glob("SHA256SUMS-ninfer-linux-*.txt")
    archive(api, manifest, os.environ["GITHUB_SHA"], os.environ["GITHUB_RUN_ID"], os.environ["GITHUB_REF"])


if __name__ == "__main__":
    hf_client.throttle_progress()
    done = threading.Event()
    def heartbeat():
        while not done.wait(60):
            print("Linux artifact upload or checksum verification is still running.", flush=True)
    threading.Thread(target=heartbeat, daemon=True).start()
    try:
        main()
    except Exception as error:
        print(f"::error::Linux artifact backup failed ({failure(error)})", flush=True)
        sys.exit(1)
    finally:
        done.set()
