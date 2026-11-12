"""Cancel older master CI and Native release runs without touching other branches.

An older Native release run that still compiles is left to finish its objects, which serve the next
commit's build: a cancelled compile loses its work. Once it only links, packages or publishes a
revision that is no longer master's head, it is cancelled so the newer run, which waits in the same
concurrency group, starts at once."""
import json
import os
import subprocess
import time


RELEASE = ".github/workflows/native-release.yml"
WORKFLOWS = {".github/workflows/ci.yml", RELEASE}
# A temporary runner compiles, links and packages Linux in one job.
RENTED_LABEL = "ninfer-native-debian13"


def superseded(run, trigger):
    return (run["status"] != "completed" and run["head_branch"] == "master"
            and run["event"] in ("push", "workflow_dispatch")
            and run["path"].split("@", 1)[0] in WORKFLOWS
            and run["head_sha"] != trigger["head_sha"]
            and run["created_at"] <= trigger["created_at"])


def compiling(jobs):
    """Whether a Native release run has an object job running: a shard, or a rented full build."""
    return any(job["status"] == "in_progress" and (
        job["name"].startswith(("windows-objects", "linux-objects"))
        or (job["name"].startswith("linux (") and RENTED_LABEL in job.get("labels", [])))
        for job in jobs)


def cancel(run, api, post, sleep=time.sleep, checks=13, interval=5):
    """Cancel one run, or accept its own end within about a minute.

    CI also cancels older master runs through its concurrency group. GitHub
    refuses a second cancellation of a run it is already stopping, and that run still reads
    in_progress for a few seconds.
    """
    error = post(run["id"])
    if not error:
        print(f"Cancellation requested: {run['name']} ({run['id']})", flush=True)
        return
    # A run GitHub never queued has no jobs and cannot be cancelled; five such runs of 11 November
    # (e4b8f4485) stopped every coordinator since before it reached the runs it should cancel.
    if "has not been queued yet" in error:
        print(f"::notice::Never queued, left as is: {run['name']} ({run['id']})", flush=True)
        return
    for check in range(checks):
        if check:
            sleep(interval)
        if api(f"actions/runs/{run['id']}")["status"] == "completed":
            print(f"Already stopped: {run['name']} ({run['id']})", flush=True)
            return
    raise RuntimeError(f"Could not cancel {run['name']} ({run['id']}): {error}")


def main():
    repo = os.environ["GITHUB_REPOSITORY"]

    def api(path):
        return json.loads(subprocess.check_output(["gh", "api", f"repos/{repo}/{path}"], text=True))

    def post(run_id):
        result = subprocess.run(
            ["gh", "api", "-X", "POST", f"repos/{repo}/actions/runs/{run_id}/cancel", "--silent"],
            capture_output=True, text=True)
        return (result.stderr.strip() or f"exit {result.returncode}") if result.returncode else ""

    trigger = api(f"actions/runs/{os.environ['GITHUB_RUN_ID']}")
    if api("commits/master")["sha"] != trigger["head_sha"]:
        print("A newer master coordinator owns cancellation.")
        return
    pages = []
    for status in ("queued", "in_progress", "waiting", "pending", "requested"):
        pages.extend(json.loads(subprocess.check_output(
            ["gh", "api", "--paginate", "--slurp",
             f"repos/{repo}/actions/runs?branch=master&status={status}&per_page=100"], text=True)))
    for page in pages:
        for run in page["workflow_runs"]:
            if not superseded(run, trigger):
                continue
            # Never cancel a newer commit if a push races this coordinator.
            comparison = api(f"compare/{trigger['head_sha']}...{run['head_sha']}")
            if comparison["status"] != "behind":
                continue
            if run["path"].split("@", 1)[0] == RELEASE:
                jobs = [job for page in json.loads(subprocess.check_output(
                    ["gh", "api", "--paginate", "--slurp",
                     f"repos/{repo}/actions/runs/{run['id']}/jobs?per_page=100"], text=True))
                    for job in page["jobs"]]
                if compiling(jobs):
                    # Its revision check after the objects then skips packaging.
                    print(f"Still compiling, left to finish its objects: {run['name']} ({run['id']})",
                          flush=True)
                    continue
            cancel(run, api, post)


if __name__ == "__main__":
    main()
