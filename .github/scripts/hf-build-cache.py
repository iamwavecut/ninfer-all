"""Restore, checkpoint and retire compiler/dependency caches in the private HF bucket."""
import argparse
from contextlib import contextmanager
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import sys
import tarfile
import tempfile
import time
import threading

import hf_client

BUCKET = 'WaveCut/ninfer-cache'
ROOT = 'github-actions/v2/iamwavecut/ninfer-all'
# Retention owns both areas; archive-linux-artifact.py writes the second one.
AREAS = {'github-actions': ROOT, 'native-artifacts': 'native-artifacts/v2/iamwavecut/ninfer-all'}
MASTER = 'refs/heads/master'
MANIFEST = 'snapshot.json'
STALE_SECONDS = 3 * 24 * 3600
# A merged branch's snapshots seed master's first build of the merge, which can wait in the queue.
MERGED_SECONDS = 24 * 3600
MERGE_SUBJECT = re.compile(r'Merge pull request #\d+ from ([^/\s]+)/(\S+)$')


def digest(path):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, 'sha256').hexdigest()


def ref_key(ref):
    return hashlib.sha256(ref.encode()).hexdigest()[:20]


def prefix(family, ref):
    if not re.fullmatch(r'[a-zA-Z0-9_.-]+', family):
        raise ValueError('Invalid cache family')
    return f'{ROOT}/{ref_key(ref)}/{family}'


def unpack(archive, directory):
    with tarfile.open(archive) as package:
        members = []
        for item in package:
            path = PurePosixPath(item.name)
            if path.is_absolute() or '..' in path.parts or '\\' in item.name or ':' in item.name or \
                    not (item.isdir() or item.isfile()):
                raise ValueError('Invalid cache archive member')
            # Snapshots restored into one directory overlap: a Windows shard that started from the
            # aggregate still holds its copies of other shards' entries. A file at least as new as
            # the member stays, so an old copy of a ccache manifest cannot replace the one its
            # owner shard has since extended with the current headers' entry.
            if item.isfile():
                try:
                    if (directory / path).stat().st_mtime >= item.mtime:
                        continue
                except FileNotFoundError:
                    pass
            members.append(item)
        package.extractall(directory, members=members, filter='data')


def entries(directory):
    """Yield the cache files a snapshot holds, as (path, relative POSIX name)."""
    for path in sorted(directory.rglob('*')):
        relative = path.relative_to(directory)
        if path.is_symlink() or 'tmp' in relative.parts or path.name in ('stats', 'lock') or \
                path.name.endswith(('.lock', '.tmp')):
            continue
        yield path, relative.as_posix()


def write_baseline(directory, output):
    """Record each cache file's size, so that a later save can leave out the unchanged ones."""
    lines = []
    for path, name in entries(directory):
        try:
            if path.is_file():
                lines.append(f'{path.stat().st_size}\t{name}')
        except FileNotFoundError:
            continue
    output.write_text(''.join(line + '\n' for line in lines))


def read_baseline(path):
    baseline = {}
    for line in path.read_text().splitlines():
        size, _, name = line.partition('\t')
        baseline[name] = int(size)
    return baseline


def pack(directory, archive, baseline=None):
    """Archive the cache files, without those still as large as in the baseline.

    A baseline lists the files another family's snapshot supplied. Result files are named by their
    content and a manifest grows when ccache adds an entry, so a file of the recorded size is still
    the one that snapshot holds, also after a hit renewed its modification time."""
    count = 0
    with tarfile.open(archive, 'w') as package:
        for path, name in entries(directory):
            try:
                with path.open('rb') as stream:
                    # Read metadata from the open inode: ccache replaces entries atomically.
                    item = package.gettarinfo(fileobj=stream, arcname=name)
                    if not item.isfile() or (baseline and baseline.get(name) == item.size):
                        continue
                    package.addfile(item, stream)
                    count += 1
            except (FileNotFoundError, IsADirectoryError, PermissionError):
                continue
    return count


@contextmanager
def writer_lock(directory):
    path = directory.parent / (directory.name + '.hf-lock')
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('a+b') as stream:
        stream.write(b'0')
        stream.flush()
        limit = time.monotonic() + 180
        while True:
            try:
                if os.name == 'nt':
                    import msvcrt
                    stream.seek(0)
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                else:
                    import fcntl
                    fcntl.flock(stream, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except OSError:
                if time.monotonic() >= limit:
                    raise TimeoutError('Another cache checkpoint is still running')
                time.sleep(.2)
        try:
            yield
        finally:
            if os.name == 'nt':
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                fcntl.flock(stream, fcntl.LOCK_UN)


def current(api, base, work):
    """Return the family's published snapshot, or None when it has none."""
    path = f'{base}/{MANIFEST}'
    if not list(api.get_bucket_paths_info(BUCKET, [path])):
        return None
    local = work / MANIFEST
    api.download_bucket_files(BUCKET, files=[(path, local)], raise_on_missing_files=True)
    item = json.loads(local.read_text())
    if item['schema'] != 2 or not re.fullmatch(r'[0-9a-f]{64}', item['sha256']) or \
            item['archive'] != f"{base}/{item['sha256']}.tar":
        raise ValueError('Invalid cache manifest')
    return item


def save(api, directory, family, ref, verify=False, baseline=None):
    if not directory.is_dir():
        raise ValueError('Cache directory is missing')
    base = prefix(family, ref)
    with writer_lock(directory), tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        work = Path(temporary)
        previous = current(api, base, work)
        run_id = os.environ.get('GITHUB_RUN_ID', 'local')
        if previous and run_id.isdecimal() and str(previous.get('run_id', '')).isdecimal() and \
                int(previous['run_id']) > int(run_id):
            print('A newer native run already saved this family; skipping stale writer.', flush=True)
            return False
        archive = work / 'cache.tar'
        count = pack(directory, archive, baseline)
        # With a baseline, an empty archive says that every file the build holds is in the
        # snapshot the baseline came from: a package job that requires this family finds it.
        if not count and baseline is None:
            raise ValueError('Refusing to replace a checkpoint with an empty cache')
        sha256 = digest(archive)
        item = {'schema': 2, 'family': family, 'ref': ref,
                'archive': f'{base}/{sha256}.tar', 'sha256': sha256,
                'bytes': archive.stat().st_size, 'files': count,
                'created_ns': time.time_ns(), 'run_id': run_id}
        print(json.dumps({'state': 'uploading', 'family': family, 'bytes': item['bytes'], 'files': count}), flush=True)
        # Archives are named by content: an unchanged cache only republishes its manifest.
        stored = list(api.get_bucket_paths_info(BUCKET, [item['archive']]))
        if not stored or stored[0].size != item['bytes']:
            api.batch_bucket_files(BUCKET, add=[(archive, item['archive'])])
            stored = list(api.get_bucket_paths_info(BUCKET, [item['archive']]))
        if len(stored) != 1 or stored[0].size != item['bytes']:
            raise ValueError('Uploaded cache size differs')
        if verify:
            check = work / 'check.tar'
            api.download_bucket_files(BUCKET, files=[(item['archive'], check)], raise_on_missing_files=True)
            if digest(check) != item['sha256']:
                raise ValueError('Uploaded cache checksum differs')
        # Publish metadata only after the archive is available. Until then readers keep the
        # previous archive, which an interrupted upload leaves in place.
        metadata = work / MANIFEST
        metadata.write_text(json.dumps(item) + '\n')
        api.batch_bucket_files(BUCKET, add=[(metadata, f'{base}/{MANIFEST}')])
        print(json.dumps({'state': 'saved', **item}), flush=True)
        # The family keeps only its current pair. Readers of the replaced archive follow the new manifest.
        try:
            stale = [entry.path for entry in api.list_bucket_tree(BUCKET, prefix=base + '/', recursive=True)
                     if entry.type == 'file' and entry.path not in (item['archive'], f'{base}/{MANIFEST}')]
            if stale:
                api.batch_bucket_files(BUCKET, delete=stale)
            print(json.dumps({'state': 'replaced', 'files': len(stale)}), flush=True)
        except Exception as error:
            print(f'::warning::The replaced cache archive remains ({type(error).__name__}); the next save removes it.', flush=True)
        return True


def fetch(api, base, item, directory, work):
    """Unpack a verified snapshot, following a replacement published while reading it."""
    for _ in range(3):
        archive = work / 'cache.tar'
        try:
            api.download_bucket_files(BUCKET, files=[(item['archive'], archive)], raise_on_missing_files=True)
            if archive.stat().st_size != item['bytes'] or digest(archive) != item['sha256']:
                raise ValueError('Cache checksum differs')
            unpack(archive, directory)
        except Exception as error:
            print(json.dumps({'state': 'checkpoint-rejected', 'error': type(error).__name__}), flush=True)
            newer = current(api, base, work)
            if newer is None or newer['archive'] == item['archive']:
                return False
            item = newer
            continue
        print(json.dumps({'state': 'restored', **item}), flush=True)
        return True
    return False


def restore(api, directory, family, ref, required=False, fallback_families=(), extra_refs=()):
    """Unpack the family's snapshot of the branch, or else of master, then those of extra_refs.

    An extra ref is a branch whose objects this build can reuse, such as the one master merged.
    Its snapshot is optional; where it holds the same file as the first, the newer copy stays."""
    directory.mkdir(parents=True, exist_ok=True)
    had_snapshots = restored = False
    with tempfile.TemporaryDirectory(dir=directory.parent) as temporary:
        work = Path(temporary)
        for source_ref in dict.fromkeys([ref, MASTER]):
            base = prefix(family, source_ref)
            item = current(api, base, work)
            if item is None:
                continue
            had_snapshots = True
            if fetch(api, base, item, directory, work):
                restored = True
                break
        if had_snapshots and not restored:
            raise ValueError('No verified cache checkpoint exists')
        for extra in dict.fromkeys(extra_refs):
            if extra in (ref, MASTER):
                continue
            base = prefix(family, extra)
            item = current(api, base, work)
            if item is not None and fetch(api, base, item, directory, work):
                restored = True
    if restored or had_snapshots:
        return restored
    # Read previous families without replacing their snapshots. Compiler input hashes
    # still decide which restored entries are valid under the current toolchain.
    for fallback in dict.fromkeys(fallback_families):
        if fallback == family:
            raise ValueError('A cache family cannot fall back to itself')
        if restore(api, directory, fallback, ref, extra_refs=extra_refs):
            return True
    if required:
        raise ValueError('No verified cache checkpoint exists')
    print(json.dumps({'state':'cold', 'family':family}), flush=True)
    return False


def prune(api, live_refs, dry_run=False, now=None):
    """Delete bucket state that no current build reads.

    Each area holds <branch key>/<family or artifact kind>/ slots. A branch's artifact backups go
    once the branch is merged or deleted, its compiler caches MERGED_SECONDS after their last write
    then, and both STALE_SECONDS after the branch's last write. On a kept branch, master
    included, a slot goes once it lags the branch's newest write by STALE_SECONDS: renamed
    families and retired Toolkits receive no writes. Earlier layouts of an area go STALE_SECONDS
    after their last writer.
    """
    now = time.time() if now is None else now
    live = {ref_key(ref) for ref in (MASTER, *live_refs)}
    slots = {}
    for area, root in AREAS.items():
        for entry in api.list_bucket_tree(BUCKET, prefix=area + '/', recursive=True):
            if entry.type != 'file':
                continue
            if entry.path.startswith(root + '/'):
                key, _, rest = entry.path[len(root) + 1:].partition('/')
                slot = (root, key, rest.partition('/')[0])
            else:
                slot = (area, entry.path.split('/')[1], None)
            slots.setdefault(slot, []).append(entry)
    # A file without a timestamp counts as fresh.
    stamp = lambda entry: (entry.uploaded_at or entry.mtime).timestamp() if entry.uploaded_at or entry.mtime else now
    written = {slot: max(map(stamp, entries)) for slot, entries in slots.items()}
    branch_written = {}
    for (root, key, name), stamp in written.items():
        branch_written[root, key] = max(branch_written.get((root, key), stamp), stamp)
    doomed = []
    for slot, entries in sorted(slots.items(), key=lambda item: str(item[0])):
        root, key, name = slot
        latest = branch_written[root, key]
        if name is None:
            reason = 'earlier-layout' if latest < now - STALE_SECONDS else None
        elif key not in live:
            # Compiler caches of a merged branch seed master's build of the merge; backups do not.
            reason = 'merged-or-deleted-branch' if root != ROOT or latest < now - MERGED_SECONDS else None
        elif key != ref_key(MASTER) and latest < now - STALE_SECONDS:
            reason = 'idle-branch'
        else:
            reason = 'superseded' if written[slot] < latest - STALE_SECONDS else None
        if reason:
            doomed += [entry.path for entry in entries]
            print(json.dumps({'state': 'planned' if dry_run else 'deleting', 'reason': reason,
                              'slot': '/'.join(filter(None, slot)), 'files': len(entries),
                              'bytes': sum(entry.size for entry in entries)}), flush=True)
    if doomed and not dry_run:
        api.batch_bucket_files(BUCKET, delete=doomed)
    print(json.dumps({'state': 'dry-run' if dry_run else 'pruned', 'files': len(doomed)}), flush=True)
    return doomed


def merged_branch(subject, repository):
    """The branch a merge commit's subject names, when it is one of this repository's branches."""
    match = MERGE_SUBJECT.match(subject.strip())
    if not match or match[1] != repository.split('/')[0]:
        return None
    return f'refs/heads/{match[2]}'


def head_merged_branch():
    import subprocess
    result = subprocess.run(['git', 'log', '-1', '--format=%s'], capture_output=True, text=True)
    if result.returncode:
        # Without the merged branch, master's first build of a merge starts from its older snapshot.
        error = (result.stderr.strip().splitlines() or [f'exit {result.returncode}'])[0]
        print(f'::warning::Cannot read the head commit for its merged branch: {error}', flush=True)
        return None
    return merged_branch(result.stdout, os.environ.get('GITHUB_REPOSITORY', ''))


def ci_target_ref():
    """The branch a CI run writes for: its pull request's head branch, or the pushed branch."""
    event, ref = os.environ.get('GITHUB_EVENT_NAME'), os.environ.get('GITHUB_REF', '')
    if event == 'pull_request':
        head = os.environ.get('GITHUB_HEAD_REF', '')
        return f'refs/heads/{head}' if head else None
    if event == 'push' and (ref == MASTER or ref.startswith('refs/heads/ci/')):
        return ref
    return None


def require_cache_writer(family, ref):
    """Reject writes from other jobs, other branches and retired recovery workflows.

    Native release and the Linux bootstrap write the native families. CI writes only its own family,
    for the branch it builds: master, a ci/ branch, or a pull request's head branch. A pull request
    from a fork receives no bucket credential."""
    workflow = os.environ.get('GITHUB_WORKFLOW_REF', '').split('@')[0]
    if workflow == 'iamwavecut/ninfer-all/.github/workflows/ci.yml':
        if (ref != ci_target_ref() or os.environ.get('GITHUB_JOB') != 'cuda' or
                not family.startswith('ci-linux-cuda-') or
                not os.environ.get('GITHUB_RUN_ID', '').isdecimal()):
            raise ValueError("CI may save only its own family for the branch it builds")
        return
    bootstrap = workflow == 'iamwavecut/ninfer-all/.github/workflows/linux-cache-bootstrap.yml'
    if ((workflow != 'iamwavecut/ninfer-all/.github/workflows/native-release.yml' and not bootstrap) or
            os.environ.get('GITHUB_EVENT_NAME') not in ('push', 'workflow_dispatch') or
            not ref.startswith('refs/heads/') or ref != os.environ.get('GITHUB_REF') or
            not os.environ.get('GITHUB_RUN_ID', '').isdecimal()):
        raise ValueError('Only a native artifact build may save shared caches')
    job = os.environ.get('GITHUB_JOB')
    if bootstrap and (job != 'linux' or not family.startswith('native-linux-cuda-') or
                      not family.endswith('-shared-v1')):
        raise ValueError('Bootstrap may write only the shared Linux artifact cache')
    if not ((job == 'linux' and family.startswith('native-linux-cuda-')) or
            (job == 'windows' and family.startswith(('native-windows-cuda-', 'vcpkg-')))):
        raise ValueError('Cache family does not belong to this native job')


def require_retention_job():
    """Only master's retention workflow deletes shared state; a maintainer may run it locally."""
    if os.environ.get('GITHUB_ACTIONS') == 'true' and os.environ.get('GITHUB_WORKFLOW_REF') != \
            'iamwavecut/ninfer-all/.github/workflows/build-cache-retention.yml@refs/heads/master':
        raise ValueError('Only the master retention workflow may delete shared caches')


def main():
    parser = argparse.ArgumentParser()
    actions = parser.add_subparsers(dest='action', required=True)
    writer = actions.add_parser('save')
    reader = actions.add_parser('restore')
    for command in (writer, reader):
        command.add_argument('--directory', type=Path, required=True)
        command.add_argument('--family', required=True)
    writer.add_argument('--verify-upload', action='store_true')
    writer.add_argument('--target-ref', default=os.environ.get('HF_CACHE_TARGET_REF') or None,
                        help="Branch whose snapshot to write; defaults to the run's own ref")
    writer.add_argument('--exclude-baseline', type=Path, default=os.environ.get('HF_CACHE_BASELINE') or None,
                        help='Leave out the files a restore recorded and the build did not change')
    reader.add_argument('--fallback-family', action='append', default=[])
    reader.add_argument('--source-ref', help='Reader branch; defaults to the current branch')
    reader.add_argument('--also-ref', action='append', default=[],
                        help='Also unpack this branch\'s snapshot of the family when it has one')
    reader.add_argument('--merged-branch', action='store_true',
                        help='On master, also unpack the snapshot of the branch the head commit merged')
    reader.add_argument('--baseline', type=Path,
                        help='Afterwards record the cache files, for a save with --exclude-baseline')
    need = reader.add_mutually_exclusive_group()
    need.add_argument('--optional', action='store_true', help='Continue without an unavailable reader cache')
    need.add_argument('--required', action='store_true')
    cleaner = actions.add_parser('prune', help='Delete caches and backups that no current build reads')
    cleaner.add_argument('--live-refs', type=Path, required=True,
                         help='File naming one unmerged branch ref per line; master is always kept')
    cleaner.add_argument('--dry-run', action='store_true')
    args = parser.parse_args()
    optional = args.action == 'restore' and args.optional
    if args.action == 'prune':
        require_retention_job()
        live_refs = [line.strip() for line in args.live_refs.read_text().splitlines() if line.strip()]
        if any(not ref.startswith('refs/heads/') for ref in live_refs):
            raise ValueError('Live refs must be branches')
    else:
        args.directory = args.directory.expanduser()
        ref = os.environ['GITHUB_REF']
        if args.action == 'restore' and args.source_ref:
            ref = args.source_ref
        elif args.action == 'save' and args.target_ref:
            ref = args.target_ref
    if args.action == 'save':
        require_cache_writer(args.family, ref)
    elif args.action == 'restore':
        extra_refs = list(args.also_ref)
        if args.merged_branch and ref == MASTER and (merged := head_merged_branch()):
            extra_refs.append(merged)
    try:
        if not os.environ.get('HF_TOKEN'):
            raise ValueError('HF cache credential is not configured')
        api = hf_client.open_api()
        if api.bucket_info(BUCKET).private is not True:
            raise ValueError('Build cache bucket must be private')
        if args.action == 'save':
            baseline = read_baseline(args.exclude_baseline) if args.exclude_baseline else None
            save(api, args.directory, args.family, ref, args.verify_upload, baseline)
        elif args.action == 'restore':
            restore(api, args.directory, args.family, ref, args.required, args.fallback_family, extra_refs)
        else:
            prune(api, live_refs, args.dry_run)
    except Exception as error:
        if not optional:
            raise
        # CI can compile without the cache. Do not print transport errors with signed URLs.
        print(f'::warning::Native cache unavailable ({type(error).__name__}); continuing with local compilation.', flush=True)
    finally:
        if args.action == 'restore' and args.baseline:
            args.directory.mkdir(parents=True, exist_ok=True)
            write_baseline(args.directory, args.baseline)


def heartbeat(done):
    while not done.wait(60):
        print('HF cache transfer or verification is still running.', flush=True)


if __name__ == '__main__':
    hf_client.throttle_progress()
    done = threading.Event()
    threading.Thread(target=heartbeat, args=(done,), daemon=True).start()
    try:
        main()
    except Exception as error:
        # Transport exceptions can contain signed URLs. Keep them out of public CI logs.
        print(f'::error::HF cache operation failed ({type(error).__name__})', flush=True)
        sys.exit(1)
    finally:
        done.set()
