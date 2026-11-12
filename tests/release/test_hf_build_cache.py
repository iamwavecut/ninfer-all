"""Check checkpoint recovery and preservation without network access."""
import importlib.util
from contextlib import redirect_stdout
from datetime import datetime, timedelta, timezone
import io
import json
import os
import subprocess
import sys
from pathlib import Path
import tarfile
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

SPEC = importlib.util.spec_from_file_location('hf_cache', Path(__file__).resolve().parents[2] / '.github/scripts/hf-build-cache.py')
CACHE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(CACHE)


class Bucket:
    def __init__(self):
        self.files = {}
        self.times = {}
        self.now = datetime(2026, 10, 26, tzinfo=timezone.utc)
        self.interrupt = False
        self.uploads = []

    def entry(self, path):
        return SimpleNamespace(type='file', path=path, size=len(self.files[path]),
                               uploaded_at=self.times[path], mtime=None)

    def write(self, path, data):
        self.files[path] = data
        self.times[path] = self.now

    def get_bucket_paths_info(self, bucket, paths):
        return [self.entry(p) for p in paths if p in self.files]

    def batch_bucket_files(self, bucket, add=None, delete=None):
        for local, remote in add or ():
            self.uploads.append(remote)
            if self.interrupt and remote.endswith('.tar'):
                self.write(remote, b'partial upload')
                raise ConnectionError('Interrupted upload')
            self.write(remote, Path(local).read_bytes())
        for remote in delete or ():
            del self.files[remote], self.times[remote]

    def download_bucket_files(self, bucket, files, **kwargs):
        for remote, local in files:
            Path(local).write_bytes(self.files[remote])

    def list_bucket_tree(self, bucket, prefix, **kwargs):
        return [self.entry(p) for p in sorted(self.files) if p.startswith(prefix)]


class CheckpointTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.source = self.root / 'source'
        self.source.mkdir()
        self.bucket = Bucket()
        self.ref = 'refs/heads/build/windows-release'
        self.family = 'linux-cuda-13.4.3'

    def save(self, content):
        (self.source / 'object').write_text(content)
        CACHE.save(self.bucket, self.source, self.family, self.ref, verify=True)

    def restored(self):
        target = self.root / 'restored'
        self.assertTrue(CACHE.restore(self.bucket, target, self.family, self.ref, required=True))
        return (target / 'object').read_text()

    def family_files(self, family=None, ref=None):
        base = CACHE.prefix(family or self.family, ref or self.ref) + '/'
        return sorted(p for p in self.bucket.files if p.startswith(base))

    def manifest(self):
        return json.loads(self.bucket.files[CACHE.prefix(self.family, self.ref) + '/snapshot.json'])

    def test_each_save_replaces_the_previous_snapshot(self):
        for content in ('first', 'second', 'third'):
            self.save(content)
        self.assertEqual(self.restored(), 'third')
        self.assertEqual(self.family_files(), sorted([self.manifest()['archive'],
                                                      CACHE.prefix(self.family, self.ref) + '/snapshot.json']))

    def test_unchanged_cache_republishes_its_manifest_without_uploading_the_archive(self):
        (self.source / 'object').write_text('same')
        for run_id in ('1', '2'):
            with patch.dict(os.environ, GITHUB_RUN_ID=run_id):
                CACHE.save(self.bucket, self.source, self.family, self.ref)
        self.assertEqual(sum(remote.endswith('.tar') for remote in self.bucket.uploads), 1)
        self.assertEqual(self.manifest()['run_id'], '2')
        self.assertEqual(len(self.family_files()), 2)

    def test_family_prefix_does_not_cover_longer_family_names(self):
        self.save('aggregate')
        shard = self.family + '-shard-0'
        CACHE.save(self.bucket, self.source, shard, self.ref)
        (self.source / 'object').write_text('aggregate again')
        CACHE.save(self.bucket, self.source, self.family, self.ref)
        self.assertEqual(len(self.family_files(shard)), 2)

    def test_new_family_reads_previous_objects_without_replacing_old_snapshots(self):
        self.save('old compiler objects')
        preserved = self.bucket.files.copy()
        target = self.root / 'new-cache'
        new_family = self.family + '-host-v2'
        self.assertTrue(CACHE.restore(self.bucket, target, new_family, self.ref,
                                     fallback_families=[self.family], required=True))
        (target / 'new-object').write_text('new compiler objects')
        CACHE.save(self.bucket, target, new_family, self.ref)
        self.assertTrue(all(self.bucket.files[key] == value for key, value in preserved.items()))
        again = self.root / 'again'
        self.assertTrue(CACHE.restore(self.bucket, again, new_family, self.ref,
                                     fallback_families=[self.family]))
        self.assertEqual((again / 'new-object').read_text(), 'new compiler objects')

    def test_corrupt_new_family_cannot_silently_use_an_old_family(self):
        self.save('previous')
        new_family = self.family + '-host-v2'
        CACHE.save(self.bucket, self.source, new_family, self.ref)
        archive, = (p for p in self.family_files(new_family) if p.endswith('.tar'))
        self.bucket.files[archive] = b'corrupt'
        with self.assertRaises(ValueError):
            CACHE.restore(self.bucket, self.root / 'new-cache', new_family, self.ref,
                          fallback_families=[self.family])

    def test_interrupted_replacement_retains_last_completed_snapshot(self):
        self.save('first')
        self.save('second')
        self.bucket.interrupt = True
        with self.assertRaises(ConnectionError):
            self.save('third')
        self.assertEqual(self.restored(), 'second')
        self.bucket.interrupt = False
        self.save('fourth')
        self.assertEqual(self.restored(), 'fourth')
        self.assertEqual(len(self.family_files()), 2)

    def test_reader_follows_a_snapshot_replaced_while_reading(self):
        self.save('first')
        download = self.bucket.download_bucket_files

        def replace_then_download(bucket, files, **kwargs):
            if files[0][0].endswith('.tar') and not hasattr(self, 'replaced'):
                self.replaced = True
                self.save('second')
            download(bucket, files, **kwargs)

        self.bucket.download_bucket_files = replace_then_download
        self.assertEqual(self.restored(), 'second')

    def test_corrupt_snapshot_fails_instead_of_silently_starting_cold(self):
        self.save('first')
        self.bucket.files[self.manifest()['archive']] = b'corrupt'
        with self.assertRaises(ValueError):
            CACHE.restore(self.bucket, self.root / 'restored', self.family, self.ref)

    def test_empty_cache_cannot_replace_completed_checkpoint(self):
        self.save('first')
        (self.source / 'object').unlink()
        with self.assertRaises(ValueError):
            CACHE.save(self.bucket, self.source, self.family, self.ref)
        self.assertEqual(self.restored(), 'first')

    def test_other_branch_cannot_supply_objects_but_master_can(self):
        self.save('private branch')
        target = self.root / 'other'
        self.assertFalse(CACHE.restore(self.bucket, target, self.family, 'refs/heads/other'))
        CACHE.save(self.bucket, self.source, self.family, 'refs/heads/master')
        self.assertTrue(CACHE.restore(self.bucket, target, self.family, 'refs/heads/other'))

    def test_unsafe_members_are_rejected_before_extracting_any_files(self):
        for name, kind in (('../escape', tarfile.REGTYPE), ('/absolute', tarfile.REGTYPE),
                           ('C:/escape', tarfile.REGTYPE), ('link', tarfile.SYMTYPE)):
            with self.subTest(name=name):
                archive = self.root / 'unsafe.tar'
                with tarfile.open(archive, 'w') as package:
                    good = tarfile.TarInfo('good')
                    good.size = 1
                    package.addfile(good, io.BytesIO(b'x'))
                    bad = tarfile.TarInfo(name)
                    bad.type = kind
                    package.addfile(bad)
                with self.assertRaises(ValueError):
                    CACHE.unpack(archive, self.root / 'destination')
                self.assertFalse((self.root / 'destination/good').exists())

    def test_old_copy_in_a_later_snapshot_cannot_replace_a_newer_file(self):
        # A shard that started from the aggregate keeps the manifest of an entry another shard owns.
        aggregate, owner, seeded = self.family, self.family + '-shard-0', self.family + '-shard-1'
        manifest = self.source / 'manifest'
        manifest.write_text('previous headers')
        os.utime(manifest, (1000, 1000))
        CACHE.save(self.bucket, self.source, aggregate, self.ref)
        CACHE.save(self.bucket, self.source, seeded, self.ref)
        manifest.write_text('previous and current headers')
        os.utime(manifest, (2000, 2000))
        CACHE.save(self.bucket, self.source, owner, self.ref)
        target = self.root / 'package'
        for family in (aggregate, owner, seeded):
            self.assertTrue(CACHE.restore(self.bucket, target, family, self.ref, required=True))
        self.assertEqual((target / 'manifest').read_text(), 'previous and current headers')

    def test_baseline_leaves_out_the_files_another_snapshot_supplied_unchanged(self):
        # A shard restores the aggregate first: its own snapshot holds only what the shard changed.
        for name, content in (('result', 'object of another shard'), ('manifest', 'headers')):
            (self.source / name).write_text(content)
        baseline = self.root / 'baseline.tsv'
        CACHE.write_baseline(self.source, baseline)
        os.utime(self.source / 'result', (5000, 5000))  # a hit renews the time, not the content
        (self.source / 'manifest').write_text('headers and the current headers')
        (self.source / 'new-result').write_text('object this shard compiled')
        archive = self.root / 'test.tar'
        self.assertEqual(CACHE.pack(self.source, archive, CACHE.read_baseline(baseline)), 2)
        with tarfile.open(archive) as package:
            self.assertEqual(sorted(package.getnames()), ['manifest', 'new-result'])

    def test_shard_without_changes_still_publishes_its_snapshot(self):
        # Every object of the shard was in the aggregate: the package job's required restore of the
        # shard family must still find a snapshot.
        (self.source / 'object').write_text('aggregate object')
        baseline = self.root / 'baseline.tsv'
        CACHE.write_baseline(self.source, baseline)
        self.assertTrue(CACHE.save(self.bucket, self.source, self.family, self.ref,
                                   baseline=CACHE.read_baseline(baseline)))
        target = self.root / 'package'
        self.assertTrue(CACHE.restore(self.bucket, target, self.family, self.ref, required=True))
        self.assertEqual(list(target.iterdir()), [])

    def test_extra_ref_adds_a_merged_branch_snapshot_to_master(self):
        manifest = self.source / 'manifest'
        manifest.write_text('master headers')
        os.utime(manifest, (1000, 1000))
        CACHE.save(self.bucket, self.source, self.family, 'refs/heads/master')
        branch = self.root / 'branch'
        branch.mkdir()
        (branch / 'manifest').write_text('master and branch headers')
        os.utime(branch / 'manifest', (2000, 2000))
        (branch / 'branch-result').write_text('object the branch compiled')
        CACHE.save(self.bucket, branch, self.family, 'refs/heads/dev/merged')
        target = self.root / 'master-build'
        self.assertTrue(CACHE.restore(self.bucket, target, self.family, 'refs/heads/master',
                                      extra_refs=['refs/heads/dev/merged', 'refs/heads/dev/absent']))
        self.assertEqual((target / 'manifest').read_text(), 'master and branch headers')
        self.assertEqual((target / 'branch-result').read_text(), 'object the branch compiled')
        cold = self.root / 'cold'
        self.assertTrue(CACHE.restore(self.bucket, cold, 'other-family', 'refs/heads/master',
                                      extra_refs=['refs/heads/dev/absent']) is False)

    def test_merged_branch_comes_only_from_this_repository_merge_subject(self):
        repository = 'iamwavecut/ninfer-all'
        self.assertEqual(CACHE.merged_branch('Merge pull request #330 from iamwavecut/dev/launch-config', repository),
                         'refs/heads/dev/launch-config')
        for subject in ('Merge pull request #155 from witherhoard99/master',
                        'fix(launcher): name the missing backend files',
                        'Merge branch master into dev/x'):
            with self.subTest(subject=subject):
                self.assertIsNone(CACHE.merged_branch(subject, repository))

    def test_unreadable_head_commit_names_no_merged_branch_and_warns(self):
        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, redirect_stdout(output), \
                patch.dict(os.environ, {'GIT_CEILING_DIRECTORIES': directory}):
            cwd = os.getcwd()
            os.chdir(directory)
            try:
                self.assertIsNone(CACHE.head_merged_branch())
            finally:
                os.chdir(cwd)
        self.assertIn('::warning::Cannot read the head commit', output.getvalue())

    def test_pack_excludes_mutable_statistics_and_temporary_entries(self):
        (self.source / 'tmp').mkdir()
        for name in ('object', 'stats', 'lock', 'entry.tmp', 'tmp/uncommitted'):
            (self.source / name).write_text(name)
        archive = self.root / 'test.tar'
        self.assertEqual(CACHE.pack(self.source, archive), 1)
        with tarfile.open(archive) as package:
            self.assertEqual(package.getnames(), ['object'])


class RetentionTests(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.source = Path(self.work.name) / 'objects'
        self.source.mkdir()
        (self.source / 'object').write_text('object')
        self.bucket = Bucket()

    def save(self, family, ref):
        CACHE.save(self.bucket, self.source, family, ref)

    def advance(self, days):
        self.bucket.now += timedelta(days=days)

    def prune(self, *live_refs, dry_run=False):
        return CACHE.prune(self.bucket, live_refs, dry_run, self.bucket.now.timestamp())

    def kept(self, family, ref):
        return any(p.startswith(CACHE.prefix(family, ref) + '/') for p in self.bucket.files)

    def test_merged_deleted_and_idle_branches_lose_their_caches(self):
        self.save('linux', 'refs/heads/idle')
        self.advance(4)
        for ref in ('refs/heads/master', 'refs/heads/live', 'refs/heads/merged'):
            self.save('linux', ref)
        self.prune('refs/heads/live', 'refs/heads/idle')
        self.assertTrue(self.kept('linux', 'refs/heads/master'))
        self.assertTrue(self.kept('linux', 'refs/heads/live'))
        # Master's first build of the merge reads the merged branch's snapshot.
        self.assertTrue(self.kept('linux', 'refs/heads/merged'))
        self.assertFalse(self.kept('linux', 'refs/heads/idle'))
        self.advance(1.1)
        self.save('linux', 'refs/heads/master')
        self.prune('refs/heads/live')
        self.assertFalse(self.kept('linux', 'refs/heads/merged'))
        self.assertTrue(self.kept('linux', 'refs/heads/live'))

    def test_family_without_writes_is_retired_only_while_its_branch_keeps_building(self):
        self.save('linux-host-v2', 'refs/heads/master')
        self.save('windows', 'refs/heads/master')
        self.advance(10)
        self.prune()
        self.assertTrue(self.kept('linux-host-v2', 'refs/heads/master'))
        self.save('linux-shared-v1', 'refs/heads/master')
        self.prune()
        self.assertFalse(self.kept('linux-host-v2', 'refs/heads/master'))
        self.assertFalse(self.kept('windows', 'refs/heads/master'))
        self.assertTrue(self.kept('linux-shared-v1', 'refs/heads/master'))

    def test_earlier_layouts_and_branch_backups_are_removed(self):
        self.bucket.write('github-actions/v1/iamwavecut/ninfer-all/0123/linux/0.tar', b'old layout')
        backup = f"{CACHE.AREAS['native-artifacts']}/{CACHE.ref_key('refs/heads/merged')}/ninfer-linux-x64-cuda13.4/7/COMPLETE.json"
        self.bucket.write(backup, b'{}')
        self.save('linux', 'refs/heads/master')
        self.prune()
        self.assertNotIn(backup, self.bucket.files)
        self.assertIn('github-actions/v1/iamwavecut/ninfer-all/0123/linux/0.tar', self.bucket.files)
        self.advance(4)
        self.prune()
        self.assertNotIn('github-actions/v1/iamwavecut/ninfer-all/0123/linux/0.tar', self.bucket.files)
        self.assertTrue(self.kept('linux', 'refs/heads/master'))

    def test_dry_run_deletes_nothing(self):
        self.save('linux', 'refs/heads/merged')
        self.advance(2)
        before = self.bucket.files.copy()
        self.assertEqual(len(self.prune(dry_run=True)), 2)
        self.assertEqual(self.bucket.files, before)

    def test_artifact_backup_root_belongs_to_retention(self):
        spec = importlib.util.spec_from_file_location(
            'artifact_backup', Path(__file__).resolve().parents[2] / '.github/scripts/archive-linux-artifact.py')
        backup = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(backup)
        self.assertEqual(backup.ROOT, CACHE.AREAS['native-artifacts'])

    def test_only_master_retention_workflow_deletes_in_actions(self):
        workflow = 'iamwavecut/ninfer-all/.github/workflows/build-cache-retention.yml'
        with patch.dict(os.environ, GITHUB_ACTIONS='true', GITHUB_WORKFLOW_REF=workflow + '@refs/heads/master'):
            CACHE.require_retention_job()
        for ref in (workflow + '@refs/heads/other',
                    'iamwavecut/ninfer-all/.github/workflows/ci.yml@refs/heads/master'):
            with patch.dict(os.environ, GITHUB_ACTIONS='true', GITHUB_WORKFLOW_REF=ref), \
                    self.assertRaises(ValueError):
                CACHE.require_retention_job()


class CacheOwnershipTests(unittest.TestCase):
    def setUp(self):
        self.env = dict(GITHUB_REF='refs/heads/master', GITHUB_RUN_ID='200',
                        GITHUB_JOB='linux', GITHUB_EVENT_NAME='push',
                        GITHUB_WORKFLOW_REF='iamwavecut/ninfer-all/.github/workflows/native-release.yml@refs/heads/master')

    def test_only_native_builds_can_write_their_own_families(self):
        with patch.dict(os.environ, self.env, clear=True):
            CACHE.require_cache_writer('native-linux-cuda-13.4.3', 'refs/heads/master')
            for override in (dict(GITHUB_JOB='cuda'), dict(GITHUB_EVENT_NAME='pull_request'),
                             dict(GITHUB_WORKFLOW_REF='iamwavecut/ninfer-all/.github/workflows/ci.yml@refs/heads/master')):
                with patch.dict(os.environ, override), self.assertRaises(ValueError):
                    CACHE.require_cache_writer('native-linux-cuda-13.4.3', 'refs/heads/master')
            with self.assertRaises(ValueError):
                CACHE.require_cache_writer('native-windows-cuda-13.4.3', 'refs/heads/master')
            with patch.dict(os.environ, GITHUB_JOB='windows'):
                CACHE.require_cache_writer('vcpkg-baseline', 'refs/heads/master')

    def test_ci_writes_only_its_own_family_from_master_pushes(self):
        env = dict(self.env, GITHUB_JOB='cuda',
                   GITHUB_WORKFLOW_REF='iamwavecut/ninfer-all/.github/workflows/ci.yml@refs/heads/master')
        with patch.dict(os.environ, env, clear=True):
            CACHE.require_cache_writer('ci-linux-cuda-13.4.3-v1', 'refs/heads/master')
            for family, ref, override in (
                    ('native-linux-cuda-13.4.3-shared-v1', 'refs/heads/master', {}),
                    ('ci-linux-cuda-13.4.3-v1', 'refs/heads/dev/x', {}),
                    ('ci-linux-cuda-13.4.3-v1', 'refs/heads/master', dict(GITHUB_EVENT_NAME='pull_request')),
                    ('ci-linux-cuda-13.4.3-v1', 'refs/heads/master', dict(GITHUB_JOB='windows-msvc'))):
                with self.subTest(family=family, ref=ref, override=override), \
                        patch.dict(os.environ, override), self.assertRaises(ValueError):
                    CACHE.require_cache_writer(family, ref)
        with patch.dict(os.environ, self.env, clear=True), self.assertRaises(ValueError):
            CACHE.require_cache_writer('ci-linux-cuda-13.4.3-v1', 'refs/heads/master')

    def test_restore_records_a_baseline_even_when_the_optional_reader_starts_cold(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            (root / 'objects').mkdir()
            (root / 'objects/kept').write_text('local entry')
            command = [sys.executable, SPEC.origin, 'restore', '--directory', str(root / 'objects'),
                       '--family', 'native-linux-cuda-test', '--source-ref', 'refs/heads/master',
                       '--optional', '--baseline', str(root / 'baseline.tsv')]
            env = os.environ | self.env
            env.pop('HF_TOKEN', None)
            result = subprocess.run(command, env=env, capture_output=True, text=True)
            self.assertEqual(result.returncode, 0, result.stderr)
            self.assertEqual(CACHE.read_baseline(root / 'baseline.tsv'), {'kept': 11})

    def test_bootstrap_can_write_only_its_linux_shared_cache(self):
        env = dict(self.env, GITHUB_EVENT_NAME='workflow_dispatch',
                   GITHUB_WORKFLOW_REF='iamwavecut/ninfer-all/.github/workflows/linux-cache-bootstrap.yml@refs/heads/build/windows-release')
        with patch.dict(os.environ, env, clear=True):
            CACHE.require_cache_writer('native-linux-cuda-13.4.3-shared-v1', env['GITHUB_REF'])
            for family in ('native-linux-cuda-13.4.3-host-v2', 'native-windows-cuda-13.4.3', 'vcpkg-test'):
                with self.assertRaises(ValueError):
                    CACHE.require_cache_writer(family, env['GITHUB_REF'])
            with patch.dict(os.environ, GITHUB_EVENT_NAME='pull_request'), self.assertRaises(ValueError):
                CACHE.require_cache_writer('native-linux-cuda-13.4.3-shared-v1', env['GITHUB_REF'])

    def test_older_rerun_cannot_replace_newer_native_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'objects'
            source.mkdir()
            (source / 'object').write_text('new build')
            bucket = Bucket()
            with patch.dict(os.environ, GITHUB_RUN_ID='200'):
                CACHE.save(bucket, source, 'native-linux-cuda-test', 'refs/heads/master')
            original = bucket.files.copy()
            (source / 'object').write_text('old rerun')
            with patch.dict(os.environ, GITHUB_RUN_ID='100'):
                self.assertFalse(CACHE.save(bucket, source, 'native-linux-cuda-test', 'refs/heads/master'))
            self.assertEqual(bucket.files, original)

    def test_optional_reader_without_secret_can_start_cold_but_writer_cannot(self):
        with tempfile.TemporaryDirectory() as directory:
            command = [sys.executable, SPEC.origin, 'restore', '--directory', directory,
                       '--family', 'native-linux-cuda-test', '--source-ref', 'refs/heads/master']
            env = os.environ | self.env
            env.pop('HF_TOKEN', None)
            optional = subprocess.run(command + ['--optional'], env=env, capture_output=True, text=True)
            self.assertEqual(optional.returncode, 0, optional.stderr)
            self.assertIn('::warning::Native cache unavailable', optional.stdout)
            required = subprocess.run(command + ['--required'], env=env, capture_output=True)
            self.assertNotEqual(required.returncode, 0)

    def test_optional_reader_transport_failure_warns_without_leaking_url(self):
        class Unavailable:
            def bucket_info(self, *_args):
                raise ConnectionError('https://secret-signed-url.invalid')

        output = io.StringIO()
        with tempfile.TemporaryDirectory() as directory, \
                patch.dict(os.environ, self.env | {'HF_TOKEN': 'test-only'}), \
                patch.dict(sys.modules, huggingface_hub=SimpleNamespace(HfApi=Unavailable)), \
                patch.object(sys, 'argv', ['cache', 'restore', '--directory', directory,
                                          '--family', 'native-linux-cuda-test', '--optional']), \
                redirect_stdout(output):
            CACHE.main()
        self.assertIn('ConnectionError', output.getvalue())
        self.assertNotIn('secret-signed-url', output.getvalue())

    def test_pr_reads_target_branch_without_modifying_bucket(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / 'producer'
            source.mkdir()
            (source / 'object').write_text('release object')
            bucket = Bucket()
            family = 'native-linux-cuda-test'
            CACHE.save(bucket, source, family, 'refs/heads/master')
            snapshot = bucket.files.copy()
            bucket.bucket_info = lambda *_args: SimpleNamespace(private=True)
            with patch.dict(os.environ, self.env | {'GITHUB_REF': 'refs/pull/1/merge', 'HF_TOKEN': 'test-only'}), \
                    patch.dict(sys.modules, huggingface_hub=SimpleNamespace(HfApi=lambda: bucket)), \
                    patch.object(sys, 'argv', ['cache', 'restore', '--directory', str(root / 'reader'),
                                              '--family', family, '--source-ref', 'refs/heads/master']):
                CACHE.main()
            self.assertEqual((root / 'reader/object').read_text(), 'release object')
            self.assertEqual(bucket.files, snapshot)
