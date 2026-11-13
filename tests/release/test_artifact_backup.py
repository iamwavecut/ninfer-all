"""Verify artifact backup receipts and reject corrupt transfers."""
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import unittest

# The scripts import their sibling hf_client, as when run from .github/scripts.
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / ".github/scripts"))
SPEC = importlib.util.spec_from_file_location(
    "artifact_backup", Path(__file__).resolve().parents[2] / ".github/scripts/archive-linux-artifact.py")
BACKUP = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(BACKUP)


class Bucket:
    def __init__(self, corrupt=False):
        self.files = {}
        self.corrupt = corrupt

    def batch_bucket_files(self, bucket, add=None, delete=None):
        for path, key in add or ():
            self.files[key] = path.read_bytes()
        for key in delete or ():
            del self.files[key]

    def download_bucket_files(self, bucket, files, **kwargs):
        for key, path in files:
            path.write_bytes(b"corrupt" if self.corrupt else self.files[key])

    def list_bucket_tree(self, bucket, prefix, **kwargs):
        return [SimpleNamespace(type="file", path=key) for key in sorted(self.files) if key.startswith(prefix)]


class ArtifactBackupTest(unittest.TestCase):
    def setUp(self):
        self.work = tempfile.TemporaryDirectory()
        self.addCleanup(self.work.cleanup)
        self.root = Path(self.work.name)
        self.revision = "a" * 40
        self.ref = "refs/heads/master"
        self.name = "ninfer-linux-x64-cuda13.4-sm75-80-86-89-120a-" + "a" * 12
        (self.root / self.name).mkdir()
        (self.root / self.name / "BUILD-INFO.json").write_text(json.dumps(
            dict(source_revision=self.revision, platform="linux-x64")))
        self.volume = self.root / (self.name + ".tar.gz.001")
        self.volume.write_bytes(b"archive content")
        self.manifest = self.root / ("SHA256SUMS-" + self.name + ".txt")
        self.manifest.write_text(f"{BACKUP.digest(self.volume)}  {self.volume.name}\n")

    def test_receipt_published_only_after_complete_roundtrip(self):
        api = Bucket()
        BACKUP.archive(api, self.manifest, self.revision, "123", self.ref)
        key = next(key for key in api.files if key.endswith("/COMPLETE.json"))
        receipt = json.loads(api.files[key])
        self.assertEqual(len(receipt["files"]), 3)
        self.assertEqual(receipt["run_id"], "123")
        self.assertEqual(receipt["ref"], self.ref)
        for item in receipt["files"]:
            self.assertEqual(len(api.files[receipt["prefix"] + "/" + item["name"]]), item["bytes"])

    def test_universal_linux_archive_is_preserved(self):
        name = 'ninfer-linux-x64-cuda12.9-13.4-' + 'a' * 12
        (self.root / self.name).rename(self.root / name)
        self.volume.rename(self.root / (name + '.tar.gz.001'))
        volume = self.root / (name + '.tar.gz.001')
        manifest = self.root / ('SHA256SUMS-' + name + '.txt')
        manifest.write_text(f'{BACKUP.digest(volume)}  {volume.name}\n')
        api = Bucket()
        BACKUP.archive(api, manifest, self.revision, '123', self.ref)
        self.assertTrue(any(key.endswith('/COMPLETE.json') for key in api.files))

    def test_corrupt_download_never_publishes_completion(self):
        api = Bucket(corrupt=True)
        with self.assertRaisesRegex(ValueError, "round-trip"):
            BACKUP.archive(api, self.manifest, self.revision, "123", self.ref)
        self.assertFalse(any(key.endswith("/COMPLETE.json") for key in api.files))

    def test_failure_reports_hub_status_without_urls(self):
        class Response:
            status_code = 403
            def __init__(self, body): self.body = body
            def json(self):
                if self.body is None: raise ValueError("not JSON")
                return self.body
        class HubError(Exception):
            def __init__(self, body): self.response = Response(body)
        quota = HubError({"error": "Storage limit reached, see https://hub.example/x?sig=secret"})
        self.assertEqual(BACKUP.failure(quota), "HubError, HTTP 403: Storage limit reached, see <url>")
        self.assertEqual(BACKUP.failure(HubError(None)), "HubError, HTTP 403")
        self.assertEqual(BACKUP.failure(ValueError("https://hub.example/x")), "ValueError")

    def test_wrong_revision_and_local_corruption_are_not_uploaded(self):
        api = Bucket()
        with self.assertRaises(ValueError):
            BACKUP.archive(api, self.manifest, "b" * 40, "123", self.ref)
        self.volume.write_bytes(b"truncated")
        with self.assertRaises(ValueError):
            BACKUP.archive(api, self.manifest, self.revision, "123", self.ref)
        self.assertEqual(api.files, {})

    def test_newest_complete_backup_replaces_older_ones_of_its_kind(self):
        api = Bucket()
        BACKUP.archive(api, self.manifest, self.revision, "100", self.ref)
        BACKUP.archive(api, self.manifest, self.revision, "300", "refs/heads/other")
        kind = next(key for key in api.files if "/100/" in key).rsplit("/100/", 1)[0]
        api.files[f"{kind}/400/partial.tar.gz.001"] = b"upload of a newer run in progress"
        BACKUP.archive(api, self.manifest, self.revision, "200", self.ref)
        # A rerun of an older run must not replace the newer backup.
        BACKUP.archive(api, self.manifest, self.revision, "150", self.ref)
        runs = {key[len(kind) + 1:].split("/")[0] for key in api.files if key.startswith(kind + "/")}
        self.assertEqual(runs, {"200", "400"})
        self.assertTrue(any("/300/" in key for key in api.files))
