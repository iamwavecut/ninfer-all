"""Check the HF client's per-call retries and its progress-bar throttle without network access."""
import io
from pathlib import Path
import sys
from types import SimpleNamespace
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / '.github/scripts'))
import hf_client  # noqa: E402


class HttpError(Exception):
    def __init__(self, status):
        super().__init__(f'HTTP {status} at https://signed.invalid/url')
        self.response = SimpleNamespace(status_code=status)


class RemoteProtocolError(Exception):
    """Stands in for httpx.RemoteProtocolError, the failure of Native release 43396928446."""


class Flaky:
    def __init__(self, failures):
        self.failures = list(failures)
        self.calls = 0

    def download_bucket_files(self, *_args, **_kwargs):
        self.calls += 1
        if self.failures:
            raise self.failures.pop(0)
        return 'done'

    def list_bucket_tree(self, *_args, **_kwargs):
        self.calls += 1

        def entries():
            yield 'first'
            if self.failures:
                raise self.failures.pop(0)
            yield 'second'
        return entries()


class RetryTests(unittest.TestCase):
    def api(self, failures):
        self.sleeps = []
        self.flaky = Flaky(failures)
        return hf_client.RetryingApi(self.flaky, sleep=self.sleeps.append)

    def test_transient_failures_retry_with_backoff_until_the_call_succeeds(self):
        api = self.api([RemoteProtocolError(), HttpError(503), HttpError(429), ConnectionError()])
        with patch('builtins.print') as printed:
            self.assertEqual(api.download_bucket_files('bucket', files=[]), 'done')
        self.assertEqual(self.flaky.calls, 5)
        self.assertEqual(self.sleeps, [5, 10, 20, 40])
        self.assertNotIn('signed.invalid', ''.join(str(call) for call in printed.call_args_list))

    def test_each_call_has_its_own_budget(self):
        # Four failures, then success: the next call again tolerates five failures.
        api = self.api([RemoteProtocolError()] * 4)
        with patch('builtins.print'):
            api.download_bucket_files('bucket', files=[])
            self.flaky.failures = [RemoteProtocolError()] * 5
            self.assertEqual(api.download_bucket_files('bucket', files=[]), 'done')
        self.assertEqual(self.flaky.calls, 11)

    def test_the_sixth_consecutive_failure_is_raised(self):
        api = self.api([RemoteProtocolError()] * 6)
        with patch('builtins.print'), self.assertRaises(RemoteProtocolError):
            api.download_bucket_files('bucket', files=[])
        self.assertEqual(self.flaky.calls, 6)
        self.assertEqual(self.sleeps, [5, 10, 20, 40, 60])

    def test_permanent_failures_are_raised_at_once(self):
        for error in (HttpError(404), HttpError(401), HttpError(403), FileNotFoundError('missing'),
                      type('RemoteEntryNotFoundError', (Exception,), {})(), ValueError('bad')):
            with self.subTest(error=type(error).__name__):
                api = self.api([error])
                with self.assertRaises(type(error)):
                    api.download_bucket_files('bucket', files=[])
                self.assertEqual(self.flaky.calls, 1)
                self.assertEqual(self.sleeps, [])

    def test_a_failure_while_reading_a_listing_retries_the_listing(self):
        api = self.api([RemoteProtocolError()])
        with patch('builtins.print'):
            self.assertEqual(api.list_bucket_tree('bucket', prefix='x/'), ['first', 'second'])
        self.assertEqual(self.flaky.calls, 2)


class ProgressLogTests(unittest.TestCase):
    def setUp(self):
        self.now = 0.0
        self.out = io.StringIO()
        self.log = hf_client.ProgressLog(self.out, interval=30, clock=lambda: self.now)

    def bar(self, name, percent):
        # tqdm without a terminal: carriage return, bar, cursor movement for the other rows.
        return f'\r  {name}: {percent:3d}%|##        | 1.0MB / 2.0GB\x1b[A\n\n'

    def test_bars_show_at_most_once_per_interval_and_their_last_state_at_close(self):
        for second in range(0, 95):
            self.now = float(second)
            self.log.write(self.bar('...tar.gz.001', second))
            self.log.write(self.bar('...tar.gz.002', second // 2))
        self.log.close()
        lines = self.out.getvalue().splitlines()
        self.assertEqual([line.split(':')[1].split('%')[0].strip() for line in lines],
                         ['0', '30', '14', '60', '29', '90', '44', '94', '47'])
        self.assertTrue(all(line.strip() for line in lines))

    def test_changing_descriptions_are_one_bar(self):
        self.log.write('\rProcessing Files (0 / 6)      :   0%|          | 2.1MB / 6.2GB\n')
        self.now = 10
        self.log.write('\rProcessing Files (3 / 6)      :  50%|#####     | 3.1GB / 6.2GB\n')
        self.log.close()
        self.assertEqual(self.out.getvalue().splitlines(),
                         ['Processing Files (0 / 6)      :   0%|          | 2.1MB / 6.2GB',
                          'Processing Files (3 / 6)      :  50%|#####     | 3.1GB / 6.2GB'])

    def test_other_text_passes_at_once(self):
        self.log.write('::warning::HF download_bucket_files failed (RemoteProtocolError)\n')
        self.log.write('partial')
        self.assertEqual(self.out.getvalue(), '::warning::HF download_bucket_files failed (RemoteProtocolError)\n')
        self.log.close()
        self.assertTrue(self.out.getvalue().endswith('partial\n'))


if __name__ == '__main__':
    unittest.main()
