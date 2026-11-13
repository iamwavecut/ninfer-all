"""HF bucket access for CI jobs: each call retried on transient failures, progress bars throttled.

Every bucket call gets its own budget of one try and RETRIES retries, so a long transfer that
succeeds on its fourth attempt leaves the next call five retries again. A transient failure is a
transport error (dropped connection, protocol error, timeout) or an HTTP 408, 425, 429 or 5xx
answer; a missing file, a refused credential or another 4xx answer fails at once.

huggingface_hub draws Xet transfers as tqdm bars that it refreshes several times a second. Without
a terminal each refresh is a line, plus empty lines from cursor movement: the step that backed up
the 6.2 GB universal Linux package of Native release 43396928446 logged 1,605 lines. ProgressLog
shows each bar's newest state at most once per interval and once more at the end.
"""
import atexit
import io
import re
import sys
import threading
import time

RETRIES = 5
DELAYS = (5, 10, 20, 40, 60)
RETRY_STATUSES = (408, 425, 429)
# Local and programming errors; a transport error is none of these.
PERMANENT = (FileNotFoundError, NotADirectoryError, IsADirectoryError, PermissionError, ValueError,
             TypeError, KeyError, AttributeError)
ESCAPE = re.compile(r'\x1b\[[0-9;?]*[A-Za-z]')
# tqdm's left part: "<description>: <percent>%|", the percent missing while the total is unknown.
BAR = re.compile(r'^(?P<name>[^|]*?):\s*(?:\d+%)?\|')


def transient(error):
    status = getattr(getattr(error, 'response', None), 'status_code', None)
    if isinstance(status, int):
        return status in RETRY_STATUSES or status >= 500
    if 'NotFound' in type(error).__name__:
        return False
    return not isinstance(error, PERMANENT)


class RetryingApi:
    """Wrap an HfApi: each method call retries transient failures with its own budget."""

    def __init__(self, api, retries=RETRIES, delays=DELAYS, sleep=None):
        self._api, self._retries, self._delays, self._sleep = api, retries, delays, sleep

    def __getattr__(self, name):
        target = getattr(self._api, name)
        if not callable(target):
            return target

        def call(*args, **kwargs):
            for attempt in range(self._retries + 1):
                try:
                    result = target(*args, **kwargs)
                    # Listings are lazy: a failure while reading them belongs to this call.
                    return list(result) if hasattr(result, '__next__') else result
                except Exception as error:
                    if attempt == self._retries or not transient(error):
                        raise
                    delay = self._delays[min(attempt, len(self._delays) - 1)]
                    # The exception text can hold signed URLs; its type is enough here.
                    print(f'::warning::HF {name} failed ({type(error).__name__}); retry {attempt + 1} of '
                          f'{self._retries} in {delay} s', flush=True)
                    (self._sleep or time.sleep)(delay)
        return call


class ProgressLog(io.TextIOBase):
    """Pass text through, but print each progress bar's newest state at most once per interval."""

    def __init__(self, stream, interval=30.0, clock=time.monotonic):
        self._stream, self._interval, self._clock = stream, interval, clock
        self._pending = ''
        self._bars = {}
        self._changed = set()
        self._due = 0.0
        self._lock = threading.Lock()

    @property
    def encoding(self):
        return getattr(self._stream, 'encoding', 'utf-8')

    def writable(self):
        return True

    def isatty(self):
        return False

    def fileno(self):
        return self._stream.fileno()

    def write(self, text):
        with self._lock:
            *segments, self._pending = re.split(r'[\r\n]', self._pending + text)
            for segment in segments:
                self._take(segment)
            if self._changed and self._clock() >= self._due:
                self._emit()
        return len(text)

    def _take(self, segment):
        segment = ESCAPE.sub('', segment).rstrip()
        if not segment.strip():
            return
        match = BAR.match(segment)
        if match is None:
            self._stream.write(segment + '\n')
            return
        # "Processing Files (3 / 6)" and "[+ 4 files]" count completed files: one bar each.
        name = re.sub(r'\(\d+ / \d+\)|\[\+ \d+ files\]', '#', match['name'].strip())
        self._bars[name] = segment
        self._changed.add(name)

    def _emit(self):
        for name, segment in self._bars.items():
            if name in self._changed:
                self._stream.write(segment + '\n')
        self._changed.clear()
        self._stream.flush()
        self._due = self._clock() + self._interval

    def flush(self):
        with self._lock:
            self._stream.flush()

    def close(self):
        with self._lock:
            if self._pending:
                self._take(self._pending)
                self._pending = ''
            if self._changed:
                self._emit()
            self._stream.flush()


def throttle_progress(interval=30.0):
    """Route stderr, where tqdm draws, through a ProgressLog unless it is a terminal."""
    if sys.stderr is None or sys.stderr.isatty() or isinstance(sys.stderr, ProgressLog):
        return
    sys.stderr = ProgressLog(sys.stderr, interval)
    atexit.register(sys.stderr.close)


def open_api():
    """An HfApi whose calls retry transient failures."""
    from huggingface_hub import HfApi
    return RetryingApi(HfApi())
