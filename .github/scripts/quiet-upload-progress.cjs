// Preloaded into actions/upload-artifact through NODE_OPTIONS (--require). The action logs
// "Uploaded bytes N" for every 8 MiB block it sends: 744 lines for the 6.2 GB universal Linux
// package of Native release 43396928446. This keeps one such line per 30 seconds; the newest
// held line is printed before the action's next message and at exit, so the total still shows.
'use strict';

const INTERVAL_MS = 30000;
const PROGRESS = /^Uploaded bytes \d+\r?\n$/;
const write = process.stdout.write.bind(process.stdout);
let shownAt = 0;
let held = null;

function release() {
  if (held !== null) {
    const line = held;
    held = null;
    write(line);
  }
}

process.stdout.write = function (chunk, encoding, callback) {
  const text = typeof chunk === 'string' ? chunk : Buffer.from(chunk).toString('utf8');
  const done = typeof encoding === 'function' ? encoding : callback;
  if (PROGRESS.test(text)) {
    const now = Date.now();
    if (now - shownAt >= INTERVAL_MS) {
      shownAt = now;
      held = null;
      return write(chunk, encoding, callback);
    }
    held = text;
    if (typeof done === 'function') {
      process.nextTick(done);
    }
    return true;
  }
  release();
  return write(chunk, encoding, callback);
};

process.on('exit', release);
