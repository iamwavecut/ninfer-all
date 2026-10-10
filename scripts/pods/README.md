# Pod routines

Use Python 3.11: `python3.11 scripts/pods/harness.py COMMAND`.
Provider credentials stay in the provider CLI's local configuration. Source snapshots include
local edits; no push is required. All remote jobs run in named tmux sessions and retain their
script, source identity, output and exit code under `/workspace/ninfer-work/jobs/`.
Deployment preserves timestamps for unchanged contents and timestamps changed files on arrival,
so a local edit made during an older remote build cannot leave its stale object selected by Ninja.
`deploy --snapshot DIRECTORY` reuses a saved `source.tar.gz`/`source.json` pair across hosts.
`NINFER_POD_STATE=.local/pods/a6000/pod.json` selects another rental record and its result directory;
the guard inherits that exact path. The default record and its retained files remain separate.

```sh
python3.11 scripts/pods/harness.py inventory
python3.11 scripts/pods/harness.py offers 'gpu_name=RTX_3090 num_gpus=2 verified=true rentable=true'
python3.11 scripts/pods/harness.py rent OFFER --hours 3 --disk 180 --image CUDA_DEVEL_IMAGE
python3.11 scripts/pods/harness.py wait
python3.11 scripts/pods/harness.py probe
python3.11 scripts/pods/harness.py deploy
python3.11 scripts/pods/harness.py start bootstrap scripts/pods/bootstrap.sh
python3.11 scripts/pods/harness.py start build scripts/pods/build.sh --after bootstrap
python3.11 scripts/pods/harness.py start models scripts/pods/models.sh --after bootstrap
python3.11 scripts/pods/harness.py start checks scripts/pods/engine_checks.sh --after build --gpu
python3.11 scripts/pods/harness.py start ngram-checks scripts/pods/ngram_checks.sh --after build --gpu
python3.11 scripts/pods/harness.py start host-checks scripts/pods/host_checks.sh --after ngram-checks --gpu
python3.11 scripts/pods/harness.py watch build-observer
python3.11 scripts/pods/harness.py status build
python3.11 scripts/pods/harness.py status ngram-checks --file profile-heldout.txt --lines 30
python3.11 scripts/pods/harness.py collect --job checks
python3.11 scripts/pods/harness.py collect
python3.11 scripts/pods/harness.py destroy
```

Install tmux on a fresh image before dispatching jobs. `attach --host HOST --port PORT` also works
with an existing SSH host, including a RunPod created with `~/runpod/pod-up.sh`; attached hosts
cannot be destroyed by this harness. Vast endpoints are refreshed before each operation.
Rentals inject the public SSH key with their startup script. `wait` checks SSH for at most ten
minutes and installs tmux if needed; `discard-empty` removes only an undeployed harness rental
after failed provisioning. `NINFER_POD_SSH_KEY` selects another private-key file locally.

Each job name is unique. `stop-job NAME` interrupts only that tmux session and retains its log;
`start NAME SCRIPT --retry` archives the finished attempt before retrying. The wrapper runs the
routine in a child shell, so a routine's `exec` still produces a completion marker. Repeat `--after`
to require several jobs; a failed dependency or a different source snapshot blocks execution. `--gpu` queues jobs under
one host lock, so independent model checks can run asynchronously without competing for VRAM;
the `executing` timestamp records admission after the lock. `--input FILE` stages a
helper under `$NINFER_JOB_DIR/inputs` and records its SHA256; it can be repeated. Use it to repair
an independent routine without changing the source snapshot during a build. Do not deploy while
jobs are running. `status` distinguishes a live job, a live dependency wait, and an orphaned job
whose tmux session disappeared without an exit marker. A satellite samples host/process/GPU metrics for
two minutes, then exits. Process command arguments and environment are deliberately excluded.

A local guard stops the exact rental at `--hours`, preserving its files. On macOS, launchd owns
the guard independently of the app's tool processes and caffeinate prevents idle sleep. The first
detached-process guard disappeared during an app interruption on October 8; its empty log did not
establish the cause. `guard --hours 1` replaces the deadline for the attached rental and logs its
admission immediately. Sleep prevention does not survive shutdown or logout; this is not a
provider-side spending cap. Collect results before destroying the finished ephemeral pod;
`collect` requires finished jobs and includes their logs, the source snapshot, conversion report,
hot-row profile and the 110 MB layer-12 CPU fixture when present; it excludes full model weights.
`collect --job NAME` downloads one finished job while other jobs may continue; it does not mark
the entire rental collected or permit teardown. Each collection uses its own remote archive, so
collecting a second receipt cannot truncate a large archive still being downloaded.
A new job invalidates the collection marker. Explicitly copy any full model that must
be retained before teardown. Local state and archives live in the ignored `.local/pods/` directory.

`stop` requires all jobs to be terminal and confirms the provider's stopped state, preserving
files while local implementation continues. `resume --hours 0.5` requests compute again and
rearms the guard; follow it with `wait` to confirm SSH/tmux readiness. Retained disk storage may
still be charged by the provider while compute is stopped.
Both explicit stop and the guard require `actual_status=exited/stopped` and
`intended_status=stopped`. A failed or queued resume may still have `actual_status=exited`
while its desired state is running; it must receive a stop request before the guard is removed.
An October 8 inventory found an old two-GPU rental running after the earlier one-field
confirmation. The corrected condition and queued-start regression tests prevent that false
confirmation. Provider uptime is not an invoice or a timestamped utilization record.
Stopping does not reserve the GPUs: a later start can fail because another tenant uses the
capacity. The installed Vast CLI prints some rejections with exit status zero; the harness
requires a success acknowledgment and reports a redacted rejection category. A guard with an
obsolete deadline cannot stop a newer rental stage.

`copy-from SOURCE_STATE build|models|jobs/runtime-package` requests a provider transfer from a retained rental;
`copy-status` checks the provider marker and delivered files. An acknowledgment is not completion.
The October 8 transfer to A6000 left `build` and `models` inaccessible with `ESTALE` while the
container was running. Canceling with `cancel-copy`, then `reboot` after all jobs finished, restored
access. The fallback `restore_slice_inputs.sh` downloads pinned HF inputs, verifies their digests,
and preserves the private native artifact's revision. It takes `restore_slice_inputs.py` and the
retained `model-inputs.json` as job inputs; stage HF authentication first. It omits the unused table.

## Compiler cache, A/B runs and the residency check

`bootstrap.sh` installs ccache and `build.sh` compiles through it. The cache lives in
`/workspace/ccache` and travels between rentals through the private bucket `WaveCut/ninfer-cache`
(`ccache/cuda<release>-sm<arch>-<distribution>-<machine>.tar`): `build.sh` pulls it into a new pod
before the first compile and pushes it after every build (`NINFER_CCACHE_PUSH=0` skips the push).
Both steps need the credential `hf_archive.py --stage-auth` leaves at `/run/ninfer-hf/token`;
without it, or without a remote cache, the build compiles cold and says so. The Hub stores only the
chunks that changed since the last push. `build.sh` prints the build's seconds and ccache's hit
counts.

```sh
uv run --python 3.11 --with huggingface_hub python scripts/pods/hf_archive.py --stage-auth
python3.11 scripts/pods/harness.py start bootstrap scripts/pods/bootstrap.sh
python3.11 scripts/pods/harness.py start build scripts/pods/build.sh --after bootstrap
```

`serve_ab.py PLAN.json --out DIR` runs ninfer-serve configurations interleaved: round r starts at
configuration r mod n, each start passes over the prompts `repeat` times, and every request is
greedy with EOS ignored. It records decode and prefill rates, time to first token, the expert
cache's hits and bytes moved, and each answer's digest, and writes a per-configuration summary with
ranges and the change against the first configuration. Servers run in their own process group and
are ended on exit or SIGTERM, so `stop-job` leaves none behind. Stage the plan and any prompt files
with `--input`.

`residency_check.sh` (MODEL and TABLE in the environment) scores the first 40 KB of the WikiText
stream with host and with disk experts and fails unless the two agree exactly. Run it after any
change to the expert cache, the misses, the slot pool or the prefetch.

## N-gram cache qualification

The October 8 release routines reuse the existing public Flash-Next repositories.
`mtp_assemble_local.sh` attaches verified donor objects on the GPU pod, preserving the existing
text, Vision, tokenizer and table objects. `mtp_qualify.sh` checks host/disk MTP repetition;
`mtp_vision.sh` checks the existing media fallback and subsequent text drafting.
`mtp_publish.sh` requires both qualifications and verifies public sizes, hashes and metadata.
Each publication has a separate credential file to avoid interference with concurrent jobs.

`ngram_corpus_build.py` samples pinned multilingual/prose/code sources with document/repository
splits. `ngram_profile_broad.py` builds the profile and reports every held-out group.
`ngram_profile_native_check.sh` compares the complete profile bytes with the native C++ profiler.
`ngram_profile_attach.sh` preserves the IQ4 rows and appends the profile resource.
`ngram_profile_engine.sh` checks embedded/explicit profiles against uncached Engine tokens and
requires actual RAM hits. `ngram_profile_publish.sh` updates the existing table repository and
archives the corpus provenance and profile in the private bucket. Cache residency remains opt-in.
These routines retain all samples and limitations. They do not establish an inference speedup.

For the n-gram cache comparison, `ngram_build.sh` saves the pre-cache executable before rebuilding;
`ngram_cpu_checks.sh` checks the core OS queue plus exact reader bytes and failure/eviction
behavior without a CUDA build. It reports a native-queue skip when io_uring is disabled and
still tests the positioned-read fallback. The same routine can run in a local Linux container
with sources mounted read-only at `/workspace/ninfer-work/src` and a writable `$NINFER_JOB_DIR`;
the October 8 native queue test used the existing `orbitar-backend:latest` image, no network,
all capabilities dropped and seccomp unconfined for io_uring. No GPU or CUDA toolkit is needed.
`ngram_row_cache_checks.sh` runs the CLI contracts, a focused Engine repeat test, and one
baseline/cache pair. `ngram_compare.sh` resumes only that pair if the Engine checks already passed.
`compare_ngram_cache.py ENGINE_JOB COMPARISON_JOB` reports both without a stable-throughput claim.
`ngram_default_check.sh` checks that an unspecified disk budget leaves the row cache disabled.
`build_space.sh` removes this rental's download caches and compresses the completed A16 baseline
when disk space is tight. It preserves model files and the Hugging Face upload resume state.
The build runner puts temporary compiler files in tmpfs and uses fatbinary compression `balance`;
model storage and compiler scratch exhausted the disk on the first cache build. Focused wrappers
build only their named targets; an existing serving executable is not evidence of a new serving build.

`ngram_profile_checks.sh` runs the CPU reader and profile-framing checks without CUDA.
The converter embeds `ngram.hot` as a resource only in artifacts storing table rows; the
C++ component and Python writer-interoperability tests cover that resource in the next
combined build. `hc_build.sh` builds the CLI, server, tests and benchmark bundle;
`hc_oracles.sh` runs the HC, reader and draft/verify checks. `hc_checks.sh` resumes the
draft/verify check and paired HC benchmark after the other two oracles have already passed.
`fp8_table_engine_checks.sh` uses explicit retained FP8-table/native-Q2 companion paths,
direct I/O and the public Engine's hybrid repeat/cancel/profile test.

## Layer slice qualification

`layer_slice_build.sh` builds the CLI, server and test bundle. `layer_slice_conversion.sh` converts source
blocks 3..5 (end excluded) from the retained native-Q2 inputs and compares every selected physical
object, binding range, activation use and frontend resource with the previously qualified full
artifact. `layer_slice_checks.sh` then checks the public Engine's fixed-mode repetition and its
absence of table reads. `layer_slice_ple_conversion.sh` checks source blocks 1..4, including PLE
rebased to local block zero; `layer_slice_ple_checks.sh` checks its table reads and missing-companion
refusal. The slices are debug artifacts and must be archived privately with that limitation.
`standalone_table.sh` downloads the pinned IQ4_NL shard and converts it to
`flash-next-iq4-table.ninfer`; its stored rows must hash to the source-table digest. It takes
`standalone_table.py` and `model-inputs.json` as job inputs. Building and conversion may run together
when the host has enough memory and disk space; the 180 GiB rental needed them serialized.

## Pipeline qualification

`pipeline_single.sh` and `pipeline_split.sh` use the public Engine long-context fixture on one or
two devices, with the same native-Q2 artifact and standalone IQ4 table companion. The opt-in
`NINFER_FLASH_NEXT_PIPELINE_REPORT` mode records three requests each at 4,032, 33,024 and 131,008
input tokens, 64 generated tokens, int8 KV, no prefix reuse and no discarded warmup. It retains
exact input/output tokens, content, finish reason and per-phase times in `pipeline.json`.
`compare_pipeline.py SINGLE SPLIT` rejects incomplete or incomparable runs, reports every timing
sample and its median, and fails on any fixed-mode or cross-device output difference. Optional
`--eos-token 248046` applies the accepted M7 boundary: cross-device tokens must match through the
first EOS, while every fixed-mode repeat must still match in full. Raw full-output differences
remain in the report separately from acceptance. Without a configured EOS in the output, the
complete-output comparison applies. Decode rate uses the 63 intervals after the first generated
token. These routines are not measurement evidence until both reports have completed.

For the same Linux/CUDA image, `runtime_package.sh` packages the already-built test executable,
its SHA-256 and source identity. `bootstrap_runtime.sh` installs only runtime dependencies and
Python 3.11. `runtime_install.sh` accepts that package as an input or from a provider transfer,
checks the binary digest, matching source snapshot and shared libraries, and preserves an existing
identical executable. It avoids rebuilding CUDA code for the second host. `hf_restore.sh` takes
`hf_restore.py` and an explicit `verified.json` archive receipt as inputs; it restores pinned
private revisions and checks each full artifact's size, SHA-256 and conversion identity.

## Private model archives

Retain intermediate `.ninfer` models in private Hugging Face model repositories. Each experiment
uses an explicitly selected repository so a candidate can be kept for production qualification or
removed later. Temporary MTP candidates use the existing private bucket `WaveCut/ninfer-cache`,
under `flash-next-mtp-20261008/`; after GPU qualification the same public model repositories are
updated. Do not create model repositories for temporary transfers. Update `hf_archive_manifest.json` with explicit artifact
paths, repo names, qualifications, limitations and any companion table before archiving.
The archive contains the model, conversion report, input revisions and a model card. It does not
include arbitrary job logs or source checkpoints. Uploading is not production qualification.

```sh
uv run --python 3.11 --with huggingface_hub==1.17.0 python scripts/pods/hf_archive.py --stage-auth
python3.11 scripts/pods/harness.py start hf-archive scripts/pods/hf_archive.sh \
  --input scripts/pods/hf_archive.py --input scripts/pods/hf_archive_manifest.json
python3.11 scripts/pods/harness.py watch hf-archive-observe
python3.11 scripts/pods/harness.py status hf-archive
python3.11 scripts/pods/harness.py collect --job hf-archive
```

Authentication crosses SSH stdin into a mode-600 file under `/run`, outside the job and upload
folders, and is removed when the upload routine exits. Staging uses hard links; large model
payloads are not copied locally. The resumable Hub upload cache remains on the pod. The uploader
refuses an existing public repository, then checks private visibility, final revision, file size,
the upload client's SHA-256 against Hub metadata, and downloaded report bytes. A completed
`verified.json` record is the archive receipt. A failed attempt preserves the model and cache;
stage authentication again and retry the finished job with `--retry`.
Keep model deletion separate from upload; a failed backup does not authorize deleting the source.

For the published Flash-Next MTP updates, `runpod_cpu.py create` provisions a 4-vCPU CPU Pod and
a temporary 160-GB volume. Select `.local/pods/runpod-cpu/pod.json` with `NINFER_POD_STATE`, then
use the same harness to deploy, run `bootstrap_cpu.sh`, stage authentication with
`hf_archive.py --stage-auth`, and run `mtp_release.sh --input AUDIT_JSON`. The audit input is the
pinned inventory from `hf_artifact_audit.py`, which selects NInfer repositories only. Model weights
must remain on the remote Linux Pod; the release script refuses macOS and paths outside
`/workspace`. Each source is hash-checked, its original objects and metadata are preserved, and
only the donor's MTP objects are fetched. Every output object passes readback before bucket upload.
The uploader records the local SHA-256, remote Xet hash and verified size before deleting that variant's
temporary source/output copies. Collect the small job receipts, verify them, then authorize
`runpod_cpu.py destroy` with the local `cleanup-approved.json` receipt. It deletes only this task's
Pod and volume and verifies their absence. The consuming Pod verifies the full SHA-256 before
GPU tests. Remove only this task's bucket prefix after the public updates are verified. A successful
upload does not establish GPU qualification.

After qualification, `hf_archive.py --manifest MANIFEST --output DIR --refresh-metadata` updates
only existing private model cards and `artifact.json`. It uses local HF authentication, preserves
the model payload, refuses a concurrent revision change and verifies metadata readback plus
unchanged payload size/digest and privacy. Use `--select FILE --only BASENAME` to prepare a
subset from the canonical manifest. This path does not download or reupload model weights.
