# Contributing to NInfer-all

NInfer-all welcomes precise problem reports, reproducible performance evidence, and carefully scoped
code contributions. The most useful first contribution is often an Issue that establishes what is
happening and why it matters. A pull request is an implementation of an agreed change, not the
place to ask the maintainer to discover the problem, choose its semantics, or redesign its core
approach.

## Project scope and decisions

NInfer-all is an intentionally focused inference engine. Contributions are evaluated against the
supported product described in [`README.md`](README.md), the public documentation map in
[`docs/README.md`](docs/README.md), and the applicable architecture documents. Broader
compatibility or generality is not a benefit by itself when it adds a product contract or
maintenance surface that the project does not intend to own.

The maintainer makes the final decision about product scope, architecture, and long-term
maintenance. Establishing that a bug is real or that an idea is useful does not imply that a
particular implementation will be accepted. The maintainer may preserve the report or core idea
and implement it independently within the current architecture.

## Start with an Issue

Every bug, performance opportunity, feature request, protocol change, and architecture proposal
must begin with an Issue before implementation starts. Use the Issue to establish:

- the concrete problem or use case;
- whether it occurs within the supported product scope;
- the expected behavior or performance claim;
- the affected contract and ownership boundary; and
- whether an external implementation is appropriate.

Wait for the maintainer to confirm the scope and implementation direction before investing in a
pull request. Opening an Issue does not by itself approve a proposed design. A pull request without
a linked, confirmed Issue may be closed without detailed review.

### Bug reports

A bug report must contain enough information for the maintainer to locate and reason about the
failure. Include, as applicable:

- the exact NInfer commit or release, artifact source, and recipe when using custom weights;
- the GPU, driver, CUDA toolchain, build configuration, and relevant runtime options;
- the complete command, request, or smallest practical reproduction;
- the expected behavior and the observed behavior;
- relevant logs or error output as text;
- the reproduction frequency and any known trigger or boundary condition; and
- checks already performed, including anything that could not be verified.

State clearly when a reproduction is incomplete or when the behavior occurs outside the advertised
product target. An unsupported use case may still motivate a proposal, but it is not a defect in
the supported product contract.

### Performance reports

A performance report must identify the level of the claim: operator, schedule, request phase, or
end-to-end inference. Include the baseline and candidate measurements under comparable conditions,
the exact workload and commands, hardware and toolchain, warmup and repetition method, and a useful
summary of the results.

Account for relevant tradeoffs such as workspace, resident memory, transfer cost, numerical quality,
and effects on other execution paths. End-to-end results can establish an end-to-end observation;
they do not isolate an operator change. A proposed operator optimization should therefore include a
direct operator benchmark and appropriate correctness evidence.

### Feature and architecture proposals

Describe the real supported use case, the limitation of the current behavior, the desired semantics,
and the new contract or maintenance responsibility the project would acquire. Explain why the
existing architecture cannot satisfy the use case. Do not begin a broad implementation until the
maintainer has accepted both the product direction and the proposed ownership model.

## Pull request scope

A pull request is appropriate only after its linked Issue has established the problem and the
maintainer has agreed to its scope and implementation direction. Target pull requests at the
`master` branch.

Each pull request must represent one coherent engineering decision:

- address one confirmed Issue with one consistent design;
- include all implementation, tests, tooling, and active documentation required to close that
  decision correctly;
- keep independent bugs, optimizations, refactors, and product changes in separate pull requests;
- exclude unrelated cleanup and speculative compatibility work; and
- avoid changing public semantics or product scope beyond what was agreed in the Issue.

There is no mechanical line-count or file-count limit. A necessary solution may cross several
files, but every changed part must belong to the same contract and be reviewable and verifiable as
one unit. A large change that combines independent decisions, crosses ownership boundaries without
prior agreement, or requires the maintainer to reconstruct its design is not maintainable and will
not be accepted. Use the Issue, not a draft implementation, to resolve architecture and scope; a
Draft PR is welcome for early measurement and testing once that direction is agreed.

Work ported from a fork or from upstream keeps its author: port it as the original commits or with
the author recorded, name the source repository and commit, and say whether the change is a clean
cherry-pick or adapted to this tree. When the same change is still under review upstream, let it
settle there before porting it.

## Engineering rules

These rules apply to every change, from people and from coding agents. If a change cannot meet a
rule, its pull request names the rule and gives the reason.

### Lossless by default

An optimization must not change model output. Exact transforms, codecs and expert placements stay
exact: device, host and disk residency and the CPU share of expert products give the same results.
Floating-point changes stay within the oracle criterion of each Op
([Op development](docs/maintainer/op-development.md)).

A change that trades accuracy for speed is lossy. Examples: skipped or truncated experts, tokens or
vocabulary; approximate attention or sampling; lower precision than the current default in storage,
accumulation or exchanged partials; early exits. A lossy change needs the maintainer's agreement in
its Issue. It ships as an option that is off by default, and its pull request reports the quality
cost (perplexity, task accuracy) next to the speed. Formats that the user selects, such as
`--kv-dtype` and quantized artifacts, are options of this kind.

After a change to expert residency, slot pools, prefetch or the expert cache, `ninfer-perplexity`
gives the same value on the same text with `--expert-residency host` and with `disk`. Time a prompt
of 5K tokens or more as well as short ones, and compare greedy token ids and logprobs with `master`
on long prompts: a perplexity run does not reach every prompt path.

### Every family

A change to a shared mechanism covers every model family that uses that mechanism: Qwen3.5-core
(dense and MoE releases), Qwen3.8-Flash-Next, Gemma 4, Flint and DeepSeek-V4.1. Shared mechanisms
include KV formats, rotary scaling, prompt chunking, CUDA Graphs, concurrency, speculative decoding,
sampling, expert residency and the CPU share, pipeline stages, tensor parallelism, serving features,
and kernels that several families call.

- Write the change once where the families share code. A speedup that one checkpoint, shape or
  family gets while the others could use it is incomplete.
- A dense change names its MoE counterpart and covers it in the same pull request, and the reverse.
- A family that does not get the change is named in the pull request and in its guide, with the
  reason.
- A new family brings the shared features of the other families: the `--kv-dtype` formats it can
  store, concurrency above one, CUDA Graphs, 2048-token prompt chunks, grouped Q/K/V projections,
  `--tensor-parallel`, YaRN, and the serving features in [docs/serving.md](docs/serving.md).

### Every platform

A change builds and runs on every supported target: `sm_70`, `sm_75`, `sm_80`, `sm_86`, `sm_89` and
`120a`; CUDA 12.x and 13.x (`sm_70` builds only with 12.x, `120a` only with 13.1 or newer except 13.2.0 and 13.2.1); Linux
with GCC and Windows with MSVC ([build system](docs/maintainer/build-system.md),
[Windows build](docs/build-windows.md)).

- A kernel that needs newer instructions or more shared memory is gated by compute capability, and
  the other targets keep a working route. A format, option or artifact that runs on one card class
  only is not a supported feature.
- Routes and tuning constants come from measured [device profiles](docs/device-profiles.md). A
  constant from upstream, another fork or another card is a hypothesis until it is measured on each
  card that it affects.
- Add a profile for a GPU model that has no built-in one: calibrate it at its default power limit
  ([Calibrating by hand](docs/device-profiles.md#calibrating-by-hand)) and add its row to
  [docs/gpus.md](docs/gpus.md).
- Memory budgets and defaults follow the free memory of the card and the host. They do not assume
  one card size: check the smallest card that the change affects.

### Best default

Users get the best configuration without flags. The default is the fastest lossless configuration
measured on the affected families, card classes and KV formats.

- A lossless change that wins on all of them becomes the default in its pull request, with the
  measurements.
- A change that wins on some and loses on others selects itself from measured conditions (device
  profile, request shape, free memory), or stays behind an option until it can.
- A lossy mode is never the default.

### Comparable measurements

- A speed claim covers every affected family and card class, and every KV format that each family
  supports, with at least one point each. One short run per configuration is enough when it
  separates the arms. Report the worst change and every regression next to the best case.
- Both arms of an A/B run on the same host with the same build options, artifact and inputs: the
  same prompt token ids, a unique nonce in each prompt so that the prefix cache cannot reuse a
  shared prefix, interleaved runs, and no other CPU or GPU load on the host.
- A comparison with another engine uses that engine's latest release with its fastest documented
  configuration, on the same host and weights. Record its version and each flag with the reason for
  it, and show its lossy modes as separate rows. A perplexity comparison with llama.cpp uses the
  same windows and KV type ([Perplexity](docs/perplexity.md)).

### Documentation describes the current state

- Documents describe what the project is and does now. They contain no dated logs, run or CI
  histories, retry stories, accounts of what was tried, author host paths, server setups or copies
  of upstream pages. Measurements are reference tables with their conditions. A rejected approach
  gets at most one sentence, where it explains a current choice. Credits go to
  [CONTRIBUTORS.md](CONTRIBUTORS.md); git history keeps the rest.
- A user-visible change updates, in the same pull request, the family guide,
  [docs/cli.md](docs/cli.md) and [docs/serving.md](docs/serving.md) where they apply, the
  executable's `--help`, the README Artifacts table for a new artifact, and What's New when the
  change is a novelty.
- README's What's New lists novelties only. A novelty gives users something they did not have
  before: a new model, family, artifact, platform, option, endpoint, format or mode, or a measured
  speedup of a path that worked. A fix gets no item, whatever its size. A fix repairs a crash, wrong
  output, a failed start, a missed case or a regression; a speedup that wins back a slowdown and an
  option or default that a fix brings along are part of the fix. Guards, refusals, diagnostics,
  internal changes, CI, build, tests, tooling and documentation get no item either. What's New is
  not a chronicle of the day's work: a date without a novelty gets no line.
- Each date with a novelty gets one short line. Each item names the novelty in a few words and
  links inline to the document with its details. Descriptions, mechanisms, measurements and
  hardware belong in the linked documents, not in What's New. A later novelty of the same date adds
  an item to that line or extends an existing item.

### Tests and tooling

- A new kernel gets an independent FP32 or FP64 oracle test at real model shapes and route
  boundaries, for every schedule it can take, and one `compute-sanitizer` racecheck and synccheck
  run ([Op development](docs/maintainer/op-development.md)).
- Tests follow [tests/README.md](tests/README.md). A test that needs a real artifact is skipped when
  the artifact is absent.
- Product modes and tuning are command-line or serve options, or device-profile entries.
  Environment variables only force routes for diagnosis and A/B runs; a mode that users need is
  never an environment variable.
- Commit subjects follow Conventional Commits with lowercase types such as `feat`, `fix`, `perf`,
  `bench`, `test`, `build`, `refactor`, `docs` or `chore`.

## Engineering and verification

Read the active documentation governing the code being changed and preserve its ownership and
semantic contracts. In particular:

- validate exact transformations and formats with exact comparison;
- validate floating-point work against an independent mathematical oracle with a justified
  numerical criterion;
- verify the complete affected transition for stateful behavior;
- exercise relevant supported shapes and execution routes for CUDA changes;
- measure performance at the level being claimed; and
- keep externally observable implementation, schema tests, and documentation consistent.

Use the existing project workflows:

- [`tests/README.md`](tests/README.md) for test organization and commands;
- [`bench/README.md`](bench/README.md) for product and operator benchmarks; and
- [`docs/maintainer/op-development.md`](docs/maintainer/op-development.md) for numerical Op
  development and qualification.

Run the focused checks needed to establish the changed behavior and its material claims. Record the
exact commands and summarize the results. If a relevant check could not be run, state that
limitation and its consequence. Do not claim validation that was not performed.

## Contributor responsibility and AI assistance

The contributor is responsible for every submitted change, regardless of which tools were used to
produce it. Tool choice neither qualifies nor disqualifies a contribution.

The contributor must understand the complete diff, explain the design and its boundary conditions,
review generated output, perform the stated verification, and respond to technical review. Code
that the contributor cannot explain, validate, or maintain is not ready for submission. Maintainer
review is not a substitute for contributor debugging or engineering supervision.

## Pull request description

Use the [PR template](.github/pull_request_template.md) to describe the problem and scope,
implementation, and verification. One PR completes one agreed change, including its necessary
implementation, verification and report; submit independent changes separately.

A pull request description must include:

- the linked Issue and the agreed scope;
- the concrete design and why it fits the current architecture;
- the affected behavior, ownership boundary, or public contract;
- the exact verification commands and summarized results;
- the workload, hardware, toolchain, and methodology for any performance claim; and
- every relevant check that was not run and the resulting limitation.

Choose the verification details relevant to your change:

| Change | Evidence to include |
|---|---|
| Bug fix | The trigger, expected behavior, and verification of the fix. |
| Recipe or conversion method | The source, target representation, conversion command, and which stages were actually verified: conversion, loading, or execution. |
| New Op or shape | Supported inputs, independent numerical verification, and relevant boundaries. Follow [Op development](docs/maintainer/op-development.md). |
| Performance tuning | Correctness evidence and measurements at the claimed scope on the cards it affects: an Op benchmark for an Op claim, a whole request for an end-to-end claim. A route or schedule change states which [device profiles](docs/device-profiles.md) it changes. |
| Runtime or interface change | Reproduction and verification of the affected behavior. |
| Documentation, build, or tooling | Checks relevant to the change, such as links, affected build targets, or tool invocations. |

## Review policy

Review is a limited maintainer resource and is not guaranteed. A pull request may be closed without
a line-by-line review when it:

- was submitted before its problem, scope, or implementation direction was confirmed;
- falls outside the supported product or expands an unapproved contract;
- combines independent changes into an unreviewable implementation;
- contains evident correctness, ownership, lifetime, state, or boundary problems;
- lacks evidence appropriate to its correctness, numerical, or performance claims;
- cannot be explained and supported by its contributor; or
- would require the maintainer to redesign, split, debug, or complete its core implementation.

Closing a pull request does not necessarily reject the underlying report or idea. It means that the
submitted implementation does not meet the project's review and maintenance threshold. The
maintainer may address a confirmed problem through a separate implementation and reference the
original report where appropriate.
