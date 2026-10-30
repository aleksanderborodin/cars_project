# External baseline interface and synthetic verification

Updated 2026-10-30. [Pinned sources](MRQ_MRSQ_AUDIT_2026-10-30.md) ·
[Comparison roles](COMPARATOR_REVIEW_2026-10-30.md) · [Status](STATUS.md)

**Status:** the local-array interface, source stager and RNG isolation are
implemented. Official MR.Q and MRS.Q passed small CPU synthetic learner/action/
save-reload checks after an explicit MR.Q replay fix. **No external driving
training, evaluation score or competence result exists.** These checks instantiate
no driving environment and consume zero real or internal physics frames.

**Origin:** the user requests continued autonomous progress toward strong paper
comparisons. The assistant implements the engineering boundary and chooses these
readiness checks. MR.Q/MRS.Q are prior work; neither their algorithms nor this
implementation fix is claimed as our scientific contribution.

## Implemented boundary

- `research/external_interface.py`: `LocalVectorCodec` checks the exact allowed
  local fields, shapes, binary road and visibility footprint. It rejects added
  truth fields and nonzero road outside visibility. Float32 road, raw ego values
  and previous physical action flatten into 3,727 values without resizing or
  numeric normalization; the constant footprint is reconstructed exactly.
- Normalized actions map affinely to steering in [−1,1] and gas/brake in [0,1].
  Invalid inputs are rejected, not clipped. Replay must retain normalized actions;
  the previous-action sensor retains actual physical controls. Float32 round-trip
  quantization is checked rather than claimed mathematically exact for all values.
- `ExternalLocalEnv` is owned by the harness, never handed to the actor. It wraps
  only the local train/development interface, preserves reward/termination/timeout,
  emits only native-frame counts in info and keeps cumulative real-frame/decision
  clocks. Calibration and global/native inputs are rejected by this adapter.
- `research/external_rng.py` restores Python, NumPy and Torch random streams around
  validation or planning. CUDA use must be initialized before entry; late CUDA
  initialization fails explicitly. CPU isolation/error paths are tested; actual
  CUDA-state preservation remains a GPU-readiness gate.
- `research/external_sources.py` verifies pinned revisions and every selected file
  before creating a new immutable stage. It retains license, hashes and exact
  changes. It copies no upstream environment runner or result file and refuses
  changed source/overwrites. The original cache remains unchanged.

The codec is a **feedforward history-one preflight representation**, not an
established competitive encoder or a recurrent-PPO-equivalent controller. The
current programmed-search loader still expects SB3 recurrent checkpoints; these
modules do not yet enable search around external actors.

## Observed MR.Q failure and explicit correction

The first official-code synthetic attempt, `runs/external_synthetic/mrq_20261030`,
failed on its first learner call with an out-of-bounds replay sampling index. Its
configuration, source, runner and failure are preserved. A separate
[minimal diagnosis](../reports/checks/mrq_priority_diagnosis_2026-10-30.json)
confirmed the mechanism: under deterministic PyTorch, the upstream `torch.empty`
priority allocation contains NaNs in unused slots. Multiplication by a zero mask
does not remove them; the sampling cumulative sum becomes nonfinite.

The corrected stage initializes unused priorities to zero. Filled-slot updates,
priority formulas and learner hyperparameters remain as declared. This is an
explicit source adaptation; equivalence to undefined/uninitialized upstream
sampling is not claimed. Both algorithms also use explicit CPU observation
storage, avoiding the audited automatic storage-helper problems. These are the
only staged replay changes; source manifests retain exact replacements and hashes.

| Synthetic source check | Outcome |
|---|---|
| MR.Q, initial stage | Failed; preserved and diagnosed |
| MR.Q, defined unused priorities | Passed one learner call, finite/changed encoder, repeat action and exact CPU reload |
| MRS.Q, explicit CPU replay | Passed the same narrow checks, including fixed-RNG stochastic planning |

Each passing check uses sixteen synthetic transitions, reduced networks/batch/
buffer and one learner call, with every override recorded. They do not establish
default-size memory, sustained learning, ring-wrap correctness, full replay/resume
parity, pixel-path correctness, GPU behavior or driving quality. Upstream scalar-
conversion warnings were retained. See [all outcomes and provenance](../reports/checks/external_official_synthetic_2026-10-30.json).

## Software verification and measurement overlap

The first CPU suite passed 132 tests with one CUDA-only skip. The subsequent full
CPU suite passed **133 tests with one CUDA-only skip**, covering the replay
correction and late-CUDA-initialization guard; its result is recorded separately
in [the final check record](../reports/checks/external_interface_v2_2026-10-30.json).
The first [check record](../reports/checks/external_interface_2026-10-30.json) is
retained. This is software verification, including existing small training/reload
fixtures, not new paper-policy training or a throughput profile.

**Assistant scheduling decision:** permit bounded verification at nice19 on one
CPU with CUDA hidden while the immutable search queue continues. Record overlap
instead of claiming resource isolation. This revises the earlier preparation-only
schedule; full driving training and throughput profiles remain deferred. Both
verification windows and the synthetic checks overlapped
`paper_search_20261030_s1_search7`. Keep all raw timing observations. Before an
efficiency claim, perform a separately declared clean timing comparison; this
queue's overlapping latencies are descriptive. No driving method, checkpoint,
candidate distribution, source snapshot or data selection was changed.

## Commands and next gate

```bash
uv run python scripts/stage_external_baseline.py --help
uv run python scripts/stage_external_baseline.py --algorithm MRQ --output runs/external_sources/NEW_MRQ_STAGE
uv run python scripts/stage_external_baseline.py --algorithm MRSQ --output runs/external_sources/NEW_MRSQ_STAGE
```

The default cache/audit paths are recorded in the stager. Official immutable URLs
and hashes in the audit permit refetching missing files. Always choose new output
directories. Staging does not import third-party algorithms or start training.

Next implement the full training/checkpoint/evaluation harness, test replay
boundaries/ring wrap and continuation semantics, and resolve storage/encoder/history
choices. Then freeze a bounded serial driving profile after the current queue and
use measured throughput/memory to declare performance budgets. No performance run
is launched by these commands. Strong baseline, final-test and novelty gates remain
open; private diagnostic outputs and held-out identities remain untouched.
