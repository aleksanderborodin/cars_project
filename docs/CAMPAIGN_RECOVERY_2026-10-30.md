# Frozen comparison recovery after disk exhaustion

Updated 2026-10-30. [Paper ledger](PAPER_CAMPAIGN_2026-10-30.md) · [Status](STATUS.md)

The original queue stopped while saving episode 59 of seed1 search7: the system
filesystem reported `OSError: [Errno 28] No space left on device`. Its last valid
status still says evaluating because even the status/error writes ran out of space.
Both original PIDs are dead. Eighteen complete condition panels and 58 complete
episodes of the interrupted condition are preserved; no result was dropped based
on performance. The partial next NPZ and empty error/status files remain evidence
of the interruption, not complete episodes.

`uv cache prune` removed unreachable cache entries and reported 5.4 GiB logical
size, but usable root space was still only about 4.6 MB. No experiment artifacts
were removed. The assistant therefore stages recovery on the already mounted
secondary volume, with approximately 16.1 GB free, under:

`/media/sasha/72AECF44AECEFF99/cars_project_recovery_20261030`

This storage/recovery choice follows the user's ongoing authorization to continue
the paper experiments. It is an assistant infrastructure decision, not a new
scientific method or information privilege.

## What is preserved and checked

- Original method settings, selected actors, 200-development-track panel,
  partitions, renderer and analysis are unchanged. All 69 frozen source hashes
  and the original scientific declaration must match before recovery.
- All eighteen completed panels are reused only after config/metric hashes and
  exact development-manifest order are validated. Their paths in the recovered
  ledger are absolute; the original files are not overwritten.
- Seed1 search7 is rerun in a new directory, with a distinct recovery suffix.
  After its full 200 tracks, all 58 previously complete observation/action/reward/
  termination traces must match exactly. Latency/path metadata is excluded from
  equality. Failure stops the controller before subsequent search conditions.
- The failed attempt is charged conservatively: **56,488 native frames and
  7,908,320 internal physics ticks** are upper bounds, including one possibly
  completed but unlogged final episode. These bounds are not reported as exact
  measurements or additional scored episodes.
- Recovery adds a declared maximum allowance of 200,000 native and 28,000,000
  internal frames to the old caps. Actual new completed-run costs are recorded
  separately. The statistical condition matrix stays three actors by ten methods.
- Free space on the output volume is checked during evaluations, not only between
  them; stop below 2 GB. Temporary/cache writes are routed to that volume as well.
  No training, calibration/final evaluation or private monitor is introduced.

## Recovery interface and status

The [declaration](../configs/campaigns/paper-search-recovery-2026-10-30.json) pins
the controller and source. `scripts/recover_paper_search.py --campaign PATH --check`
verifies completed panels, declaration, code and storage without running a car.
The runnable controller/source/reporter and all future outputs are stored on the
secondary volume. Original failed queue files remain unchanged.

Live recovery status is `queue/status.json` under the workspace above. The earlier
root-filesystem status is historical and must not be treated as a heartbeat.
The final report is written to the recovery workspace's `reports/` directory;
its public summary can subsequently be copied into the repository with provenance.

The recovery preflight passed. Full software checks run with temporary storage on
the secondary volume before launch; their log/result are retained in `queue/`.
The full CPU suite passed 133 tests with one CUDA-only skip. Recovery launched at
**2026-10-30 15:15:55 UTC (18:15:55 Moscow)** after those checks. The full software
suite completed before driving restarted. Earlier software-overlap caveats remain attached to
the interrupted attempt, which is not substituted for a complete scored panel.
The first three recovered episodes already match the originals exactly; the full
58-episode gate remains pending and is enforced after the full rerun.

## ETA and interpretation

The first 58 search7 episodes took approximately 29.4 minutes. Scaling that pace,
using the measured search21/search7 cost ratio and allowing for the unsuccessful
actor's episode lengths gives approximately **12–15 hours after restart** for
the remaining fixed comparison batch. This includes rerunning the interrupted
condition; recovery/preparation time is additional. It is an estimate, not a
deadline guarantee. The batch still does not complete strong external-baseline,
component-study or final-paper requirements.
