# Native recipe workstation profile

Technical profile only; no development driving performance selection.

| Device | Backend | First update (s) | Later mean (s) | Frames/s | Projected 4M (h) | Peak GPU allocated (GiB) |
|---|---|---:|---:|---:|---:|---:|
| cpu | dummy | 30.50 | 28.19 | 145.3 | 7.65 | 0.00 |
| cpu | subproc | 19.53 | 17.25 | 237.5 | 4.68 | 0.00 |
| cuda | dummy | 20.23 | 18.09 | 226.5 | 4.91 | 0.13 |
| cuda | subproc | 9.50 | 7.54 | 543.4 | 2.05 | 0.13 |

Selected: `native_profile_20261029_cuda_subproc_reloadfix`.

All attempts: 73,728 training frames, 8–10 reload frames, 102 reset physics ticks, 18 full updates; 5.41 campaign minutes.

Original GPU failures and subsequent technical corrections are retained in summary.json.

- Three rollouts per setup, one seed; not a driving-competence result
- Full-run projection excludes evaluation/checkpoint overhead and changes in later recurrent batch lengths
- Parent RSS is not total subprocess memory; GPU memory excludes other processes
- Original GPU attempts failed cross-device assertions and remain failed; retry costs are included
- Original failed reload step counters are bounded, not falsely reported as exact
- Unit/integration test work is separate from the campaign ledger
