# Completed conditions: 200-track development snapshot

Created from completed runs on 2026-10-28 while search 21 was still running.
This is an interim campaign report: each included condition completed all 200
tracks, but the declared four-condition comparison is not yet complete.
[Campaign](../../docs/BROAD_DEVELOPMENT_2026-10-28.md) · [Verified data and provenance](analysis.json)

All three use the same frozen recurrent checkpoint, local geometry/ego/history
inputs, four-frame action repeat and 200 development tracks. No new training was
performed. Source/dependency/checkpoint/partition hashes, search settings, constant
bias, episode counts, costs and exact first-20 replay records were verified.

| Condition | Mean official return | Completed laps | Completion | Mean off-road frames/race | Decision pipeline p95 | Run wall time |
|---|---:|---:|---:|---:|---:|---:|
| Direct policy | 862.42 | 71/200 | 35.5% | 25.82 | 1.50 ms | 1.40 min |
| Policy + fixed correction | 865.95 | 79/200 | 39.5% | 25.86 | 1.53 ms | 1.40 min |
| Policy + seven-candidate simulated search | 873.05 | 117/200 | 58.5% | 4.53 | 136.77 ms | 93.03 min |

Seven-candidate search versus direct improves completion by **23 percentage points**
(46 additional completed laps); its paired 95% track-bootstrap interval is
[16.0, 30.5] percentage points. Mean return increases **10.64**, with interval
[-2.56, 22.80]. The current evidence supports a completion benefit for this frozen
policy; a mean-return improvement remains uncertain. Off-road exposure decreases,
while search p95 latency exceeds the 80 ms decision interval. The environment
waits for each action; this is not a demonstrated real-time controller.

The additional descriptive search-versus-fixed-correction comparison is +19
percentage points completion [12.5, 25.5] and +7.10 return [-6.86, 20.48]. The
fixed correction versus direct is +4 percentage points [-1.5, 9.5] and +3.53
return [-5.21, 11.20]. These two contrasts supplement the originally declared
search-7-versus-direct contrast; they were requested during the running campaign.
All intervals use 5,000 paired bootstrap draws, seed zero, and are nominal,
unadjusted development intervals conditional on one trained policy and planning
stream. They do not measure variability across independently trained policies.

The remaining 150 tracks outside the first-50 checkpoint-selection panel show
37.3%, 39.3% and 57.3% completion for direct, fixed correction and search 7;
mean returns are 863.85, 866.98 and 870.88. These remain development tracks.
All prespecified nested panels, including the original 20, are retained in the
JSON. The smaller pilot was more favorable than the complete development panel.

Completed-condition costs: 560,744 native evaluation frames, 25,820,480 internal
native physics steps, zero new training. Earlier policy training, failed trials
and verification are additional costs. All internal physics here belongs to
search 7. Decision latency includes sensing and policy/search, excluding real
environment stepping and disk output. Fixed execution order and uncontrolled
ordinary workstation load limit timing comparisons.

The unfinished search-21 condition is excluded. No policy, setting, analysis
panel or queued training decision changes in response to this interim view.
Final test and private candidate-ranking outputs were not accessed. This table
does not establish SOTA or general performance across trained policies.
