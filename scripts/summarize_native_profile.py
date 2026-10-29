"""Export all native profile attempts, with immutable sources and failure accounting."""
import argparse
import json
from pathlib import Path

from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--campaign', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    attempts, campaigns, source_sets, configs = [], [], [], []
    verified = reset_ticks = training_frames = rollout_updates = 0
    reload_low = reload_high = 0
    for campaign in args.campaign:
        status = json.loads((campaign / 'status.json').read_text())
        if status['status'] not in ('complete', 'complete_with_failures'):
            raise ValueError('Only finished campaigns may be summarized')
        spec = json.loads((campaign / 'config.json').read_text())
        rows = json.loads((campaign / 'results.json').read_text())
        assert len(rows) == len(spec['conditions'])
        for row, case in zip(rows, spec['conditions'], strict=True):
            assert row['run_id'] == case['run_id']
            root = Path('runs/native_training') / row['run_id']
            config = json.loads((root / 'config.json').read_text())
            assert config['partition_sha256'] == spec['partition_sha256']
            training = config['settings']['training']
            assert training['workers'] == spec['workers_per_case']
            assert training['workers'] * training['n_steps'] == spec['transitions_per_rollout']
            assert training['total_decisions'] == spec['global_schedule_decisions']
            assert row['native_frames'] == spec['rollouts_per_case'] * spec['transitions_per_rollout']
            configs.append({k: v for k, v in training.items() if k not in ('device', 'vector_backend')})
            sources = config['source_sha256']
            source_sets.append(sources)
            for name, expected in sources.items():
                assert sha256(root / 'source' / name) == expected, name
                verified += 1
            for name, expected in row.get('artifact_sha256', {}).items():
                assert sha256(root / name) == expected, name
            row['verified_artifact_sha256'] = {name: sha256(root / name) for name in (
                'config.json', 'initial_policy.json', 'runtime.json', 'learning.jsonl',
                'model.zip', 'normalization.pkl', 'training_state.json', 'training_tracks.json',
                'reload_check.json', 'cost.json', 'failure.json', 'status.json') if (root / name).exists()}
            learning = [json.loads(line) for line in (root / 'learning.jsonl').read_text().splitlines()]
            assert len(learning) == spec['rollouts_per_case']
            tracks = json.loads((root / 'training_tracks.json').read_text())
            resets = sum(map(len, tracks))
            training_frames += row['native_frames']
            rollout_updates += len(learning)
            reset_ticks += resets + 1  # every recorded attempt entered one reload environment reset
            if row['status'] == 'complete':
                assert row['reload_check']['passed'] and row['reload_check']['normalization_frozen']
                reload_low += row['reload_check']['native_frames']
                reload_high += row['reload_check']['native_frames']
            else:
                # Original failure artifacts prove assertions occurred before the second
                # step, but did not persist the exact intermediate step counter.
                assert 'Not equal to tolerance' in row['error']
                reload_high += 1
                row['unpersisted_costs'] = 'reload actions 0..1; GPU peak memory and total case wall unavailable'
            attempts.append(row)
        campaigns.append({'path': str(campaign), 'status': status,
            'artifact_sha256': {name: sha256(campaign / name) for name in (
                'config.json', 'runner.py', 'provenance.json', 'results.json', 'status.json')}})
    assert all(config == configs[0] for config in configs)
    changed = sorted({name for sources in source_sets[1:] for name in set(sources) | set(source_sets[0])
                      if sources.get(name) != source_sets[0].get(name)})
    if set(changed) - {'src/cars/research/native_training.py'}:
        raise ValueError(f'Unexpected cross-profile source changes: {changed}')
    valid = [row for row in attempts if row['status'] == 'complete']
    winner = min(valid, key=lambda row: row['steady_learn_seconds_mean'])
    initial = [json.loads((Path('runs/native_training') / row['run_id'] / 'initial_policy.json').read_text())
               ['parameters_sha256'] for row in attempts]
    result = {'campaigns': campaigns, 'selected_run_id': winner['run_id'],
        'selection': 'successful finite updates and reload, then mean learn time of rollouts 2 and 3; no driving score',
        'training_native_frames_all_attempts': training_frames,
        'reload_native_frames_range': [reload_low, reload_high], 'reset_physics_ticks': reset_ticks,
        'rollout_updates_all_attempts': rollout_updates,
        'campaign_wall_seconds': sum(c['status']['wall_seconds'] for c in campaigns),
        'source_files_verified': verified, 'cross_profile_changed_source_files': changed,
        'source_change_reason': 'same-device/CPU-repeatability checks; cross-device diagnostic; failure cost persistence; metadata only, no learning-rule change',
        'identical_initial_parameters': len(set(initial)) == 1,
        'development_episodes': 0, 'final_test_episodes': 0, 'internal_search_physics_steps': 0,
        'conditions': valid, 'all_attempts': attempts,
        'limitations': ['Three rollouts per setup, one seed; not a driving-competence result',
            'Full-run projection excludes evaluation/checkpoint overhead and changes in later recurrent batch lengths',
            'Parent RSS is not total subprocess memory; GPU memory excludes other processes',
            'Original GPU attempts failed cross-device assertions and remain failed; retry costs are included',
            'Original failed reload step counters are bounded, not falsely reported as exact',
            'Unit/integration test work is separate from the campaign ledger']}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / 'summary.json', result)
    lines = ['# Native recipe workstation profile', '',
        'Technical profile only; no development driving performance selection.', '',
        '| Device | Backend | First update (s) | Later mean (s) | Frames/s | Projected 4M (h) | Peak GPU allocated (GiB) |',
        '|---|---|---:|---:|---:|---:|---:|']
    for row in valid:
        lines.append(f"| {row['device']} | {row['vector_backend']} | {row['first_learn_seconds']:.2f} | "
            f"{row['steady_learn_seconds_mean']:.2f} | {row['steady_native_frames_per_second']:.1f} | "
            f"{row['projected_977_rollouts_hours']:.2f} | {row['cuda_peak_allocated_bytes'] / 2**30:.2f} |")
    lines += ['', f"Selected: `{winner['run_id']}`.", '',
        f'All attempts: {training_frames:,} training frames, {reload_low}–{reload_high} reload frames, '
        f'{reset_ticks} reset physics ticks, {rollout_updates} full updates; '
        f"{result['campaign_wall_seconds'] / 60:.2f} campaign minutes.", '',
        'Original GPU failures and subsequent technical corrections are retained in summary.json.', '',
        *[f'- {item}' for item in result['limitations']]]
    (args.output / 'summary.md').write_text('\n'.join(lines) + '\n')
    print(json.dumps({k: result[k] for k in ('selected_run_id', 'training_native_frames_all_attempts',
        'reload_native_frames_range', 'source_files_verified', 'identical_initial_parameters')}))


if __name__ == '__main__':
    main()
