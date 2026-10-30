"""Verify and publish the completed offline actor audit and protocol-specific plots."""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from cars.training import sha256, write_json


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--campaign", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    status = json.loads((args.campaign / "status.json").read_text())
    assert status["status"] == "complete"
    rows = json.loads((args.campaign / "results.json").read_text())
    provenance, windows = {}, {}
    verified = 0
    for name in status["completed"]:
        root = Path(name)
        config = json.loads((root / "config.json").read_text())
        for path, expected in config["source_sha256"].items():
            assert sha256(root / "source" / path) == expected
            verified += 1
        provenance[name] = {str(p.relative_to(root)): sha256(p) for p in root.rglob("*")
                            if p.is_file() and "source" not in p.relative_to(root).parts}
        windows[root.name] = json.loads((root / "update_window.json").read_text())
    assert len(rows) == 24
    own = [r for r in rows if r["panel"] == "own"]
    assert all(r["own_recorded_action_max_error"] <= 1e-6 for r in own)
    assert all(r["predictor_parity_max_error"] == 0 for r in rows)
    result = {"status": status, "rows": rows, "update_windows": windows,
        "source_files_verified": verified,
        "all_saved_actions_match": True, "predictor_api_parity_exact": True,
        "scope": "12 frozen checkpoints, 5 development episodes each on own/common histories; conditional marginal analysis, not closed-loop sampled driving",
        "interpretation": "Effective action variation increases late in original local policy despite falling parameter scale; native variation decreases. No shared exploration-collapse cause established."}
    args.output.mkdir(parents=True, exist_ok=False)
    write_json(args.output / "summary.json", result)
    write_json(args.output / "provenance.json", {"runs": provenance,
        "campaign": {p.name: sha256(p) for p in args.campaign.iterdir() if p.is_file()},
        "analysis_script_sha256": sha256(Path(__file__))})
    fig, axes = plt.subplots(1, 3, figsize=(12, 4), constrained_layout=True)
    for ax, family in zip(axes, ("local_original", "local_discount", "native"), strict=True):
        panel = [r for r in rows if r["panel"] == "common" and r["family"] == family]
        x = np.arange(len(panel))
        ax.plot(x, [r["parameter_scale_mean"] for r in panel], "o--", label="Learned scale mean")
        for channel in ("steer", "gas", "brake"):
            ax.plot(x, [r["channels"][channel]["marginal_std_median"] for r in panel], "o-", label=channel+" marginal SD")
        ax.set_xticks(x, [r["checkpoint_label"] for r in panel])
        ax.set(title=family.replace("_", " ").title(), xlabel="Training checkpoint", ylabel="Scale / median marginal SD")
        ax.grid(alpha=.2)
    axes[0].legend(fontsize=8)
    fig.suptitle("Offline fixed-history audit: local checkpoints use native-frame labels; native uses decisions")
    fig.savefig(args.output / "action_variation.png", dpi=170)
    fig.savefig(args.output / "action_variation.svg")
    plt.close(fig)
    print(json.dumps({"source_files_verified": verified, **status}))


if __name__ == "__main__":
    main()
