"""State-only training, evaluation, and isolated Rust builds."""
import argparse

from cars.env.splits import SPLITS
from cars.eval.runner import evaluate
from cars.policies.baseline import BaselinePolicy
from cars.policies.rust import RustPolicy, build_rust


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    evaluation = commands.add_parser("eval")
    evaluation.add_argument("--method", choices=("base", "rust", "il_bc", "il_dagger", "rl_ppo", "rl_sac"), required=True)
    evaluation.add_argument("--rust-path")
    evaluation.add_argument("--model-path", help="learned run directory or model file (with normalization alongside)")
    evaluation.add_argument("--split", choices=SPLITS, default="seen-10")
    evaluation.add_argument("--n", type=int, default=10)
    evaluation.add_argument("--out", required=True)
    evaluation.add_argument("--workers", type=int, default=1)
    evaluation.add_argument("--seed", type=int, default=0, help="seed sequence for seen-inf")
    build = commands.add_parser("build-rust")
    build.add_argument("crate_dir")
    build.add_argument("--target-dir")
    for name in ("train-bc", "train-dagger", "train-rl"):
        train = commands.add_parser(name)
        train.add_argument("--split", choices=("seen-1", "seen-10", "seen-100", "seen-inf"), default="seen-10")
        train.add_argument("--seed", type=int, default=0)
        train.add_argument("--steps", type=int)
        train.add_argument("--episodes", type=int, help="IL rollout budget; RL nominal budget of episodes * 1000 steps")
        train.add_argument("--run-id", required=True)
        train.add_argument("--runs-dir", default="runs")
        train.add_argument("--batch-size", type=int, default=256)
        train.add_argument("--learning-rate", type=float, default=None)
        train.add_argument("--hidden-sizes", nargs="+", type=int, default=[256, 256])
        train.add_argument("--torch-threads", type=int, default=1)
        if name == "train-rl":
            train.add_argument("--algo", choices=("ppo", "sac"), required=True)
            train.add_argument("--n-envs", type=int)
            train.add_argument("--action-repeat", type=int, default=1, help="env frames per agent decision")
            train.add_argument("--throttle-axis", action="store_true", help="use [steer, signed throttle] actions")
            train.add_argument("--device", default="auto")
            train.add_argument("--eval-freq", type=int, default=50_000)
            train.add_argument("--eval-episodes", type=int, default=10)
            train.add_argument("--n-steps", type=int, default=1024, help="PPO decisions per rollout per worker")
            train.add_argument("--n-epochs", type=int, default=10)
            train.add_argument("--use-sde", action="store_true")
            train.add_argument("--learning-starts", type=int, default=10_000)
            train.add_argument("--buffer-size", type=int, default=1_000_000)
            train.add_argument("--train-freq", type=int, default=1)
            train.add_argument("--gradient-steps", type=int, default=1)
        else:
            train.add_argument("--epochs", type=int, default=30)
            if name == "train-dagger":
                train.add_argument("--iterations", type=int, default=10)
                train.add_argument("--bootstrap-episodes", type=int, default=10)
                train.add_argument("--beta-start", type=float, default=0.5)
                train.add_argument("--beta-end", type=float, default=0.0)
    args = parser.parse_args()
    if args.command == "build-rust":
        print(build_rust(args.crate_dir, args.target_dir))
        return
    if args.command.startswith("train-"):
        values = vars(args).copy()
        command, run_id, runs_dir = (values.pop(key) for key in ("command", "run_id", "runs_dir"))
        values["hidden_sizes"] = tuple(values["hidden_sizes"])
        if values["learning_rate"] is None:
            del values["learning_rate"]
        if command == "train-rl":
            from cars.rl.train import RLConfig, train_rl

            episodes = values.pop("episodes")
            if episodes is not None and values["steps"] is not None:
                parser.error("For RL, supply either --steps or --episodes")
            if episodes is not None and episodes < 1:
                parser.error("--episodes must be positive")
            values["steps"] = values["steps"] if values["steps"] is not None else (episodes * 1000 if episodes else 1_000_000)
            root = train_rl(RLConfig(**values), run_id, runs_dir)
        elif command == "train-bc":
            from cars.il.bc import BCConfig, train_bc

            root = train_bc(BCConfig(**values), run_id, runs_dir)
        else:
            from cars.il.dagger import DAggerConfig, train_dagger

            root = train_dagger(DAggerConfig(**values), run_id, runs_dir)
        print(root)
        return
    if args.method == "rust" and not args.rust_path:
        parser.error("--rust-path is required for --method rust")
    if args.method.startswith(("il_", "rl_")):
        if not args.model_path:
            parser.error("--model-path is required for learned methods")
        from cars.policies.sb3 import load_learned_policy

        policy = load_learned_policy(args.method, args.model_path)
    else:
        policy = RustPolicy(args.rust_path) if args.method == "rust" else BaselinePolicy()
    frame = evaluate(policy, args.split, args.n, args.out, args.workers, evaluation_seed=args.seed)
    columns = ["seed", "success", "return", "lap_time_s", "off_track_events", "off_track_steps", "crash", "progress", "steps"]
    print(frame[columns].to_string(index=False))
    print("\nMean ± sample standard deviation:")
    for name in columns[1:]:
        print(f"{name}: {frame[name].mean():.4f} ± {frame[name].std(ddof=1):.4f}")


if __name__ == "__main__":
    main()
