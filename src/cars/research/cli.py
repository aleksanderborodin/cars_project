"""Development-only experiment commands; final held-out evaluation is unavailable."""
import argparse
from pathlib import Path

from cars.env.partitions import Partitions
from cars.env.protocol import ProtocolConfig
from cars.models.dynamics import Dynamics
from cars.planning.local import FixedBehavior, LocalPursuit
from cars.planning.search import BehaviorSearch, SearchConfig
from cars.policies.baseline import BaselinePolicy
from cars.research.artifacts import read_config
from cars.research.data import collect, fit
from cars.research.evaluation import evaluate_driver, paired_report


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    reserve = commands.add_parser("reserve")
    reserve.add_argument("--output", type=Path, default=Path("configs/partitions/standard-v1.json"))
    collection = commands.add_parser("collect")
    collection.add_argument("--partition", choices=("train", "development"), default="train")
    collection.add_argument("--episodes", type=int, default=20)
    collection.add_argument("--seed", type=int, default=0)
    collection.add_argument("--noise", type=float, default=.08)
    collection.add_argument("--collector", choices=("local", "mixed"), default="local")
    collection.add_argument("--run-id", required=True)
    fitting = commands.add_parser("fit")
    fitting.add_argument("--training", type=Path, required=True)
    fitting.add_argument("--development", type=Path, required=True)
    fitting.add_argument("--ridge", type=float, default=.01)
    fitting.add_argument("--run-id", required=True)
    evaluation = commands.add_parser("evaluate")
    evaluation.add_argument("--method", choices=("local", "fixed", "search", "expert", "recurrent"), required=True)
    evaluation.add_argument("--model", type=Path)
    evaluation.add_argument("--episodes", type=int, default=20)
    evaluation.add_argument("--horizon", type=int, default=20)
    evaluation.add_argument("--speed", type=float, default=60.)
    evaluation.add_argument("--speed-grid", type=float, nargs="+")
    evaluation.add_argument("--user-monitor", action="store_true", help="isolated private oracle-ranking report; never read by training or agents")
    evaluation.add_argument("--run-id", required=True)
    training = commands.add_parser("train-recurrent")
    training.add_argument("--frames", type=int, default=100000)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--view", choices=("local", "native"), default="local")
    training.add_argument("--run-id", required=True)
    training.add_argument("--resume", type=Path, help="completed research training run; fresh episodes, preserved training state")
    report = commands.add_parser("report")
    report.add_argument("runs", type=Path, nargs="+")
    report.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "reserve":
        Partitions().write(args.output)
        print(f"Partition manifest saved: {args.output}; fingerprint={Partitions().fingerprint}; final identities sealed")
    elif args.command == "collect":
        print(collect(args.run_id, partition=args.partition, episodes=args.episodes, seed=args.seed, noise=args.noise,
                      collector=args.collector))
    elif args.command == "fit":
        print(fit(args.run_id, args.training, args.development, ridge=args.ridge))
    elif args.command == "train-recurrent":
        from cars.research.recurrent import train
        print(train(args.run_id, frames=args.frames, seed=args.seed, workers=args.workers,
                    protocol=ProtocolConfig(view=args.view), resume=args.resume))
    elif args.command == "evaluate":
        protocol = ProtocolConfig()
        if args.method == "local":
            driver = LocalPursuit(protocol)
        elif args.method == "fixed":
            driver = FixedBehavior(protocol, speed=args.speed)
        elif args.method == "expert":
            driver = BaselinePolicy()
        else:
            if args.model is None:
                parser.error("--model is required")
            protocol = ProtocolConfig(**read_config(args.model)["protocol"])
            if args.method == "search":
                driver = BehaviorSearch(protocol, Dynamics.load(args.model / "model.npz"),
                                        SearchConfig(horizon=args.horizon, speeds=tuple(args.speed_grid) if args.speed_grid else SearchConfig().speeds),
                                        model_artifact=args.model)
            else:
                from cars.research.recurrent import RecurrentDriver
                driver = RecurrentDriver(args.model)
        print(evaluate_driver(driver, args.run_id, episodes=args.episodes, protocol=protocol, user_monitor=args.user_monitor))
    else:
        paired_report(args.runs, args.output)
        print(args.output)


if __name__ == "__main__":
    main()
