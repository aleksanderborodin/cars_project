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
    evaluation.add_argument("--method", choices=("local", "fixed", "search", "physics", "physics-global", "expert", "recurrent", "policy-search", "policy-random", "policy-bias"), required=True)
    evaluation.add_argument("--model", type=Path)
    evaluation.add_argument("--episodes", type=int, default=20)
    evaluation.add_argument("--horizon", type=int, default=20)
    evaluation.add_argument("--speed", type=float, default=60.)
    evaluation.add_argument("--gain", type=float, default=1., help="fixed-controller steering gain")
    evaluation.add_argument("--speed-grid", type=float, nargs="+")
    evaluation.add_argument("--gain-grid", type=float, nargs="+", help="candidate steering gains for search/physics")
    evaluation.add_argument("--candidates", type=int, default=7)
    evaluation.add_argument("--planning-seed", type=int, default=0)
    evaluation.add_argument("--steer-sigma", type=float, default=.10)
    evaluation.add_argument("--gas-sigma", type=float, default=.15)
    evaluation.add_argument("--brake-sigma", type=float, default=.15)
    evaluation.add_argument("--observation-renderer", choices=("reference", "sparse"), default="reference")
    evaluation.add_argument("--action-bias", type=float, nargs=3)
    evaluation.add_argument("--user-monitor", action="store_true", help="isolated private oracle-ranking report; never read by training or agents")
    evaluation.add_argument("--run-id", required=True)
    training = commands.add_parser("train-recurrent")
    training.add_argument("--frames", type=int, default=100000)
    training.add_argument("--seed", type=int, default=0)
    training.add_argument("--workers", type=int, default=4)
    training.add_argument("--view", choices=("local", "native"), default="local")
    training.add_argument("--run-id", required=True)
    training.add_argument("--resume", type=Path, help="completed research training run; fresh episodes, preserved training state")
    configured = commands.add_parser("train-driver", help="configured recurrent baseline with native-frame schedule")
    configured.add_argument("--config", type=Path, required=True)
    configured.add_argument("--stop-at", type=int, required=True, help="cumulative native-frame milestone")
    configured.add_argument("--run-id", required=True)
    configured.add_argument("--resume", type=Path)
    configured.add_argument("--view", choices=("local", "native"), default="local")
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
    elif args.command == "train-driver":
        import json
        from cars.research.driver_config import DriverTrainingConfig
        from cars.research.driver_training import train_driver
        config = DriverTrainingConfig(**json.loads(args.config.read_text()))
        print(train_driver(args.run_id, config, stop_at=args.stop_at,
                           protocol=ProtocolConfig(view=args.view), resume=args.resume))
    elif args.command == "evaluate":
        protocol = ProtocolConfig()
        if args.method == "local":
            driver = LocalPursuit(protocol)
        elif args.method == "fixed":
            driver = FixedBehavior(protocol, speed=args.speed, gain=args.gain)
        elif args.method == "expert":
            driver = BaselinePolicy()
        elif args.method in ("physics", "physics-global"):
            from cars.planning.physics_search import PhysicsSearch
            from cars.planning.global_search import GlobalPhysicsSearch
            if args.method == "physics-global":
                protocol = ProtocolConfig(view="global")
            driver_class = GlobalPhysicsSearch if args.method == "physics-global" else PhysicsSearch
            driver = driver_class(protocol, SearchConfig(horizon=args.horizon,
                speeds=tuple(args.speed_grid) if args.speed_grid else SearchConfig().speeds,
                steering_gains=tuple(args.gain_grid) if args.gain_grid else SearchConfig().steering_gains))
        else:
            if args.model is None:
                parser.error("--model is required")
            protocol = ProtocolConfig(**read_config(args.model)["protocol"])
            if args.method == "search":
                driver = BehaviorSearch(protocol, Dynamics.load(args.model / "model.npz"),
                                        SearchConfig(horizon=args.horizon,
                                            speeds=tuple(args.speed_grid) if args.speed_grid else SearchConfig().speeds,
                                            steering_gains=tuple(args.gain_grid) if args.gain_grid else SearchConfig().steering_gains),
                                        model_artifact=args.model)
            elif args.method == "policy-bias":
                from cars.planning.policy_controls import FixedPolicyBias
                if args.action_bias is None:
                    parser.error("policy-bias requires --action-bias STEER GAS BRAKE")
                driver = FixedPolicyBias(args.model, tuple(args.action_bias))
            elif args.method in ("policy-search", "policy-random"):
                from cars.planning.policy_search import PolicyPhysicsSearch, PolicySearchConfig, RandomPolicyVariation
                driver_class = PolicyPhysicsSearch if args.method == "policy-search" else RandomPolicyVariation
                driver = driver_class(args.model, protocol, PolicySearchConfig(candidates=args.candidates,
                    horizon=args.horizon, planning_seed=args.planning_seed, steer_sigma=args.steer_sigma,
                    gas_sigma=args.gas_sigma, brake_sigma=args.brake_sigma,
                    observation_renderer=args.observation_renderer))
            else:
                from cars.research.recurrent import RecurrentDriver
                driver = RecurrentDriver(args.model)
        print(evaluate_driver(driver, args.run_id, episodes=args.episodes, protocol=protocol, user_monitor=args.user_monitor))
    else:
        paired_report(args.runs, args.output)
        print(args.output)


if __name__ == "__main__":
    main()
