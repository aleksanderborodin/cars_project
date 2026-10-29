"""Programmed CarRacing vehicle physics in independently constructed worlds.

Only permitted observations, past actions and an inferred road enter this model.
No evaluation environment/body, hidden road, track ID or true wheel state enters.
Road contact is an explicit five-point wheel-footprint approximation in v1.
"""
from dataclasses import dataclass
import inspect
from pathlib import Path

from Box2D import b2World
from gymnasium.envs.box2d.car_dynamics import Car
import numpy as np
from scipy.spatial import cKDTree

from cars.env.protocol import Observation
from cars.planning.world import LocalRoadWorld
from cars.training import sha256


@dataclass(eq=False)
class RoadTile:
    road_friction: float = 1.


class PhysicsDynamics:
    """Known vehicle code, estimated latent wheel state, approximate road contact.

    Each imagined candidate owns a fresh Box2D world. Between actual decisions
    we carry the chosen candidate's FIRST simulated transition's wheel estimate,
    then correct hull velocity/yaw using the next observed ego state. Rebuilding
    joints discards solver warm-start impulses; this is not an exact state clone.
    """
    kind = "gymnasium_car_box2d_observation_initialized_v1"

    def __init__(self, repeat: int = 4):
        self.repeat = repeat
        self.tile = RoadTile()
        self.reset()

    def reset(self) -> None:
        for car in getattr(self, "cars", []):
            car.destroy()
        self.estimate: np.ndarray | None = None
        self.cars: list[Car] = []
        self.first_estimates: list[np.ndarray] | None = None
        self.first_ego: np.ndarray | None = None

    def configuration(self) -> dict:
        return {"kind": self.kind, "vehicle_source_sha256": sha256(Path(inspect.getfile(Car))),
                "physics_hz": 50, "box2d_velocity_iterations": 180, "box2d_position_iterations": 60,
                "road_contact": "any_of_four_wheel_vertices_and_center_has_road_occupancy_at_least_half",
                "latent_state": "carry_first_selected_physics_transition; correct_observed_hull_velocity_yaw",
                "new_training_frames": 0, "evaluation_environment_state_access": False,
                "joint_solver_warm_start_restored": False}

    def begin(self, observation: Observation, road: LocalRoadWorld, count: int) -> None:
        self.road, self.tree = road, cKDTree(road.path)
        for car in self.cars:
            car.destroy()
        self.cars = []
        self.first_estimates, self.first_ego = None, None
        forward, left, yaw = map(float, observation["ego"])
        # Model coordinates are (forward, left); Box2D car at angle zero faces +y.
        velocity = np.array([-left, forward])
        for _ in range(count):
            car = Car(b2World(gravity=(0, 0)), 0., 0., 0.)
            car.hull.linearVelocity = tuple(velocity)
            car.hull.angularVelocity = yaw
            for i, wheel in enumerate(car.wheels):
                if self.estimate is None:
                    omega, angle, gas, relative_yaw = forward / wheel.wheel_rad, 0., 0., 0.
                else:
                    omega, angle, gas, relative_yaw = self.estimate[i]
                wheel.angle = float(angle)
                wheel.omega, wheel.gas = float(omega), float(gas)
                wheel.angularVelocity = yaw + float(relative_yaw)
                x, y = wheel.position
                wheel.linearVelocity = tuple(velocity + yaw * np.array([-y, x]))
            car.steer(-float(observation["previous_action"][0]))
            car.brake(float(observation["previous_action"][2]))
            self.cars.append(car)

    def _contact(self) -> None:
        wheels = [wheel for car in self.cars for wheel in car.wheels]
        points = []
        for wheel in wheels:
            for vertex in [*wheel.fixtures[0].shape.vertices, (0., 0.)]:
                point = wheel.GetWorldPoint(vertex)
                points.append((point.y, -point.x))
        positions = np.asarray(points)
        distances = self.tree.query(positions)[0]
        grip = self.road.query(positions, centre_distance=distances).grip.reshape(len(wheels), 5)
        for wheel, on_road in zip(wheels, np.any(grip >= .5, axis=1), strict=True):
            wheel.tiles = {self.tile} if on_road else set()

    def advance(self, pose: np.ndarray, ego: np.ndarray, action: np.ndarray,
                previous: np.ndarray, grip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # The search interface also supplies its center-contact proxy. Vehicle
        # physics instead queries each wheel's footprint in the same road model.
        for _ in range(self.repeat):
            self._contact()
            for car, control in zip(self.cars, action, strict=True):
                car.steer(-float(control[0]))
                car.gas(float(control[1]))
                car.brake(float(control[2]))
                car.step(.02)
                car.world.Step(.02, 180, 60)
        poses, egos = [], []
        for car in self.cars:
            x, y = car.hull.position
            angle = car.hull.angle
            vx, vy = car.hull.linearVelocity
            c, s = np.cos(angle), np.sin(angle)
            poses.append((y, -x, angle))
            egos.append((-s * vx + c * vy, -c * vx - s * vy, car.hull.angularVelocity))
        if self.first_estimates is None:
            self.first_estimates = [np.array([
                [w.omega, w.joint.angle, w.gas, w.angularVelocity - car.hull.angularVelocity]
                for w in car.wheels]) for car in self.cars]
            self.first_ego = np.asarray(egos)
        return np.asarray(poses), np.asarray(egos)

    def commit(self, selected: int) -> None:
        assert self.first_estimates is not None
        self.commit_estimate(self.first_estimates[selected])

    def commit_estimate(self, estimate: np.ndarray) -> None:
        """Commit an owned first-transition estimate, possibly from an earlier batch."""
        if estimate.shape != (4, 4) or not np.isfinite(estimate).all():
            raise ValueError("Finite four-wheel observation-based estimate required")
        self.estimate = estimate.copy()
