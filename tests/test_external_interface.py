import json
from pathlib import Path
import random

import gymnasium as gym
import numpy as np
import pytest
import torch

from cars.env.protocol import ProtocolConfig
from cars.research.external_interface import (
    ExternalLocalEnv, LocalVectorCodec, normalized_action, physical_action,
)
from cars.research.external_rng import isolated_rng
from cars.research.external_sources import FILES, REVISIONS, sha256, stage_sources


def observation(codec):
    road = codec.visible.copy()
    road[road.shape[0] // 2:, :road.shape[1] // 2] = 0
    return {"road": road, "visible": codec.visible,
        "ego": np.array([19.125, -.375, .0625], np.float32),
        "previous_action": np.array([-.25, .625, .125], np.float32)}


def test_affine_actions_cover_endpoints_and_physical_sensor_convention():
    normalized = np.array([[-1., -1., -1.], [1., 1., 1.], [.25, -.5, .5]])
    physical = physical_action(normalized)
    np.testing.assert_array_equal(physical, [[-1., 0., 0.], [1., 1., 1.], [.25, .25, .75]])
    np.testing.assert_array_equal(normalized_action(physical), normalized)
    values = np.random.default_rng(8).uniform(-1, 1, (64, 3)).astype(np.float32)
    np.testing.assert_allclose(normalized_action(physical_action(values)), values, atol=1.2e-7, rtol=0)


@pytest.mark.parametrize("bad", [[1.01, 0., 0.], [0., -1.01, 0.],
    [0., float("nan"), 0.], [0., float("inf"), 0.], [1., 0.], [0j, 0j, 0j]])
def test_invalid_normalized_actions_are_rejected(bad):
    with pytest.raises(ValueError):
        physical_action(bad)


def test_local_vector_preserves_allowed_information_without_aliasing():
    codec = LocalVectorCodec(ProtocolConfig())
    original = observation(codec)
    vector = codec.encode(original)
    assert vector.shape == (3727,) and vector.dtype == np.float32
    restored = codec.decode(vector)
    for name in original:
        np.testing.assert_array_equal(restored[name], original[name])
    restored["ego"][0] = 99
    assert vector[-6] == original["ego"][0]
    assert codec.configuration()["history"] == 1


@pytest.mark.parametrize("violation", ["extra_truth", "visibility", "hidden_road", "soft_road", "ego_nan", "gas_negative"])
def test_local_boundary_rejects_privileged_or_malformed_inputs(violation):
    codec = LocalVectorCodec(ProtocolConfig())
    obs = observation(codec)
    if violation == "extra_truth": obs["track_seed"] = np.array(7)
    elif violation == "visibility": obs["visible"][0, 0] = 1
    elif violation == "hidden_road": obs["road"][0, 0] = 1
    elif violation == "soft_road": obs["road"][30, 30] = .5
    elif violation == "ego_nan": obs["ego"][0] = np.nan
    else: obs["previous_action"][1] = -.1
    with pytest.raises(ValueError): codec.encode(obs)


class FakeLocalEnv(gym.Env):
    def __init__(self, partition="train"):
        self.config, self.partition = ProtocolConfig(), partition
        self.codec = LocalVectorCodec(self.config)
        self.observation_space = gym.spaces.Dict({})
        self.action_space = gym.spaces.Box(np.array([-1, 0, 0], np.float32), 1., dtype=np.float32)
        self.last_action = None

    def reset(self, *, seed=None, options=None):
        return observation(self.codec), {"track_seed": 111, "true_map": "forbidden"}

    def step(self, action):
        self.last_action = action.copy()
        obs = observation(self.codec)
        obs["previous_action"] = action.copy()
        return obs, 13.25, False, True, {"native_frames": 2, "track_seed": 111, "global_progress": .7}


def test_harness_keeps_reward_timeout_frames_and_sanitizes_info():
    base = FakeLocalEnv()
    env = ExternalLocalEnv(base)
    _, info = env.reset(seed=3)
    assert info == {}
    vector, reward, terminated, truncated, info = env.step(np.array([.25, -1., 0.]))
    np.testing.assert_array_equal(base.last_action, [.25, 0., .5])
    np.testing.assert_array_equal(vector[-3:], base.last_action)
    assert (reward, terminated, truncated, info) == (13.25, False, True, {"native_frames": 2})
    assert env.real_decisions == 1 and env.native_frames == 2
    env.reset()
    assert env.real_decisions == 1 and env.native_frames == 2  # cumulative clocks
    with pytest.raises(ValueError): ExternalLocalEnv(FakeLocalEnv("calibration"))
    with pytest.raises(ValueError): LocalVectorCodec(ProtocolConfig(view="global"))


def draw():
    return random.random(), np.random.random(), torch.rand(3)


def assert_draw_equal(a, b):
    assert a[:2] == b[:2]
    torch.testing.assert_close(a[2], b[2], rtol=0, atol=0)


def test_planning_rng_repeats_without_advancing_learner_even_on_exception():
    with isolated_rng(99): expected = draw()
    with isolated_rng(99):
        with isolated_rng(17): first = draw()
        with pytest.raises(RuntimeError), isolated_rng(17):
            assert_draw_equal(first, draw())
            raise RuntimeError("synthetic evaluation failure")
        assert_draw_equal(expected, draw())


def test_rng_scope_rejects_late_cuda_initialization(monkeypatch):
    initialized = [False]
    monkeypatch.setattr(torch.cuda, "is_initialized", lambda: initialized[0])
    with pytest.raises(RuntimeError, match="Initialize CUDA"):
        with isolated_rng(7):
            initialized[0] = True


def source_fixture(tmp_path: Path, algorithm: str):
    cache, audit = tmp_path / "cache", tmp_path / "audit.json"
    entries = []
    for name in FILES[algorithm]:
        content = b"# synthetic fixture; not official source\n"
        if name.endswith("/buffer.py"):
            content += (b"if self.device == torch.device('cuda'):\n    pass\nself.priority = torch.empty(self.max_size, device=self.device) if self.prioritized else []\n" if algorithm == "MRQ" else
                b"self.storage_device = self.device if check_memory(max_size, obs_shape, 1 if pixel_obs else 4, action_dim) else torch.device('cpu')\n")
        path = cache / algorithm / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
        entries.append({"repo": algorithm, "path": name, "commit": REVISIONS[algorithm],
            "sha256": sha256(content), "url": "https://example.invalid/synthetic"})
    audit.write_text(json.dumps({"repositories": {algorithm: {"commit": REVISIONS[algorithm]}}, "files": entries}))
    return cache, audit


@pytest.mark.parametrize("algorithm", ["MRQ", "MRSQ"])
def test_source_staging_preserves_original_and_refuses_overwrite(tmp_path, algorithm):
    cache, audit = source_fixture(tmp_path, algorithm)
    original = (cache / algorithm / algorithm / "buffer.py").read_bytes()
    destination = tmp_path / "stage"
    manifest = stage_sources(algorithm, cache, audit, destination)
    assert (cache / algorithm / algorithm / "buffer.py").read_bytes() == original
    assert (destination / "buffer.py").read_bytes() != original
    assert manifest["third_party_code_executed"] is False
    assert (destination / "LICENSE").exists()
    assert len(list(destination.iterdir())) == len(FILES[algorithm]) + 1
    with pytest.raises(FileExistsError): stage_sources(algorithm, cache, audit, destination)


def test_source_mismatch_fails_before_output_creation(tmp_path):
    cache, audit = source_fixture(tmp_path, "MRQ")
    (cache / "MRQ/MRQ/models.py").write_text("# unexpected changed source")
    with pytest.raises(ValueError, match="source mismatch"):
        stage_sources("MRQ", cache, audit, tmp_path / "stage")
    assert not (tmp_path / "stage").exists()
