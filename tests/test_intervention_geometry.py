"""Causal isolation checks for public, explicitly privileged module substitutions."""
from itertools import product

import numpy as np
import pytest
from scipy.spatial import cKDTree

from cars.env.protocol import DrivingEnv, LocalSensor, ProtocolConfig
from cars.planning.intervention import InterventionConfig, LOCAL_FIELDS, intervention_worlds
from cars.planning.policy_search import imagined_observations
from cars.planning.visible_reference import visible_reference


def test_visible_reference_clips_before_connecting_and_ignores_hidden_detours():
    loop = np.array([[-40., 0.], [40., 0.], [40., 40.], [-40., 40.]])
    expected = np.column_stack((np.arange(-30., 31.), np.zeros(61)))
    np.testing.assert_allclose(visible_reference(loop, 30., 1.), expected)
    # Remote geometry, vertex order origin and segment tessellation cannot
    # disclose a different hidden continuation to this current-only reference.
    detour = np.array([[-40., 0.], [0., 0.], [40., 0.], [100., 100.], [-100., 100.]])
    for shift in range(len(detour)):
        np.testing.assert_allclose(visible_reference(np.roll(detour, shift, axis=0), 30., 1.), expected)
    reverse = visible_reference(loop[::-1], 30., 1.)
    np.testing.assert_allclose(reverse, expected[::-1])  # direction is an explicit oracle label
    disjoint = np.array([[-50., 2.], [50., 2.], [50., 20.], [-50., 20.]])
    path = visible_reference(disjoint, 30., 1.)
    assert np.allclose(path[:, 1], 2.) and (np.linalg.norm(path, axis=1) <= 30. + 1e-10).all()
    assert visible_reference(loop + 1000, 30., 1.) is None
    with pytest.raises(ValueError):
        visible_reference(loop * np.nan, 30., 1.)


def test_world_and_objective_interventions_are_independent_and_root_is_unchanged():
    config = ProtocolConfig(view="global")
    env = DrivingEnv(config, "development")
    try:
        obs, _ = env.reset()
        contexts = {(reference, continuation): intervention_worlds(obs, config,
            InterventionConfig(reference, continuation), 6.67)
            for reference, continuation in product(("reconstructed", "oracle_visible"), ("prior", "oracle"))}
        sensor = LocalSensor(config)
        positions = np.array([[0., 0.], [15., -2.], [-10., 0.], [0., 25.]])
        base = contexts["reconstructed", "prior"]
        for (reference, continuation), (local, world, path, fallback) in contexts.items():
            assert set(local) == LOCAL_FIELDS and not fallback
            np.testing.assert_array_equal(world.path, base[1].path)
            # Road physics and objective reference have independent switches.
            counterpart = contexts["reconstructed", continuation]
            probes = np.r_[positions, obs["map_centre"][::7], [[1000., 1000.]]]
            a, b = world.query(probes), counterpart[1].query(probes)
            for field in ("grip", "margin", "visible"):
                np.testing.assert_array_equal(getattr(a, field), getattr(b, field))
                np.testing.assert_array_equal(getattr(world.query(positions), field),
                                              getattr(base[1].query(positions), field))
            np.testing.assert_array_equal(path, contexts[reference, "prior"][2])
            imagined = imagined_observations(world, sensor, cKDTree(world.path), np.zeros((1, 3)),
                local["ego"][None], local["previous_action"][None], "sparse")
            for key in LOCAL_FIELDS:
                np.testing.assert_array_equal(imagined[key][0], local[key])
        assert not np.array_equal(base[2], contexts["oracle_visible", "prior"][2])
        # Positive control: supplying the map really changes hidden road.
        probes = obs["map_centre"][np.linalg.norm(obs["map_centre"], axis=1) > 60]
        local, oracle = base[1].query(probes), contexts["reconstructed", "oracle"][1].query(probes)
        assert np.any(local.grip < .5) and np.all(oracle.grip > .9)
        np.testing.assert_array_equal(local.visible, oracle.visible)  # fixed objective penalty
        with pytest.raises(ValueError, match="schema"):
            intervention_worlds({key: obs[key] for key in LOCAL_FIELDS}, config, InterventionConfig(), 6.67)
        remote = {**obs, "map_centre": obs["map_centre"] + 10000.}
        _, unchanged, objective, fallback = intervention_worlds(remote, config, InterventionConfig(), 6.67)
        np.testing.assert_array_equal(unchanged.path, base[1].path)
        np.testing.assert_array_equal(objective, base[2])
        assert not fallback  # negative control does not use the map at all
        _, _, objective, fallback = intervention_worlds(remote, config, InterventionConfig("oracle_visible"), 6.67)
        np.testing.assert_array_equal(objective, base[2])
        assert fallback
    finally:
        env.close()
