import numpy as np
import pytest

from cars.research.failures import describe_episode


def test_failure_diagnostics_distinguish_terminal_contact_and_tail_speed():
    metrics = {"steps": 12, "off_track_steps": 5, "success": False,
               "crash": False, "lap_time_s": np.nan, "progress": .3}
    arrays = {"frames": np.array([4, 4, 4]), "truncated": [False, False, True],
              "evaluation_offroad_frame_indices": [2, 3, 10, 11, 12],
              "observation_ego": [[0, 0, 0], [20, 0, 0], [4, 0, 0], [1, 0, 0]]}
    definitions = {"tail_native_frames": 6, "low_speed_mps": 3., "low_progress_fraction": .5}
    result = describe_episode(metrics, arrays, definitions)
    assert result["outcome"] == "timeout_offroad"
    assert result["tail_mean_speed"] == 2
    assert result["low_speed_tail"] and result["low_progress"]
    assert result["final_offroad_frames"] == result["longest_offroad_frames"] == 3
    assert not result["road_contact_restored_by_end"]
    arrays["evaluation_offroad_frame_indices"] = [2, 3, 8, 9, 10]
    result = describe_episode(metrics, arrays, definitions)
    assert result["outcome"] == "timeout_onroad" and result["road_contact_restored_by_end"]
    metrics["success"] = True
    assert describe_episode(metrics, arrays, definitions)["outcome"] == "completed"
    metrics["off_track_steps"] = 4
    with pytest.raises(ValueError, match="accounting"):
        describe_episode(metrics, arrays, definitions)
