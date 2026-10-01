import numpy as np

from agents.jev_agent import build_choice_request
from experiment_motion import first_observation
from experiment_motion_grid import counterfactual_grid


def test_counterfactual_grid_changes_only_target_velocity():
    grid = counterfactual_grid()
    assert len(grid) == 8
    for approaching, moving_away in grid.values():
        a = first_observation(approaching, 30)
        b = first_observation(moving_away, 30)
        np.testing.assert_array_equal(a.numeric, b.numeric)
        assert build_choice_request(a)["states"][0]["state"] == \
            build_choice_request(b)["states"][0]["state"]
        assert a.contacts[0].relative_left_mps == -4.0
        assert b.contacts[0].relative_left_mps == 4.0
