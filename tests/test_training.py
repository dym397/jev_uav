from train import checkpoint_key


def test_checkpoint_selection_prioritizes_success_over_reward():
    successful = {"success_rate": 0.8, "on_time_rate": 0.4,
                  "collision_rate": 0.2, "average_reward": -10.0}
    stalled = {"success_rate": 0.0, "on_time_rate": 0.0,
               "collision_rate": 0.0, "average_reward": 20.0}
    assert checkpoint_key(successful) > checkpoint_key(stalled)
