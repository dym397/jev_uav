from config import EnvConfig
from environment.scenario import Scenario
from environment.uav_env import UAVEnv
from evaluation import run_episode, summarize


class MaintainAgent:
    def select_action(self, state):
        return 4


def test_evaluation_separates_success_from_punctuality_and_records_path():
    scene = Scenario((0.0, 0.0), 0.0, 6.0, (6.0, 0.0), 20.0, world_size=300.0)
    result = run_episode(UAVEnv(EnvConfig(), scenario=scene), MaintainAgent())
    metrics = summarize([result])
    assert metrics["success_rate"] == 1.0
    assert metrics["collision_rate"] == 0.0
    assert metrics["on_time_rate"] == 0.0
    assert metrics["average_path_length"] == 6.0


def test_no_success_path_length_is_json_null():
    scene = Scenario((0.0, 0.0), 0.0, 0.1, (100.0, 0.0), 20.0, world_size=300.0)
    result = run_episode(UAVEnv(EnvConfig(max_steps=1), scenario=scene), MaintainAgent())
    assert summarize([result])["average_success_path_length"] is None
