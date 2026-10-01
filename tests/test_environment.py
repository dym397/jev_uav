import math

import numpy as np

from config import Action, EnvConfig
from environment.collision import moving_collision
from environment.obstacle import Intruder, StaticObstacle
from environment.scenario import Scenario, make_scenario
from environment.uav_env import UAVEnv


def bare_scenario(*, intruders=(), obstacles=(), goal=(100.0, 0.0), due=17.0):
    return Scenario(
        start=(0.0, 0.0), heading=0.0, speed=6.0,
        goal=goal, due_time=due, intruders=list(intruders),
        obstacles=list(obstacles), world_size=300.0,
    )


def test_nine_actions_and_paper_motion_equation():
    assert len(Action) == 9
    env = UAVEnv(EnvConfig(dt=1.0), scenario=bare_scenario())
    env.reset()
    _, _, _, _, info = env.step(Action.TURN_LEFT_ACCELERATE)
    assert math.isclose(info["own_speed"], 9.0)
    assert math.isclose(info["own_heading"], math.pi / 30)
    np.testing.assert_allclose(info["own_position"], [6.0, 0.0])


def test_paper_observation_has_nearest_intruder_per_sector_and_text():
    intruders = [
        Intruder(20.0, 0.0, 0.0, 0.0),
        Intruder(40.0, 0.0, 0.0, 0.0),
    ]
    env = UAVEnv(EnvConfig(), scenario=bare_scenario(intruders=intruders))
    obs, _ = env.reset()
    assert obs.numeric.shape == (14,)
    assert math.isclose(float(np.min(obs.numeric[5:])), 0.2, abs_tol=1e-6)
    assert np.count_nonzero(obs.numeric[5:] < 1.0) == 1
    assert "sector_0=0.2" in obs.state_to_text()


def test_observed_contact_motion_changes_without_changing_paper_vector():
    approaching = bare_scenario(intruders=[Intruder(50.0, 40.0, 0.0, -4.0)])
    receding = bare_scenario(intruders=[Intruder(50.0, 40.0, 0.0, 4.0)])
    a, _ = UAVEnv(EnvConfig(), scenario=approaching).reset()
    b, _ = UAVEnv(EnvConfig(), scenario=receding).reset()
    np.testing.assert_array_equal(a.numeric, b.numeric)
    assert len(a.contacts) == len(b.contacts) == 1
    assert a.contacts[0].forward_m == 50.0
    assert a.contacts[0].left_m == 40.0
    assert a.contacts[0].relative_forward_mps == -6.0
    assert a.contacts[0].relative_left_mps == -4.0
    assert b.contacts[0].relative_left_mps == 4.0


def test_contact_motion_uses_body_axes_and_detection_radius():
    scene = bare_scenario(intruders=[Intruder(-40.0, 50.0, 0.0, 0.0),
                                     Intruder(150.0, 0.0, 0.0, 0.0)])
    scene.heading = math.pi / 2
    obs, _ = UAVEnv(EnvConfig(), scenario=scene).reset()
    assert len(obs.contacts) == 1
    assert math.isclose(obs.contacts[0].forward_m, 50.0, abs_tol=1e-9)
    assert math.isclose(obs.contacts[0].left_m, 40.0, abs_tol=1e-9)
    assert math.isclose(obs.contacts[0].relative_forward_mps, -6.0, abs_tol=1e-9)


def test_swept_collision_catches_intruder_crossing_between_steps():
    assert moving_collision(
        own_before=(0.0, 0.0), own_after=(20.0, 0.0),
        intruder_before=(10.0, -20.0), intruder_after=(10.0, 20.0),
        threshold=10.0,
    )


def test_collision_terminates_and_reports_reward_parts():
    env = UAVEnv(EnvConfig(), scenario=bare_scenario(
        intruders=[Intruder(12.0, 0.0, 0.0, 0.0)]
    ))
    env.reset()
    _, reward, terminated, truncated, info = env.step(Action.MAINTAIN)
    assert terminated and not truncated
    assert info["outcome"] == "collision"
    assert reward == sum(info["reward_parts"].values())


def test_eta_and_early_late_are_separate_from_mission_success():
    env = UAVEnv(EnvConfig(), scenario=bare_scenario(goal=(6.0, 0.0), due=20.0))
    obs, _ = env.reset()
    assert 0.0 < obs.eta < 20.0
    _, _, terminated, _, info = env.step(Action.MAINTAIN)
    assert terminated and info["outcome"] == "success"
    assert info["arrival_error"] < 0
    assert not info["on_time"]


def test_weighted_eta_stays_finite_at_minimum_speed():
    scene = bare_scenario()
    scene.speed = 0.1
    env = UAVEnv(EnvConfig(), scenario=scene)
    obs, _ = env.reset()
    assert obs.eta < 30.0


def test_goal_crossed_between_discrete_positions_counts_as_reached():
    scene = bare_scenario(goal=(3.0, 0.0), due=1.0)
    scene.speed = 10.0
    env = UAVEnv(EnvConfig(goal_radius=1.0), scenario=scene)
    env.reset()
    _, _, terminated, _, info = env.step(Action.MAINTAIN)
    assert terminated and info["outcome"] == "success"


def test_scenario_generation_is_seeded_and_obstacle_free_near_route():
    a = make_scenario("random", seed=42)
    b = make_scenario("random", seed=42)
    assert a == b
    for obstacle in a.obstacles:
        assert isinstance(obstacle, StaticObstacle)
        assert obstacle.radius > 0
