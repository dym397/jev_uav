"""Paper parameters and explicit reconstruction defaults."""

from dataclasses import dataclass
from enum import IntEnum
from math import pi


class Action(IntEnum):
    TURN_RIGHT_DECELERATE = 0
    TURN_RIGHT = 1
    TURN_RIGHT_ACCELERATE = 2
    DECELERATE = 3
    MAINTAIN = 4
    ACCELERATE = 5
    TURN_LEFT_DECELERATE = 6
    TURN_LEFT = 7
    TURN_LEFT_ACCELERATE = 8


ACTION_SPACE = {action.value: action.name.replace("_", " ").title() for action in Action}
YAW_RATES = (-pi / 30, 0.0, pi / 30)
ACCELERATIONS = (-3.0, 0.0, 3.0)


def action_control(action: int | Action) -> tuple[float, float]:
    value = Action(action).value
    return YAW_RATES[value // 3], ACCELERATIONS[value % 3]


@dataclass(frozen=True)
class EnvConfig:
    dt: float = 1.0  # Not specified by the paper.
    detection_radius: float = 100.0
    collision_distance: float = 10.0
    speed_min: float = 0.1
    speed_max: float = 10.0
    sectors: int = 9
    goal_radius: float = 5.0  # Not specified by the paper.
    max_steps: int = 120  # Not specified by the paper.
    on_time_window: float = 10.0
    eta_speed_weight: float = 0.5  # alpha_v in Eq. (25) is not specified.
    c1: float = 0.1
    c2: float = 20.0
    danger_weight: float = 1.0
    early_weight: float = 0.05
    late_weight: float = 0.05
    goal_bonus: float = 30.0
    heading_reward: float = 0.1
    progress_weight: float = 0.2


@dataclass(frozen=True)
class AgentConfig:
    learning_rate: float = 0.00005
    gamma: float = 0.99
    buffer_size: int = 1_000_000
    batch_size: int = 256
    n_step: int = 5
    train_interval: int = 10
    hidden_size: int = 128  # Not specified by the paper.
    epsilon_start: float = 1.0  # Not specified by the paper.
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 100_000
