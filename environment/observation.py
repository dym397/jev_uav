"""Paper vector and a semantic view of the same state for future agents."""

from dataclasses import dataclass
from math import atan2, cos, pi, sin

import numpy as np

from .collision import distance


def wrap_angle(angle: float) -> float:
    return (angle + pi) % (2 * pi) - pi


@dataclass(frozen=True)
class IntruderContact:
    """Current detected relative geometry and motion in own-UAV body axes."""

    forward_m: float
    left_m: float
    relative_forward_mps: float
    relative_left_mps: float


@dataclass(frozen=True)
class Observation:
    numeric: np.ndarray
    own_position: tuple[float, float]
    own_speed: float
    own_heading: float
    goal: tuple[float, float]
    current_time: float
    eta: float
    due_time: float
    nearest_distance: float | None
    contacts: tuple[IntruderContact, ...] = ()

    def state_to_text(self) -> str:
        """Render only the 14 values visible to the paper-configured D3QN.

        Physical metadata on this object is deliberately excluded so a text policy
        cannot receive extra position, clock or deadline information.
        """
        values = self.numeric
        if values.shape != (14,):
            raise ValueError("Paper text observation requires exactly 14 values")
        labels = ("heading_rad", "speed_mps", "goal_distance_ratio",
                  "goal_bearing_ccw_rad", "eta_remaining_ratio")
        fields = [f"{name}={float(value):.6g}" for name, value in zip(labels, values[:5])]
        fields.extend(f"sector_{index}={float(value):.6g}" for index, value in enumerate(values[5:]))
        return "; ".join(fields)


def observe(position, speed, heading, goal, start, due_time, now, intruders, config) -> Observation:
    goal_distance = distance(position, goal)
    leg_length = max(distance(start, goal), 1e-9)
    goal_bearing = atan2(goal[1] - position[1], goal[0] - position[0])
    relative_goal_angle = (goal_bearing - heading) % (2 * pi)
    raw_remaining = goal_distance / max(speed, config.speed_min)
    scheduled_remaining = max(due_time - now, 0.0)
    if raw_remaining > scheduled_remaining + config.on_time_window:
        weighted_speed = config.eta_speed_weight * (speed + config.speed_max)
    elif raw_remaining < scheduled_remaining - config.on_time_window:
        weighted_speed = config.eta_speed_weight * (speed + config.speed_min)
    else:
        weighted_speed = speed
    eta = now + goal_distance / max(weighted_speed, config.speed_min)
    normalized_time = (eta - now) / max(due_time, 1e-9)
    sectors = np.ones(config.sectors, dtype=np.float32)
    nearest = None
    contact_rows = []
    forward_x, forward_y = cos(heading), sin(heading)
    for intruder in intruders:
        d = distance(position, intruder.position)
        if d > config.detection_radius:
            continue
        nearest = d if nearest is None else min(nearest, d)
        bearing = atan2(intruder.y - position[1], intruder.x - position[0])
        relative = wrap_angle(bearing - heading)
        index = int(((relative + pi / config.sectors) % (2 * pi)) / (2 * pi / config.sectors))
        sectors[index] = min(sectors[index], d / config.detection_radius)
        dx, dy = intruder.x - position[0], intruder.y - position[1]
        dvx = intruder.vx - speed * forward_x
        dvy = intruder.vy - speed * forward_y
        contact_rows.append((d, IntruderContact(
            forward_m=dx * forward_x + dy * forward_y,
            left_m=-dx * forward_y + dy * forward_x,
            relative_forward_mps=dvx * forward_x + dvy * forward_y,
            relative_left_mps=-dvx * forward_y + dvy * forward_x,
        )))
    numeric = np.concatenate((np.asarray([
        heading, speed, goal_distance / leg_length, relative_goal_angle, normalized_time
    ], dtype=np.float32), sectors))
    contacts = tuple(contact for _, contact in sorted(contact_rows, key=lambda row: row[0]))
    return Observation(numeric, tuple(position), speed, heading, tuple(goal), now, eta,
                       due_time, nearest, contacts)
