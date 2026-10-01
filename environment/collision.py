"""Point and swept-segment collision checks."""

import numpy as np


def distance(a, b) -> float:
    return float(np.linalg.norm(np.asarray(a, dtype=float) - np.asarray(b, dtype=float)))


def point_to_segment_distance(point, start, end) -> float:
    start_array = np.asarray(start, dtype=float)
    delta = np.asarray(end, dtype=float) - start_array
    denominator = float(delta @ delta)
    fraction = 0.0 if denominator == 0 else float(np.clip(((np.asarray(point, dtype=float) - start_array) @ delta) / denominator, 0, 1))
    return float(np.linalg.norm(start_array + fraction * delta - np.asarray(point, dtype=float)))


def moving_collision(own_before, own_after, intruder_before, intruder_after, threshold: float) -> bool:
    relative_start = np.asarray(own_before, dtype=float) - np.asarray(intruder_before, dtype=float)
    relative_delta = (np.asarray(own_after, dtype=float) - np.asarray(own_before, dtype=float)) - (
        np.asarray(intruder_after, dtype=float) - np.asarray(intruder_before, dtype=float)
    )
    denominator = float(relative_delta @ relative_delta)
    fraction = 0.0 if denominator == 0 else float(np.clip(-(relative_start @ relative_delta) / denominator, 0, 1))
    return float(np.linalg.norm(relative_start + fraction * relative_delta)) < threshold


def static_collision(own_before, own_after, obstacle, threshold: float) -> bool:
    return point_to_segment_distance(obstacle.position, own_before, own_after) < threshold + obstacle.radius
