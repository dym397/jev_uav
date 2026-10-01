"""Seeded physical scene generation, independent of the policy."""

from dataclasses import dataclass, field
from math import cos, sin, tau

import numpy as np

from .collision import distance
from .obstacle import Intruder, StaticObstacle


@dataclass(eq=True)
class Scenario:
    start: tuple[float, float]
    heading: float
    speed: float
    goal: tuple[float, float]
    due_time: float
    intruders: list[Intruder] = field(default_factory=list)
    obstacles: list[StaticObstacle] = field(default_factory=list)
    world_size: float = 2000.0


def make_scenario(kind: str, seed: int, *, density_per_km2: float | None = None) -> Scenario:
    """A 100 m 4D leg inside the paper's 2 km square airspace."""
    if kind not in {"simple", "complex", "random"}:
        raise ValueError(f"Unknown scenario: {kind}")
    rng = np.random.default_rng(seed)
    world = 2000.0
    start = (900.0, 1000.0)
    goal = (1000.0, 1000.0)
    obstacles: list[StaticObstacle] = []
    while len(obstacles) < 5:
        x, y = rng.uniform(0, world, 2)
        radius = float(rng.uniform(8, 20))
        # Strategic planning in the paper avoids static obstacles before this 4D leg.
        if 850 <= x <= 1050 and abs(y - 1000) < radius + 30:
            continue
        obstacles.append(StaticObstacle(float(x), float(y), radius))

    intruders: list[Intruder] = []
    if kind == "simple":
        intruders.append(Intruder(950.0, 1040.0, 0.0, -4.0))
    elif kind == "complex":
        for _ in range(8):
            x = float(rng.uniform(920, 1020))
            y = float(rng.uniform(940, 1060))
            angle = float(rng.uniform(0, tau))
            speed = float(rng.uniform(2, 6))
            if distance((x, y), start) < 20 or distance((x, y), goal) < 20:
                continue
            intruders.append(Intruder(x, y, speed * cos(angle), speed * sin(angle)))
    else:
        density = 40.0 if density_per_km2 is None else density_per_km2
        count = int(round(density * (world / 1000) ** 2))
        for _ in range(count):
            x, y = rng.uniform(0, world, 2)
            angle = float(rng.uniform(0, tau))
            speed = float(rng.uniform(2, 6))
            if distance((x, y), start) < 20 or distance((x, y), goal) < 20:
                continue
            intruders.append(Intruder(float(x), float(y), speed * cos(angle), speed * sin(angle)))
    return Scenario(start, 0.0, 6.0, goal, 100.0 / 6.0, intruders, obstacles, world)
