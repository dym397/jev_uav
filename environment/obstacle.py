"""Physical entities; no decision logic."""

from dataclasses import dataclass


@dataclass(eq=True)
class Intruder:
    x: float
    y: float
    vx: float
    vy: float

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y

    def advance(self, dt: float, world_size: float) -> None:
        self.x += self.vx * dt
        self.y += self.vy * dt
        if self.x < 0 or self.x > world_size:
            self.x = min(max(self.x, 0.0), world_size)
            self.vx = -self.vx
        if self.y < 0 or self.y > world_size:
            self.y = min(max(self.y, 0.0), world_size)
            self.vy = -self.vy


@dataclass(frozen=True)
class StaticObstacle:
    x: float
    y: float
    radius: float

    @property
    def position(self) -> tuple[float, float]:
        return self.x, self.y
