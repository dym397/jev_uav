"""2D ETA tactical conflict environment. No agent or network imports."""

from copy import deepcopy
from math import cos, pi, sin

from config import Action, EnvConfig, action_control

from .collision import distance, moving_collision, point_to_segment_distance, static_collision
from .observation import observe
from .scenario import Scenario, make_scenario


class UAVEnv:
    def __init__(self, config: EnvConfig | None = None, *, scenario: Scenario | None = None):
        self.config = config or EnvConfig()
        self.template = deepcopy(scenario) if scenario is not None else None
        self.reset()

    def reset(self, *, seed: int = 0, scenario_kind: str = "random", density_per_km2: float | None = None):
        self.scenario = deepcopy(self.template) if self.template is not None else make_scenario(
            scenario_kind, seed, density_per_km2=density_per_km2
        )
        self.position = tuple(self.scenario.start)
        self.heading = float(self.scenario.heading)
        self.speed = float(self.scenario.speed)
        self.time = 0.0
        self.steps = 0
        self.path_length = 0.0
        self.trajectory = [self.position]
        self.intruder_trajectories = [[i.position] for i in self.scenario.intruders]
        self.finished = False
        return self._observation(), self._info("running", None)

    def _observation(self):
        return observe(
            self.position, self.speed, self.heading, self.scenario.goal,
            self.scenario.start, self.scenario.due_time, self.time,
            self.scenario.intruders, self.config,
        )

    def _info(self, outcome: str, reward_parts):
        error = self.time - self.scenario.due_time if outcome == "success" else None
        return {
            "outcome": outcome,
            "on_time": error is not None and abs(error) <= self.config.on_time_window,
            "arrival_error": error,
            "own_position": self.position,
            "own_speed": self.speed,
            "own_heading": self.heading,
            "path_length": self.path_length,
            "reward_parts": reward_parts or {"avoidance": 0.0, "temporal": 0.0, "mission": 0.0},
        }

    def step(self, action: int | Action):
        if self.finished:
            raise RuntimeError("Episode ended; call reset() before step().")
        previous = self._observation()
        previous_position = self.position
        intruder_before = [i.position for i in self.scenario.intruders]
        yaw_rate, acceleration = action_control(action)
        dt = self.config.dt
        # Equation (5) uses v_k and theta_k for position, then updates v and theta.
        self.position = (
            self.position[0] + self.speed * dt * cos(self.heading),
            self.position[1] + self.speed * dt * sin(self.heading),
        )
        self.speed = min(max(self.speed + acceleration * dt, self.config.speed_min), self.config.speed_max)
        self.heading = (self.heading + yaw_rate * dt + pi) % (2 * pi) - pi
        for intruder in self.scenario.intruders:
            intruder.advance(dt, self.scenario.world_size)
        self.time += dt
        self.steps += 1
        self.path_length += distance(previous_position, self.position)
        self.trajectory.append(self.position)
        for track, intruder in zip(self.intruder_trajectories, self.scenario.intruders):
            track.append(intruder.position)
        current = self._observation()

        collision = any(moving_collision(previous_position, self.position, before, intruder.position,
                                         self.config.collision_distance)
                        for before, intruder in zip(intruder_before, self.scenario.intruders))
        static_hit = any(static_collision(previous_position, self.position, obstacle,
                                          self.config.collision_distance)
                         for obstacle in self.scenario.obstacles)
        out_of_bounds = not (0 <= self.position[0] <= self.scenario.world_size and
                             0 <= self.position[1] <= self.scenario.world_size)
        reached = point_to_segment_distance(self.scenario.goal, previous_position, self.position) <= self.config.goal_radius
        timed_out = self.steps >= self.config.max_steps
        outcome = "collision" if collision or static_hit else (
            "out_of_bounds" if out_of_bounds else (
                "success" if reached else ("timeout" if timed_out else "running")
            )
        )
        terminated = outcome in {"collision", "out_of_bounds", "success"}
        truncated = outcome == "timeout"
        self.finished = terminated or truncated
        rewards = self._reward(previous, current, outcome, static_hit)
        return current, sum(rewards.values()), terminated, truncated, self._info(outcome, rewards)

    def _reward(self, previous, current, outcome: str, static_hit: bool) -> dict[str, float]:
        old_sectors = previous.numeric[5:]
        new_sectors = current.numeric[5:]
        occupied = new_sectors < 1
        avoidance = 0.0
        if occupied.any():
            trends = new_sectors[occupied] - old_sectors[occupied]
            avoidance += float((2 * (trends >= 0).astype(float) - 1).mean() * self.config.c1)
            avoidance -= float((1 - new_sectors[occupied]).mean() * self.config.danger_weight)
        if outcome == "collision":
            avoidance -= self.config.c2 if static_hit else 2 * self.config.c2

        arrival_delta = self.scenario.due_time - current.eta
        early = max(arrival_delta - self.config.on_time_window, 0.0)
        late = max(-arrival_delta - self.config.on_time_window, 0.0)
        temporal = -self.config.early_weight * early - self.config.late_weight * late

        if current.nearest_distance is None:
            goal_change = distance(previous.own_position, self.scenario.goal) - distance(current.own_position, self.scenario.goal)
            mission = self.config.progress_weight * goal_change
            angle = (current.numeric[3] + pi) % (2 * pi) - pi
            mission += self.config.heading_reward * cos(angle)
        else:
            # Safety-first mission shaping: do not reward progress toward the goal during a threat.
            change = current.nearest_distance - (previous.nearest_distance or current.nearest_distance)
            mission = self.config.progress_weight * change
        if outcome == "success":
            mission += self.config.goal_bonus
        return {"avoidance": avoidance, "temporal": temporal, "mission": mission}
