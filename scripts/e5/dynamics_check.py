"""Does the one-step actuation lag of UAVEnv.step limit E5? Same seeds, two orderings of the update.

    python scripts/e5/dynamics_check.py [--seeds 100]

lagged (UAVEnv, the paper's Eq. 5): the position advances with v_k, theta_k, then the action sets v_k+1, theta_k+1,
    so a chosen action first moves the UAV one step later.
immediate: the action sets v and theta first and the position advances with them in the same step.
For each: the oracle's solvable seeds (beam search over the simulator), the nine constant actions, and the seeds where
every first action already collides in step 1 or 2 (lost before any policy can matter). Nothing is written.
"""

import argparse
import copy
import math
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from config import EnvConfig, action_control  # noqa: E402
from environment.scenario import make_scenario  # noqa: E402
from environment.uav_env import UAVEnv  # noqa: E402
from uav_eval.e5 import E5_SEEDS, MAX_STEPS  # noqa: E402

MAINTAIN = 4


class ImmediateEnv(UAVEnv):
    def step(self, action):
        yaw_rate, acceleration = action_control(action)
        dt = self.config.dt
        self.speed = min(max(self.speed + acceleration * dt, self.config.speed_min), self.config.speed_max)
        self.heading = (self.heading + yaw_rate * dt + math.pi) % (2 * math.pi) - math.pi
        return super().step(MAINTAIN)   # moves with the new v, theta; MAINTAIN changes nothing after


def env_for(cls, seed):
    env = cls(EnvConfig(max_steps=MAX_STEPS), scenario=make_scenario("complex", seed))
    env.reset()
    return env


def constant(cls, seed, action):
    env = env_for(cls, seed)
    while True:
        _, _, terminated, truncated, info = env.step(action)
        if terminated or truncated:
            return info["outcome"], env.steps


def oracle(cls, seed, beam=64):
    frontier = [env_for(cls, seed)]
    for _ in range(MAX_STEPS):
        children = {}
        for env in frontier:
            for action in range(9):
                child = copy.deepcopy(env)
                _, _, terminated, truncated, info = child.step(action)
                if info["outcome"] == "success":
                    return True
                if terminated or truncated:
                    continue
                key = (round(child.position[0]), round(child.position[1]), round(child.speed), round(child.heading, 1))
                score = math.dist(child.scenario.goal, child.position)
                if key not in children or score < children[key][0]:
                    children[key] = (score, child)
        if not children:
            return False
        frontier = [env for _, env in sorted(children.values(), key=lambda item: item[0])[:beam]]
    return False


def doomed_early(cls, seed, within=2):
    """True if every action sequence collides within the first `within` steps."""
    frontier = [env_for(cls, seed)]
    for _ in range(within):
        alive = []
        for env in frontier:
            for action in range(9):
                child = copy.deepcopy(env)
                _, _, terminated, truncated, info = child.step(action)
                if info["outcome"] != "collision":
                    alive.append(child)
        if not alive:
            return True
        frontier = alive
    return False


parser = argparse.ArgumentParser()
parser.add_argument("--seeds", type=int, default=len(E5_SEEDS))
seeds = list(E5_SEEDS)[:parser.parse_args().seeds]
for name, cls in (("lagged", UAVEnv), ("immediate", ImmediateEnv)):
    solvable = [s for s in seeds if oracle(cls, s)]
    runs = {a: [constant(cls, s, a) for s in seeds] for a in range(9)}
    straight = {s for s, (o, _) in zip(seeds, runs[MAINTAIN]) if o == "success"}
    print(f"== {name}")
    print(f"oracle solvable {len(solvable)}/{len(seeds)}; of those need a manoeuvre (MAINTAIN fails) "
          f"{len(set(solvable) - straight)}")
    print("lost in step 1-2 whatever the actions:", sum(doomed_early(cls, s) for s in seeds))
    for a in range(9):
        outcomes = Counter(o for o, _ in runs[a])
        print(f"  constant_{a}: success {outcomes['success']:3} collision {outcomes['collision']:3} "
              f"timeout {outcomes['timeout']:3}  first collision step (median) "
              f"{sorted(n for o, n in runs[a] if o == 'collision')[len([1 for o, _ in runs[a] if o == 'collision']) // 2] if outcomes['collision'] else '-'}")
