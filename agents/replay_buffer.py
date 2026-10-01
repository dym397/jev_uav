"""Uniform experience replay with the paper's five-step returns."""

from collections import deque
from dataclasses import dataclass
from random import Random

import numpy as np


@dataclass(frozen=True)
class Transition:
    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    steps: int


class NStepReplayBuffer:
    def __init__(self, capacity: int, n_step: int, gamma: float, seed: int = 0):
        if capacity < 1 or n_step < 1:
            raise ValueError("capacity and n_step must be positive")
        self.storage: deque[Transition] = deque(maxlen=capacity)
        self.pending: deque[tuple[np.ndarray, int, float, np.ndarray, bool]] = deque()
        self.n_step = n_step
        self.gamma = gamma
        self.random = Random(seed)

    def __len__(self) -> int:
        return len(self.storage)

    def add(self, state, action: int, reward: float, next_state, done: bool) -> None:
        self.pending.append((np.array(state, dtype=np.float32, copy=True), int(action), float(reward),
                             np.array(next_state, dtype=np.float32, copy=True), bool(done)))
        if len(self.pending) >= self.n_step:
            self._emit()
        if done:
            while self.pending:
                self._emit()

    def _emit(self) -> None:
        reward = 0.0
        next_state = self.pending[0][3]
        done = False
        steps = 0
        for _, _, step_reward, following, terminal in list(self.pending)[:self.n_step]:
            reward += (self.gamma ** steps) * step_reward
            next_state = following
            steps += 1
            done = terminal
            if terminal:
                break
        state, action, _, _, _ = self.pending.popleft()
        self.storage.append(Transition(state, action, reward, next_state, done, steps))

    def sample(self, batch_size: int) -> list[Transition]:
        return self.random.sample(list(self.storage), batch_size)
