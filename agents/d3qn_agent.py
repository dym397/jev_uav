"""Replaceable D3QN decision model; knows nothing about UAV dynamics."""

from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn

from config import AgentConfig
from models.d3qn_network import DuelingQNetwork

from .replay_buffer import NStepReplayBuffer


class D3QNAgent:
    def __init__(self, state_size: int = 14, action_size: int = 9,
                 config: AgentConfig | None = None, *, seed: int = 0, device: str | None = None):
        self.config = config or AgentConfig()
        self.state_size = state_size
        self.action_size = action_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        torch.manual_seed(seed)
        self.random = np.random.default_rng(seed)
        self.online = DuelingQNetwork(state_size, action_size, self.config.hidden_size).to(self.device)
        self.target = DuelingQNetwork(state_size, action_size, self.config.hidden_size).to(self.device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.config.learning_rate)
        self.loss_fn = nn.MSELoss()
        self.replay = NStepReplayBuffer(self.config.buffer_size, self.config.n_step, self.config.gamma, seed)
        self.total_steps = 0
        self.updates = 0
        self.eval_mode = False

    @property
    def epsilon(self) -> float:
        fraction = min(self.total_steps / max(self.config.epsilon_decay_steps, 1), 1.0)
        return self.config.epsilon_start + fraction * (self.config.epsilon_end - self.config.epsilon_start)

    def set_eval_mode(self, enabled: bool = True) -> None:
        self.eval_mode = enabled

    def select_action(self, state) -> int:
        """A future JevAgent can implement the same select_action(state) interface."""
        if not self.eval_mode and self.random.random() < self.epsilon:
            return int(self.random.integers(self.action_size))
        vector = state.numeric if hasattr(state, "numeric") else state
        tensor = torch.as_tensor(np.asarray(vector, dtype=np.float32), device=self.device).unsqueeze(0)
        with torch.no_grad():
            return int(self.online(tensor).argmax(dim=1).item())

    def observe(self, state, action: int, reward: float, next_state, done: bool) -> float | None:
        old = state.numeric if hasattr(state, "numeric") else state
        new = next_state.numeric if hasattr(next_state, "numeric") else next_state
        self.replay.add(old, action, reward, new, done)
        self.total_steps += 1
        if self.total_steps % self.config.train_interval == 0 and len(self.replay) >= self.config.batch_size:
            return self.learn()
        return None

    @staticmethod
    def double_q_target(rewards: torch.Tensor, dones: torch.Tensor, steps: torch.Tensor,
                        online_next: torch.Tensor, target_next: torch.Tensor, gamma: float) -> torch.Tensor:
        best_actions = online_next.argmax(dim=1, keepdim=True)
        target_values = target_next.gather(1, best_actions).squeeze(1)
        discount = torch.pow(torch.as_tensor(gamma, device=rewards.device), steps)
        return rewards + (~dones).float() * discount * target_values

    def learn(self) -> float:
        batch = self.replay.sample(self.config.batch_size)
        states = torch.as_tensor(np.stack([t.state for t in batch]), device=self.device)
        actions = torch.as_tensor([t.action for t in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([t.reward for t in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([t.next_state for t in batch]), device=self.device)
        dones = torch.as_tensor([t.done for t in batch], dtype=torch.bool, device=self.device)
        steps = torch.as_tensor([t.steps for t in batch], dtype=torch.float32, device=self.device)
        q_values = self.online(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            expected = self.double_q_target(rewards, dones, steps,
                                            self.online(next_states), self.target(next_states),
                                            self.config.gamma)
        loss = self.loss_fn(q_values, expected)
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        self.updates += 1
        return float(loss.item())

    def end_episode(self) -> None:
        # Table 2: target network update upon completion of each round.
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({"online": self.online.state_dict(), "target": self.target.state_dict(),
                    "optimizer": self.optimizer.state_dict(), "config": asdict(self.config),
                    "state_size": self.state_size, "action_size": self.action_size,
                    "total_steps": self.total_steps, "updates": self.updates}, path)

    @classmethod
    def load(cls, path: str | Path, *, device: str | None = None) -> "D3QNAgent":
        checkpoint = torch.load(path, map_location=device or "cpu", weights_only=False)
        agent = cls(checkpoint["state_size"], checkpoint["action_size"],
                    AgentConfig(**checkpoint["config"]), device=device)
        agent.online.load_state_dict(checkpoint["online"])
        agent.target.load_state_dict(checkpoint["target"])
        agent.optimizer.load_state_dict(checkpoint["optimizer"])
        agent.total_steps = checkpoint["total_steps"]
        agent.updates = checkpoint["updates"]
        return agent
