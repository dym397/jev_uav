import numpy as np
import torch

from agents.d3qn_agent import D3QNAgent
from agents.replay_buffer import NStepReplayBuffer
from models.d3qn_network import DuelingQNetwork


def test_dueling_network_outputs_one_q_per_paper_action():
    net = DuelingQNetwork(14, 9)
    assert net(torch.zeros(2, 14)).shape == (2, 9)


def test_n_step_buffer_accumulates_discounted_reward_and_terminal():
    buf = NStepReplayBuffer(capacity=20, n_step=3, gamma=0.9, seed=1)
    state = np.zeros(14, dtype=np.float32)
    next_state = np.ones(14, dtype=np.float32)
    for i in range(3):
        buf.add(state, 0, 1.0, next_state, False)
    assert len(buf) == 1
    item = buf.sample(1)[0]
    assert abs(item.reward - (1 + .9 + .81)) < 1e-6
    assert item.steps == 3
    buf.add(state, 0, 2.0, next_state, True)
    assert len(buf) == 4
    assert any(t.done for t in buf.sample(4))


def test_double_q_target_uses_online_argmax_and_target_value():
    online = torch.tensor([[1.0, 5.0, 2.0]])
    target = torch.tensor([[9.0, 3.0, 8.0]])
    value = D3QNAgent.double_q_target(
        torch.tensor([2.0]), torch.tensor([False]),
        torch.tensor([1]), online, target, gamma=.5,
    )
    assert torch.allclose(value, torch.tensor([3.5]))
