from buffer import Buffer
from agent_utils import OUActionNoise
from buffer_utils import Experience

import gym
import numpy as np
from typing import Tuple

import torch
from torch import Tensor, nn


class Agent:
    """Base Agent class handling the interaction with the environment."""

    def __init__(self, env: gym.Env, buffer: Buffer) -> None:
        """
        Args:
            env: training environment
            replay_buffer: replay buffer storing experiences
        """
        self.env = env
        self.buffer = buffer
        self.reset()
        self.state = self.env.reset()

        _std = 0.8
        self.noise_model = OUActionNoise(
            mean=np.zeros(1) * 1.2, std_deviation=float(_std) * np.ones(1)
        )

    def reset(self) -> None:
        """Resents the environment and updates the state."""
        self.state = self.env.reset()

    def get_action(self, net: nn.Module, device: str) -> int:
        """
        Using the given network, decide what action to carry out using the
        ornstein_uhlenbeck noise to perturb the action.

        Args:
            -   net: DQN network
            -   epsilon: value to determine likelihood of taking a random action
            -   device: current device
        """
        state = torch.tensor([self.state])
        if device not in ["cpu"]:
            state = state.cuda(device)
        sampled_actions = net(state).numpy() + self.noise_model

        # make sure action is within bounds
        action_upper_bound = self.env.action_space.high[0]
        action_lower_bound = self.env.action_space.low[0]
        legal_action = np.clip(
            sampled_actions,
            action_upper_bound,
            action_lower_bound,
        )

        return legal_action

    @torch.no_grad()
    def play_step(
        self,
        net: nn.Module,
        device: str = "cpu",
    ) -> Tuple[float, bool]:
        """
        Carries out a single interaction step between the agent and the environment.

        Args:
            -   net: DQN network
            -   device: current device
        """

        action = self.get_action(net, device)

        # do step in the environment
        new_state, reward, done, _ = self.env.step(action)

        exp = Experience(self.state, action, reward, done, new_state)
        self.replay_buffer.append(exp)

        self.state = new_state
        if done:
            self.reset()
        return reward, done
