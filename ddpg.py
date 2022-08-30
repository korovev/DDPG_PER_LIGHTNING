import gym
import numpy as np
import pandas as pd
import torch
from IPython.display import display
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.loggers import CSVLogger
from torch import Tensor, nn
import torch.nn.functional as F
from torch.optim import Adam, Optimizer
from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset


class DDPG_Actor(nn.Module):
    """Actor Model of DDPG"""

    def __init__(self, state_dim: int, action_dim: int, action_max: float):
        """
        Args:
            -   state_dim : dimensions of the state space according to specific
                environment;
            -   action_dim : dimensions of the action space according to specific
                environment;
            -   action_max : upper limit of the actions values according to specific
                environment
        """
        super(DDPG_Actor, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 256),
            nn.ReLU(),
            nn.Linear(256, 256),
            nn.ReLU(),
            nn.Linear(256, action_dim),
            nn.Tanh(),
        )
        # TODO check if weights are actually initialized well like this
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            module.weight.data.uniform_(-3e-3, 3e-3)
            if module.bias is not None:
                module.bias.data.zero_()

    def forward(self, state, action_max: float):
        return self.net(state.float()) * action_max


class DDPG_Critic(nn.Module):
    """Critic Model of DDPG"""

    def __init__(self, state_dim: int, action_dim: int):
        """
        Args:
            -   state_dim : dimensions of the state space according to specific
                environment;
            -   action_dim : dimensions of the action space according to specific
                environment;
        """
        super(DDPG_Critic, self).__init__()
        self.l1 = nn.Linear(state_dim, 400)
        self.l2 = nn.Linear(400 + action_dim, 300)
        self.l3 = nn.Linear(300, action_dim)

    def forward(self, state, action):
        x = F.relu(self.l1(state))
        x = torch.cat([x, action], 1)
        x = F.relu(self.l2(x))
        x = self.l3(x)
        return x


class DDPG(LightningModule):
    """DDPG Model"""

    def __init__(
        self,
        batch_size: int = 64,
        actor_lr: float = 1e-4,
        critic_lr: float = 5e-4,
        env: str = "Pendulum-v1",
        gamma: float = 0.99,
        tau: float = 5e-3,
        train_episodes: float = 300,
        use_prioritized_buffer: bool = 0,
    ) -> None:
        """
        Args:
            -   batch_size : dimensions of the batches to use;
            -   actor_lr : learning rate of the actor model;
            -   critic_lr : learning rate of the critic model;
            -   env : Gym environment to use. Default: Pendulum-1;
            -   gamma : discount factor;
            -   tau : update the weights of the target networks with the weights
                of the original networks W_n * tau;
            -   train_episodes : on how many game episode training the model;
            -   use_prioritized_buffer : whether to use the Prioritized Experience
                Buffer or just a normal one;


        """
        super(DDPG, self).__init__()
        self.save_hyperparameters()

        """ Env and game parameters """
        self.env = gym.make(self.hparams.env)
        obs_size = self.env.observation_space.shape[0]
        n_actions = self.env.action_space.shape[0]
        action_upper_bound = self.env.action_space.high[0]

        """ Buffer parameters """
        self.use_prioritized_buffer = self.hparams.use_prioritized_buffer

        """ 
        Initialize actor, critic, target actor and target critic. Targets are 
        assigned the same weights of the original networks. 
        """
        # Actor
        self.actor = DDPG_Actor(
            state_dim=obs_size, action_dim=n_actions, action_max=action_upper_bound
        )
        # Critic
        self.critic = DDPG_Critic(state_dim=obs_size, action_dim=n_actions)
        # Target actor and weights equalization
        self.target_actor = DDPG_Actor(
            state_dim=obs_size, action_dim=n_actions, action_max=action_upper_bound
        )
        self.target_actor.load_state_dict(self.actor.state_dict())
        # Target critic and weights equalization
        self.target_critic = DDPG_Critic(state_dim=obs_size, action_dim=n_actions)
        self.target_critic.load_state_dict(self.critic.state_dict())


if __name__ == "__main__":
    foo = DDPG()
    print(foo.actor.net[0].weight == foo.target_actor.net[0].weight)
