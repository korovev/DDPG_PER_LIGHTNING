from buffer import Buffer, PrioritizedReplayBuffer
from agent import Agent
from dataset import RLDataset

from typing import Tuple, OrderedDict, List

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


class DDPGActor(nn.Module):
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
        super(DDPGActor, self).__init__()
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


class DDPGCritic(nn.Module):
    """Critic Model of DDPG"""

    def __init__(self, state_dim: int, action_dim: int):
        """
        Args:
            -   state_dim : dimensions of the state space according to specific
                environment;
            -   action_dim : dimensions of the action space according to specific
                environment;
        """
        super(DDPGCritic, self).__init__()
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
        sync_rate: int = 10,
        episode_length: int = 200,
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

        # """ Buffer parameters """
        # self.use_prioritized_buffer = self.hparams.use_prioritized_buffer
        # self.batch_size = self.hparams.batch_size

        """ 
        Initialize actor, critic, target actor and target critic. Targets are 
        assigned the same weights of the original networks. 
        """
        # Actor
        self.actor = DDPGActor(
            state_dim=obs_size, action_dim=n_actions, action_max=action_upper_bound
        )
        # Critic
        self.critic = DDPGCritic(state_dim=obs_size, action_dim=n_actions)
        # Target actor and weights equalization
        self.target_actor = DDPGActor(
            state_dim=obs_size, action_dim=n_actions, action_max=action_upper_bound
        )
        self.target_actor.load_state_dict(self.actor.state_dict())
        # Target critic and weights equalization
        self.target_critic = DDPGCritic(state_dim=obs_size, action_dim=n_actions)
        self.target_critic.load_state_dict(self.critic.state_dict())

        """
        Initialize buffer. If use_prioritized_buffer = 0 the normal buffer will 
        be used, otherwise the prioritized experience replay buffer.
        """
        self.buffer = (
            PrioritizedReplayBuffer if self.hparams.use_prioritized_buffer else Buffer
        )

        """
        Initialize Agent to play the game
        """
        self.agent = Agent(env=self.env, buffer=self.buffer)
        self.total_reward = 0
        self.episode_reward = 0

    def forward(self, x: Tensor) -> Tensor:
        """Passes in a state x through the network and gets the q_values of each
        action as an output.

        Args:
            x: environment state
        """
        out = self.net(x)
        return out

    def ddpg_loss(self, batch: Tuple[Tensor, Tensor]) -> Tuple[Tensor, Tensor]:
        """
        Calculates the mse loss using a mini batch from the replay buffer. See
        pseudo-code formulas.

        Args:
            batch: current mini batch of replay data
        """
        state_batch, action_batch, reward_batch, dones, next_state_batch = batch

        target_actions = self.target_actor(next_state_batch)
        y = reward_batch + self.gamma * self.target_critic(
            [next_state_batch, target_actions]
        )
        critic_value = self.critic([state_batch, action_batch])
        critic_loss = nn.MSELoss()(y, critic_value)

        actions = self.actor(state_batch)
        critic_value = self.critic([state_batch, actions])
        actor_loss = -torch.mean(critic_value)

        return critic_loss, actor_loss

    def training_step(
        self, batch: Tuple[Tensor, Tensor], nb_batch
    ) -> Tuple[OrderedDict, OrderedDict]:
        """
        Carries out a single step through the environment to update the replay buffer.
        Then calculates loss based on the minibatch recieved.

        Args:
            batch : current mini batch of replay data
            nb_batch : batch number
        """
        device = self.get_device(batch)
        # epsilon = self.get_epsilon(
        #     self.hparams.eps_start, self.hparams.eps_end, self.hparams.eps_last_frame
        # )
        # self.log("epsilon", epsilon)

        # step through environment with agent
        # reward, done = self.agent.play_step(self.net, epsilon, device)
        reward, done = self.agent.play_step(self.net, device)
        self.episode_reward += reward
        self.log("episode reward", self.episode_reward)

        batch_indices = np.random.choice(len(self.buffer), self.batch_size)
        batch = self.buffer[batch_indices]

        # calculates training loss
        critic_loss, actor_loss = self.ddpg_loss(batch)

        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0

        # FIXME Not sure what this is for, investigate. Soft update of target network
        # if self.global_step % self.hparams.sync_rate == 0:
        #     self.target_net.load_state_dict(self.net.state_dict())

        self.log_dict(
            {
                "reward": reward,
                "train_critic_loss": critic_loss,
                "train_actor_loss": actor_loss,
            }
        )
        self.log("total_reward", self.total_reward, prog_bar=True)
        self.log("steps", self.global_step, logger=False, prog_bar=True)

        return critic_loss, actor_loss

    def configure_optimizers(self) -> List[Optimizer]:
        """Initialize Adam optimizer."""
        critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.hparams.critic_lr
        )
        actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.hparams.actor_lr
        )
        # optimizer = Adam(self.net.parameters(), lr=self.hparams.lr)
        return critic_optimizer, actor_optimizer

    def __dataloader(self) -> DataLoader:
        """Initialize the Replay Buffer dataset used for retrieving experiences."""
        dataset = RLDataset(self.buffer, self.hparams.episode_length)
        dataloader = DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
        )
        return dataloader

    def train_dataloader(self) -> DataLoader:
        """Get train loader."""
        return self.__dataloader()

    def get_device(self, batch) -> str:
        """Retrieve device currently being used by minibatch."""
        return batch[0].device.index if self.on_gpu else "cpu"
