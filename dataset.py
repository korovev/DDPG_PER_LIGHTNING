from buffers import Buffer, PrioritizedReplayBuffer
from buffer_utils import Experience, Experience_weight_idx
from agent import Agent

from typing import Iterator, Tuple

from torch.utils.data.dataset import IterableDataset

import gym
from pytorch_lightning import LightningModule


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated
    with new experiences during training.

    Args:
        buffer: replay buffer
        batch_size: number of experiences to sample at a time
    """

    def __init__(
        self,
        buffer: Buffer,
        batch_size: int,
        env: gym.Env,
        actor_net: LightningModule,
        train_episodes: int,
    ) -> None:
        """
        Initialize buffer. If use_prioritized_buffer = 0 the normal buffer will
        be used, otherwise the prioritized experience replay buffer.
        """
        super(RLDataset).__init__()
        self.buffer = buffer
        self.batch_size = batch_size

        self.agent = Agent(env=env, buffer=buffer)
        self.total_reward = 0
        self.episode_reward = 0

        self.train_episodes = train_episodes
        self.episodes_done = 0

        self.actor_net = actor_net

    def __iter__(self) -> Iterator[Tuple]:
        """
        Agent has to be implemented here. Super ugly but with Lightning and RL
        there are not non-wacky solutions.
        """
        # TODO implement commented stuff. Find a solution for logging.
        # step through environment with agent
        reward, done = self.agent.play_step(self.actor_net)
        self.episode_reward += reward
        # self.log("episode reward", self.episode_reward)
        if done:
            self.total_reward = self.episode_reward
            self.episode_reward = 0
            # self.log_dict(
            #         {
            #             "reward": reward,
            #         }
            #     )
            # self.log("total_reward", self.total_reward, prog_bar=True)
            self.episodes_done += 1
            # self.log("episodes done", self.episodes_done)
            if self.episodes_done >= self.train_episodes:
                return

        sampled_exps = self.buffer.sample(self.batch_size)

        yield sampled_exps
