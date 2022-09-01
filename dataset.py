from buffers import Buffer, PrioritizedReplayBuffer
from buffer_utils import Experience, Experience_weight_idx
from agent import Agent

from typing import Iterator, Tuple

from torch.utils.data.dataset import IterableDataset


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated
    with new experiences during training.

    Args:
        buffer: replay buffer
        batch_size: number of experiences to sample at a time
    """

    def __init__(self, batch_size: int) -> None:
        """
        Initialize buffer. If use_prioritized_buffer = 0 the normal buffer will
        be used, otherwise the prioritized experience replay buffer.
        """
        self.buffer = (
            PrioritizedReplayBuffer()
            if self.hparams.use_prioritized_buffer
            else Buffer()
        )

        """
        Initialize Agent to play the game
        """
        self.agent = Agent(env=self.env, buffer=self.buffer)
        self.total_reward = 0
        self.episode_reward = 0

    def __iter__(self) -> Iterator[Tuple]:
        sampled_exps = self.buffer.sample(self.batch_size)
        states, actions, rewards, dones, new_states = ([] for i in range(5))
        for exp in sampled_exps:
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.done)
            new_states.append(exp.new_state)

        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]

    def populate(self, steps: int = 1000) -> None:
        """Carries out several random steps through the environment to initially
        fill up the replay buffer with experiences. Called by Lightning Callback
        just before the traing starts.

        Args:
            steps : number of random steps to populate the buffer with
        """
        for _ in range(steps):
            self.agent.play_step(self.actor)
