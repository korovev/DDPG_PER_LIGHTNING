from buffers import Buffer
from buffer_utils import Experience, Experience_weight_idx

from typing import Iterator, Tuple

from torch.utils.data.dataset import IterableDataset


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated
    with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: Buffer, sample_size: int) -> None:
        self.buffer = buffer
        self.sample_size = sample_size  # sample_size

    def __iter__(self) -> Iterator[Tuple]:
        sampled_exps = self.buffer.sample(self.sample_size)
        states, actions, rewards, dones, new_states = ([] for i in range(5))
        for exp in sampled_exps:
            states.append(exp.state)
            actions.append(exp.action)
            rewards.append(exp.reward)
            dones.append(exp.done)
            new_states.append(exp.new_state)

        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]
