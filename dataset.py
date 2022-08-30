from buffer import Buffer

from typing import Iterator, Tuple

from torch.utils.data.dataset import IterableDataset


class RLDataset(IterableDataset):
    """Iterable Dataset containing the ExperienceBuffer which will be updated
    with new experiences during training.

    Args:
        buffer: replay buffer
        sample_size: number of experiences to sample at a time
    """

    def __init__(self, buffer: Buffer, sample_size: int = 200) -> None:
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self) -> Iterator[Tuple]:
        states, actions, rewards, dones, new_states = self.buffer.sample(
            self.sample_size
        )
        for i in range(len(dones)):
            yield states[i], actions[i], rewards[i], dones[i], new_states[i]
