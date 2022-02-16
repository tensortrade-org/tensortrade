import random
from collections import deque

import numpy as np


class BaseBuffer:
    """
    Base class for replay buffers.
    """

    def __init__(self, size, initial_size=None, batch_size=32):
        """
        Initialize replay buffer.
        Args:
            size: Buffer maximum size.
            initial_size: Buffer initial size to be filled before training starts.
                To be used by caller.
            batch_size: Size of the batch that should be used in get_sample() implementation.
        """
        assert (
            initial_size is None or initial_size > 0
        ), f'Buffer initial size should be > 0, got {initial_size}'
        assert size > 0, f'Buffer size should be > 0,  got {size}'
        assert batch_size > 0, f'Buffer batch size should be > 0, got {batch_size}'
        assert (
            batch_size <= size
        ), f'Buffer batch size `{batch_size}` should be <= size `{size}`'
        if initial_size:
            assert size >= initial_size, 'Buffer initial size exceeds max size'
        self.size = size
        self.initial_size = initial_size or size
        self.batch_size = batch_size
        self.current_size = 0

    def append(self, *args):
        """
        Add experience to buffer.
        Args:
            *args: Items to store, types are implementation specific.

        """
        raise NotImplementedError(
            f'append() should be implemented by {self.__class__.__name__} subclasses'
        )

    def get_sample(self):
        """
        Sample from stored experience.

        Returns:
            Sample as numpy array.
        """
        raise NotImplementedError(
            f'get_sample() should be implemented by {self.__class__.__name__} subclasses'
        )


class ReplayBuffer1(BaseBuffer):
    """
    deque-based replay buffer that holds state transitions
    """

    def __init__(self, size, **kwargs):
        """
        Initialize replay buffer.
        Args:
            size: Buffer maximum size.
            **kwargs: kwargs passed to BaseBuffer.
        """
        super(ReplayBuffer1, self).__init__(size, **kwargs)
        self.main_buffer = deque(maxlen=size)
        self.temp_buffer = []

    def append(self, *args):
        """
        Append experience and auto-allocate to temp buffer / main buffer(self)
        Args:
            *args: Items to store

        Returns:
            None
        """
        self.main_buffer.append(args)
        self.current_size = len(self.main_buffer)

    def get_sample(self):
        """
        Sample from stored experience.

        Returns:
            Same number of args passed to append, having self.batch_size as
            first shape.
        """
        memories = random.sample(self.main_buffer, self.batch_size)
        if self.batch_size > 1:
            return [np.array(item) for item in zip(*memories)]
        return memories[0]


class ReplayBuffer2(BaseBuffer):
    """
    numpy-based replay buffer added for compatibility with tensorflow shortcomings.
    """

    def __init__(self, size, slots, **kwargs):
        """
        Initialize replay buffer.

        Args:
            size: Buffer maximum size.
            slots: Number of args that will be passed to self.append()
            **kwargs: kwargs passed to BaseBuffer.
        """
        super(ReplayBuffer2, self).__init__(size, **kwargs)
        self.slots = [np.array([])] * slots
        self.current_size = 0

    def append(self, *args):
        """
        Add experience to buffer.
        Args:
            *args: Items to store.

        Returns:
            None
        """
        for i, arg in enumerate(args):
            if not isinstance(arg, np.ndarray):
                arg = np.array([arg])
            if not self.slots[i].shape[0]:
                self.slots[i] = np.zeros((self.size, *arg.shape), arg.dtype)
            self.slots[i][self.current_size % self.size] = arg.copy()
        if self.current_size < self.size:
            self.current_size += 1

    def get_sample(self):
        """
        Sample from stored experience.

        Returns:
            Same number of args passed to append, having self.batch_size as
            first shape.
        """
        indices = np.random.randint(
            0, min(self.current_size, self.size), self.batch_size
        )
        return [slot[indices] for slot in self.slots]
