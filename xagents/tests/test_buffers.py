import numpy as np
import pytest
from xagents.utils.buffers import BaseBuffer, ReplayBuffer2


def get_buffer_kwargs(size, buffer_type, initial_size, batch_size):
    """
    Construct buffer creation kwargs.
    Args:
        size: Buffer max size
        buffer_type: ReplayBuffer1 or ReplayBuffer2
        initial_size: Buffer initial pre-train fill size.
        batch_size: Buffer batch size when get_sample() is called.

    Returns:
        buffer kwargs as dict.
    """
    buffer_kwargs = {
        'size': size,
        'initial_size': initial_size,
        'batch_size': batch_size,
    }
    if buffer_type == ReplayBuffer2:
        buffer_kwargs['slots'] = 1
    return buffer_kwargs


def assert_sample_shapes_match(buffer, shapes):
    """
    Ensure buffer sample shapes match the expected ones.
    Args:
        buffer: ReplayBuffer1 or ReplayBuffer2 instance.
        shapes: 1 observation list of shapes.
    """
    sample = buffer.get_sample()
    size = sample[0].shape[0]
    for shape, subsample in zip(shapes, sample):
        assert np.squeeze(subsample).shape == (size, *shape)


class TestBaseBuffer:
    @pytest.mark.parametrize(
        'size, initial_size, batch_size, exception, exception_kw',
        [
            [100, 0, 32, AssertionError, 'Buffer initial size should be > 0, got'],
            [-100, 100, 32, AssertionError, 'Buffer size should be > 0'],
            [10, None, 32, AssertionError, 'should be <= size'],
            [100, 200, 32, AssertionError, 'Buffer initial size exceeds max size'],
            [100, 50, 32, None, None],
            [100, None, 32, None, None],
        ],
    )
    def test_sizes(
        self, buffer_type, size, initial_size, batch_size, exception, exception_kw
    ):
        """
        Test size parameters match the expected ones, and proper exceptions
        are raised.

        Args:
            buffer_type: ReplayBuffer1 or ReplayBuffer2
            size: Buffer max size.
            initial_size: Buffer initial pre-train fill size.
            batch_size: Buffer batch size when get_sample() is called.
            exception: Exception raised due to improper size parameters.
            exception_kw: exception keywords that are expected to be displayed.
        """
        buffer_kwargs = get_buffer_kwargs(size, buffer_type, initial_size, batch_size)
        if exception:
            with pytest.raises(exception, match=fr'{exception_kw}'):
                buffer_type(**buffer_kwargs)
        else:
            buffer = buffer_type(**buffer_kwargs)
            assert buffer.size == size
            assert (
                buffer.initial_size == buffer.size if not initial_size else initial_size
            )
            assert buffer.batch_size == batch_size

    def test_abstract_methods(self):
        """
        Test Ensure BaseBuffer abstract methods raise NotImplementedError
        """
        buffer = BaseBuffer(32)
        with pytest.raises(NotImplementedError, match=r'should be implemented'):
            buffer.append(1)
        with pytest.raises(NotImplementedError, match=r'should be implemented'):
            buffer.get_sample()


@pytest.mark.usefixtures('buffer1', 'observations')
class TestBuffer1:
    """
    Test ReplayBuffer1 methods.
    """

    @pytest.mark.parametrize(
        'args',
        [
            [1, 2],
            [1, 2, 3, 4],
            [np.random.randint(0, 1000, 10) for _ in range(5)],
            [np.random.randint(0, 1000, 10), 15, 11],
            [[np.random.randint(0, 1000, 10) for _ in range(5)]],
        ],
    )
    def test_append(self, args):
        """
        Test ReplayBuffer1.append
        Args:
            args: list of dummy observations.
        """
        self.buffer.append(*args)
        for arg, saved in zip(args, self.buffer.main_buffer[0]):
            if isinstance(arg, np.ndarray):
                assert (arg == saved).all()
            else:
                assert arg == saved

    def test_get_sample(self):
        """
        Test sample shapes.
        """
        shapes = [np.squeeze(item).shape for item in self.observations[0]]
        self.buffer.main_buffer.extend(self.observations)
        assert_sample_shapes_match(self.buffer, shapes)


@pytest.mark.usefixtures('observations', 'buffer2')
class TestBuffer2:
    """
    Test ReplayBuffer2 methods.
    """

    def test_append(self):
        """
        Ensure observations being stored properly.
        """
        expected_results = [[] for _ in range(5)]
        for observation in self.observations:
            self.buffer.append(*observation)
            for item, expected_items in zip(observation, expected_results):
                if not isinstance(item, np.ndarray):
                    expected_items.append([item])
                else:
                    expected_items.append(item)
        size = len(self.observations)
        for expected, actual in zip(expected_results, self.buffer.slots):
            assert (np.array(expected) == actual[:size]).all()

    def test_get_sample(self):
        """
        Test sample shapes.
        """
        shapes = [np.squeeze(item).shape for item in self.observations[0]]
        for observation in self.observations:
            self.buffer.append(*observation)
            assert_sample_shapes_match(self.buffer, shapes)
