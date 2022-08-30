"""Test the inference function

This module asserts the inference function.

    Typical example:
        pytest test/test_inference_function.py
"""

from numpy import ndarray
from numpy.random import uniform

from sing import inference


def test_inference_function():
    """Assert arguments and return value

    This function assert the arguments and return
    value from the inference function.

    Args:

    Returns:

    Raises:
    """
    sample_rate: int = 1
    samples: int = 100
    channels: int = 1
    audio: ndarray = uniform(low=-1.0, high=1.0, size=(samples, channels))
    song: str = "Fly Me to the Moon"
    rate, out = inference((sample_rate, audio), song)
    assert (sample_rate, audio) == (rate, out)
