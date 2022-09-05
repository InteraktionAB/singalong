"""Tests for Gradio interface

This module contains the tests for Gradio interface.

    Typical usage example:

    pytest test/test_interface.py
"""

# fmt: off
from typing import Any, List, Tuple, Union

import gradio
import pytest
from gradio import Audio
from numpy import asarray, ndarray
from numpy.testing import assert_allclose
from numpy.typing import NDArray
from soundfile import read

from singalong.inference import (choices, get_duration, get_stretched_audio,
                                 get_time_stamps, inference, inputs, outputs)

# fmt: on

expected: List[Any] = [gradio.Audio, Audio]
provided: List[Any] = list(map(type, inputs))

expected_choices: List[str] = ["Fly Me to the Moon"]
provided_choices: List[str] = choices

expected_outputs: List[Any] = [Audio]
provided_outputs: List[Any] = list(map(type, outputs))

expected_streaming_status: bool = False
provided_streaming_status: bool = inputs[0].streaming

exprected_output_streaming_status: bool = False
provided_output_streaming_status: bool = outputs[0].streaming

expected_input_audio_source: str = "upload"
provided_input_audio_source: str = inputs[0].source

expected_duration: float = 1.53
returned_duration: float = get_duration("test/duration.wav")

expected_scaled_version, _ = read("test/scale.flac")
audio: Tuple[int, ndarray] = read("test/scale_input.flac")
_, returned_scaled_version = inference(
    (audio[1], audio[0]), "test/duration.wav"
)  # noqa

# inputs[0] read file path
expected_return_type: str = "filepath"
returned_return_type: str = inputs[1].type

# Output expect numpy array
expected_output_type: str = "numpy"
returned_output_type: str = outputs[0].type

expected_time_stamps = (
    (0.06, 0.18),
    (0.18, 0.24),
    (0.24, 1.0),
)  # Order (start, end)
returned_time_stamps: Tuple[Tuple[float]] = get_time_stamps(
    "test/term.wav"
)  # Return timestamp

# Assert duration
expected_duration_: float = 0.12
returned_duration_: float = get_duration((0.06, 0.18))

# Assert stretched result
expected_stretched_audio_sr: Tuple[NDArray, int] = read("test/scale.flac")
returned_stretched_audio_sr: Tuple[NDArray, int] = get_stretched_audio(
    "test/scale_input.flac", get_duration("test/duration.wav")
)


@pytest.mark.parametrize(
    "expected, returned, rtol, atol",
    [
        (expected_scaled_version, returned_scaled_version, 1e-5, 1),
        (
            asarray(expected_time_stamps),
            asarray(returned_time_stamps),
            1e-0,
            0.2,
        ),  # noqa
        (
            expected_stretched_audio_sr[0],
            returned_stretched_audio_sr,
            1e-5,
            1,
        ),  # noqa
    ],
)
def test_array(expected: ndarray, returned: ndarray, rtol: float, atol: float):
    assert_allclose(expected, returned, rtol=rtol, atol=atol)


@pytest.mark.parametrize(
    "provided, expected",
    [
        (provided, expected),
        (provided_choices, expected_choices),
        (provided_outputs, expected_outputs),
        (provided_streaming_status, expected_streaming_status),
        (provided_input_audio_source, expected_input_audio_source),
        (returned_duration, expected_duration),
        (returned_return_type, expected_return_type),
        (returned_output_type, expected_output_type),
        (returned_scaled_version.dtype, "int16"),
        (returned_duration_, expected_duration_),
    ],  # noqa
)
def test_components(
    provided: Union[List[Any], bool, str], expected: Union[List[Any], bool, str]  # noqa
) -> None:

    """Test the input component list

    Make sure the components are as expected.

    Args:
        provided: A list containing type of Gradio components.
        expected: A list containing expected type of components.

    Returns:
        This functions returns None.

    Raise:
        AssertionError: When components are not expected.
    """

    assert provided == expected
