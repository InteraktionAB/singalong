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
from numpy import ndarray
from numpy.testing import assert_allclose
from soundfile import read

from singalong.inference import (choices, get_duration, inference, inputs,
                                 outputs)

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
returned_duration: float = get_duration(path="test/duration.wav")

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


@pytest.mark.parametrize(
    "expected, returned", [(expected_scaled_version, returned_scaled_version)]
)
def test_array(expected: ndarray, returned: ndarray):
    assert_allclose(expected, returned, rtol=1e-5, atol=1)


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
