"""Tests for Gradio interface

This module contains the tests for Gradio interface.

    Typical usage example:

    pytest test/test_interface.py
"""

from typing import Any, List, Union

import gradio
import pytest
from gradio import Audio, Dropdown

from singalong.inference import choices, inputs, outputs

expected: List[Any] = [gradio.Audio, Dropdown]
provided: List[Any] = list(map(type, inputs))

expected_choices: List[str] = ["Fly Me to the Moon"]
provided_choices: List[str] = choices

expected_outputs: List[Any] = [Audio]
provided_outputs: List[Any] = list(map(type, outputs))

expected_streaming_status: bool = False
provided_streaming_status: bool = inputs[0].streaming

exprected_output_streaming_status: bool = False
provided_output_streaming_status: bool = outputs[0].streaming


@pytest.mark.parametrize(
    "provided, expected",
    [
        (provided, expected),
        (provided_choices, expected_choices),
        (provided_outputs, expected_outputs),
        (provided_streaming_status, expected_streaming_status),
    ],  # noqa
)
def test_components(
    provided: Union[List[Any], bool], expected: Union[List[Any], bool]
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
