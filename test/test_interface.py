"""Tests for Gradio interface

This module contains the tests for Gradio interface.

    Typical usage example:

    pytest test/test_interface.py
"""

from typing import Any, Callable, List, Union

import gradio
import pytest
from gradio import Audio, Dropdown

from sing import choices, inference, inputs, interface, outputs

expected: List[Any] = [gradio.Audio, Dropdown]
provided: List[Any] = list(map(type, inputs))

expected_choices: List[str] = ["Fly Me to the Moon"]
provided_choices: List[str] = choices

expected_outputs: List[Any] = [Audio]
provided_outputs: List[Any] = list(map(type, outputs))

expected_inputs_in_interface: List[Any] = inputs
provided_inputs_in_interface: List[Any] = interface.input_components

expected_outputs_in_interface: List[Any] = outputs
provided_outputs_in_interface: List[Any] = interface.output_components

expected_function: Callable = inference
provided_function: Callable = interface.fn

expected_streaming_status: bool = True
provided_streaming_status: bool = interface.input_components[0].streaming


@pytest.mark.parametrize(
    "provided, expected",
    [
        (provided, expected),
        (provided_choices, expected_choices),
        (provided_outputs, expected_outputs),
        (provided_inputs_in_interface, expected_inputs_in_interface),
        (provided_outputs_in_interface, expected_outputs_in_interface),
        (provided_function, expected_function),
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
