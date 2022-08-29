"""Tests for Gradio interface

This module contains the tests for Gradio interface.

    Typical usage example:

    pytest test/test_interface.py
"""

from typing import Any, List

import gradio
import pytest
from gradio import Audio, Dropdown

from sing import choices, inputs, outputs

expected: List[Any] = [gradio.Audio, Dropdown]
provided: List[Any] = list(map(type, inputs))

expected_choices: List[str] = ["Fly Me to the Moon"]
provided_choices: List[str] = choices

expected_outputs: List[Any] = [Audio]
provided_outputs: List[Any] = list(map(type, outputs))


@pytest.mark.parametrize(
    "provided, expected",
    [
        (provided, expected),
        (provided_choices, expected_choices),
        (provided_outputs, expected_outputs),
    ],  # noqa
)
def test_components(provided: List[Any], expected: List[Any]) -> None:

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
