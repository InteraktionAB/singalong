"""Tests for Gradio interface

This module contains the tests for Gradio interface.

    Typical usage example:

    pytest test/test_interface.py
"""

from typing import Any, List

import gradio
import pytest

from sing import inputs

expected: List[Any] = [gradio.Audio]
provided: List[Any] = list(map(type, inputs))


@pytest.mark.parametrize("provided, expected", [(provided, expected)])
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
