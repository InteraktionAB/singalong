"""Tests for Gradio interface

This module contains the tests for Gradio interface.

    Typical usage example:

    pytest test/test_interface.py
"""

from typing import Any

import pytest
from sing import expected, inputs


@pytest.mark.parametrize("components, expected", [(inputs, expected)])
def test_components(components: list[Any], expected: list[Any]) -> None:

    """Test the input component list

    Make sure the components are as expected.

    Args:
        components: A list containing Gradio components.
        expected: A list containing expected components.

    Returns:
        This functions returns None.

    Raise:
        AssertionError: When components are not expected.
    """

    assert components == expected
