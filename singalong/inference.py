"""Define dependencies for inference

This module define the dependables for the
gradio interface.

    Typical example:
        from singalong.inference import inference, inputs, outputs
        from gradio import Interface

        interface = Interface(fn=inference, inputs=inputs, outputs=outputs)
"""

from typing import Any, List, Tuple

from gradio import Audio, Dropdown
from numpy import ndarray


def inference(sample: Tuple[int, ndarray], song: str) -> Tuple[int, ndarray]:
    """The inference function

    This function runs when data audio is connected.

    Args:
        sample: The audio sample of the format (sample_rate, audio_array).
        song: Name of the song. Eg: Fly Me to the Moon.

    Returns:
        sample: The input audio as it is.

    Raises:
    """
    return sample


choices: List[str] = ["Fly Me to the Moon"]
inputs: List[Any] = [
    Audio(source="microphone", streaming=False),
    Dropdown(choices),
]  # noqa
outputs: List[Any] = [Audio(source="microphone", streaming=False)]
