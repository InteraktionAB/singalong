from typing import Any, List, Tuple

from gradio import Audio, Dropdown
from numpy import ndarray


def inference(sample: Tuple[int, ndarray], song: str) -> Tuple[int, ndarray]:
    return sample


choices: List[str] = ["Fly Me to the Moon"]
inputs: List[Any] = [
    Audio(source="microphone", streaming=True),
    Dropdown(choices),
]  # noqa
outputs: List[Any] = [Audio()]
