from typing import Any, List, Tuple

from gradio import Audio, Dropdown, Interface
from numpy import ndarray


def inference(sample: Tuple[int, ndarray], song: str) -> Tuple[int, ndarray]:
    return sample


choices: List[str] = ["Fly Me to the Moon"]
inputs: List[Any] = [Audio(source="microphone"), Dropdown(choices)]
outputs: List[Any] = [Audio()]

interface: Interface = Interface(fn=inference, inputs=inputs, outputs=outputs)
interface.launch(share=True)
