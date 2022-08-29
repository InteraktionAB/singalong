from typing import Any, List

from gradio import Audio

inputs: List[Any] = [Audio(source="microphone")]
choices: List[str] = ["Fly Me to the Moon"]
