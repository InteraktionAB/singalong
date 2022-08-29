from typing import Any, List

from gradio import Audio, Dropdown

choices: List[str] = ["Fly Me to the Moon"]
inputs: List[Any] = [Audio(source="microphone"), Dropdown(choices)]
