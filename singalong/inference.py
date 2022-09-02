"""Define dependencies for inference

This module define the dependables for the
gradio interface.

    Typical example:
        from singalong.inference import inference, inputs, outputs
        from gradio import Interface

        interface = Interface(fn=inference, inputs=inputs, outputs=outputs)
"""

from json import loads
from typing import Any, List, Tuple
from wave import open as open_wave

from gradio import Audio
from numpy import ndarray
from pytsmod import phase_vocoder
from soundfile import SoundFile
from vosk import KaldiRecognizer, Model


def get_time_stamps(path: str):
    wave = open_wave(path)
    model = Model(lang="en-us")
    recognizer = KaldiRecognizer(model, wave.getframerate())
    recognizer.SetWords(True)
    while True:
        frame = wave.readframes(4000)
        if len(frame) == 0:
            break
        recognizer.AcceptWaveform(frame)
    metas: List[dict] = loads(recognizer.FinalResult())["result"]
    time_stamps = []
    for meta in metas:
        start: float = meta["start"]
        end: float = meta["end"]
        time_stamps.append((start, end))
    wave.close()
    return tuple(time_stamps)


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

    duration: float = get_duration(song)
    return sample[0], phase_vocoder(sample[1], duration).astype("int16")


def get_duration(path: str) -> float:

    """Return the duration of audio file

    This function return the duration of the provided
    audio file.

    Args:
        path: Path to the audio file.

    Returns:
        Return duration in seconds.

    Raises:
    """

    file: SoundFile = SoundFile(path)
    return file.frames / file.samplerate


choices: List[str] = ["Fly Me to the Moon"]
inputs: List[Any] = [
    Audio(source="upload", streaming=False),
    Audio(type="filepath"),
]  # noqa
outputs: List[Any] = [Audio(source="microphone", streaming=False)]
