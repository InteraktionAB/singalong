"""Implement gradio interface

Implement the gradio interface

    Typical example:
        python -m sing --cs [int] --sr [int]
"""

from argparse import ArgumentParser
from functools import partial
from json import loads
from typing import Callable, List, Tuple, Union

from gradio import Interface
from numpy import concatenate, int16
from numpy.typing import NDArray
from pytsmod import phase_vocoder
from vosk import KaldiRecognizer, Model


def get_chunks(data: bytes, chunk_size: int) -> List[bytes]:

    """Split byte array at intervals

    This function cut byte into equal sized
    chunks.

    Args:
        data: The byte array
        range_: A range object of indices

    Returns:
        This function returns list of byte chunks

    Raises:
        None
    """

    range_: range = range(len(data) // chunk_size)
    # fmt: off
    chunks = list(
        map(
            lambda index: (
                data[
                    index
                    * range_.stop:min(
                        len(data), (index * range_.stop) + range_.stop
                    )  # noqa
                ]
            ),
            range_,
        )
    )
    # fmt: on
    return chunks


def get_times(
    result: Union[str, bytes, bytearray]
) -> List[Tuple[float, float]]:  # noqa

    """Get the time from recognizer result

    Get the time from recognizer result.

    Args:
        result: Final result from recognizer.

    Returns:
        Start and end times.

    Raise:
        # TODO
    """

    # Result is converted to json compatible dict
    results: dict = loads(result)["result"]

    # Each start and end times are extracted
    times: List[Tuple[float, float]] = list(
        map(lambda result: (result["start"], result["end"]), results)
    )

    return times


def cut_audio(
    audio: NDArray, timestamp: Tuple[float, float], sample_rate: int
) -> NDArray:

    """Cut the audio

    Cut the audio at timestamp

    Args:
        audio: Audio array
        timestamp: Start and end time
        sample_rate: The audio sample rate

    Returns:
        Sliced array

    Raises:
        # TODO
    """

    # rate = frames / time
    from_n_frames = sample_rate * int(timestamp[0])
    to_n_frames = sample_rate * int(timestamp[1])

    return audio[from_n_frames:to_n_frames]


def get_words(
    audio: Tuple[int, NDArray],
    pcm_constant: int,
    get_chunks: Callable,
    get_times: Callable,
    cut_audio: Callable,
    recognizer: KaldiRecognizer,
) -> List[Tuple[int, NDArray]]:

    """Split audio at spoken word

    This function split audio array into subarrays at spoken interval.
    It uses VOSK for transcription.

    Args:
        audio: The audio in the order sample rate, audio array.
        pcm_constant: The constant value to be multiplied when creating bytes.
        get_chunks: Function the cuts bytes into chunks.
        get_times: Function returns time stamps.
        cut_audio: Split audio at time intervals.
        recognizer: The kaldi recognizer that recognize word.

    Returns:
        A list of audio in the order sample rate, audio array.

    Raises:
        None
    """

    sample_rate, data = audio

    # Equal sized byte chunks
    chunks: List[bytes] = get_chunks(int16(data * pcm_constant).tobytes())

    # Recognizer consume each chunks
    list(map(recognizer.AcceptWaveform, chunks))

    # Get timestamps of each word
    times: List[Tuple[float, float]] = get_times(recognizer.FinalResult())

    # Cut audio at atime stamps
    words: List[NDArray] = list(map(cut_audio, times))

    return list(zip((sample_rate,) * len(words), words))


def get_duration(audio: Tuple[int, NDArray]) -> float:

    """Get duration of audio

    Duration of the array is determined as
        total_number_of_frames / sample_rate

    Args:
        audio: Audio of the order sample rate, array.

    Returns:
        Duration in seconds

    Raises:
        None
    """

    sample_rate, array = audio

    return len(array) * sample_rate


def get_stretched_word(
    word: Tuple[int, NDArray], scale: float, stretcher: Callable
) -> Tuple[int, NDArray]:

    """Stretch word to scale

    Stretch word to scale

    Args:
        word: Audio
        scale: The scale
        stretcher: The stretching function

    Returns:
        Return the stretched word

    Raises:
    """

    sample_rate, array = word

    return sample_rate, stretcher(array, scale)


def merge_words(
    audios: List[Tuple[int, NDArray]], merger: Callable, sample_rate: int
) -> Tuple[int, NDArray]:

    """Merge audio arrays

    This function merge audio arrays in given order

    Args:
        audio: Audio in the order sample_rate, array.
        merger: Merging method.

    Returns:
        This function returns a tuple in the order sample_rate, array.

    Raises:
        None
    """

    arrays: Tuple[NDArray, ...] = tuple(map(lambda audio: audio[1], audios))

    return sample_rate, merger(arrays)


def inference(
    user_in: Tuple[int, NDArray],
    song: Tuple[int, NDArray],
    sample_rate: int,
    get_words: Callable,
    get_duration: Callable,
    get_stretched_word: Callable,
    merge_words: Callable,
) -> Tuple[int, NDArray]:

    """Run the inference

    Args:
        user_in: Audio in the order sample rate, audio array.
        song: Audio in the order sample rate, audio array.
        sample_rate: Sample rate for the returned word.
        get_words: Function returning words.
        get_duration: Function returning duration of the audio.
        get_stretched_word: Function returning stretched word.
        merge_words: Function returning merged words.

    Returns:
        Audio in the order sample rate, audio array.

    Raises:
        None
    """

    # Fing audio segments containing word in each audio inputs
    words_from_user_input: List[Tuple[int, NDArray]] = get_words(user_in)
    words_from_song: List[Tuple[int, NDArray]] = get_words(song)

    # Fing duration of each word in user input
    duration_of_words_from_user: List[float] = list(
        map(get_duration, words_from_user_input)
    )

    # Stretch the word in song according to duration
    stretched_words: List[Tuple[int, NDArray]] = list(
        map(
            get_stretched_word,
            zip(words_from_song, duration_of_words_from_user),  # noqa
        )  # noqa
    )

    # Merge each word
    merged_song: Tuple[int, NDArray] = merge_words(stretched_words)

    return merged_song


if __name__ == "__main__":

    parser: ArgumentParser = ArgumentParser(
        description=__doc__,
    )

    parser.add_argument(
        "--sr",
        help="Sample rate",
        required=False,
        default=44100,
        type=int,
    )
    parser.add_argument(
        "--cs",
        help="Chunk size",
        required=False,
        default=4000,
        type=int,
    )

    args = parser.parse_args()

    # Define get_chunks
    get_chunks_: Callable = partial(get_chunks, chunk_size=args.cs)

    # Define get_times
    get_times_: Callable = get_times

    # Define cut_audio
    cut_audio_: Callable = partial(cut_audio, sample_rate=args.sr)

    # Define recognizer
    model: Model = Model(lang="en-us")
    recognizer_: KaldiRecognizer = KaldiRecognizer(model, args.sr)
    recognizer_.SetWords(True)
    recognizer_.SetPartialWords(True)

    # Define get_words
    get_words_: Callable = partial(
        get_words,
        pcm_constant=32768,
        get_chunks=get_chunks_,
        get_times=get_times_,
        cut_audio=cut_audio_,
        recognizer=recognizer_,
    )

    # Define get_duration
    get_duration_: Callable = get_duration

    # Define get_stretched_word
    get_stretched_word_: Callable = partial(
        get_stretched_word, stretcher=phase_vocoder
    )  # noqa

    # Define merge_words
    merge_words_: Callable = partial(
        merge_words, merger=concatenate, sample_rate=args.sr
    )

    # Define inference function
    inference_: Callable = partial(
        inference,
        sample_rate=args.sr,
        get_words=get_words_,
        get_duration=get_duration_,
        get_stretched_word=get_stretched_word_,
        merge_words=merge_words_,
    )
    inference_.__name__ = "inference"

    # Launch gradio interface
    Interface(
        inference_, inputs=["audio", "audio"], outputs=["audio"], share=True
    ).launch()
