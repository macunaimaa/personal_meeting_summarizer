import numpy as np


def mix_audio_arrays(mic_array: np.ndarray, system_array: np.ndarray, system_channels: int) -> np.ndarray:
    """Mix microphone and system audio into a single mono array.

    If the system audio contains multiple channels, it is downmixed to mono
    before mixing. The output length matches the shorter of the two inputs.
    """
    if system_channels > 1:
        usable = len(system_array) // system_channels * system_channels
        system_array = system_array[:usable].reshape(-1, system_channels).mean(axis=1)

    min_len = min(len(mic_array), len(system_array))
    mic_array = mic_array[:min_len]
    system_array = system_array[:min_len]
    return (mic_array * 0.8 + system_array * 0.6).astype(np.int16)
