import sys
from pathlib import Path

import numpy as np

sys.path.append(str(Path(__file__).resolve().parents[1]))

from backend.audio_utils import mix_audio_arrays


def test_mix_audio_arrays_downmixes_stereo_system_audio():
    mic_array = np.array([1000, -1000, 1000, -1000], dtype=np.int16)
    system_array = np.array([100, 200, 300, 400, 500, 600, 700, 800], dtype=np.int16)
    mixed = mix_audio_arrays(mic_array, system_array, system_channels=2)

    system_downmixed = np.array([150, 350, 550, 750], dtype=np.float64)
    expected = (mic_array * 0.8 + system_downmixed * 0.6).astype(np.int16)

    assert mixed.shape == mic_array.shape
    np.testing.assert_array_equal(mixed, expected)
