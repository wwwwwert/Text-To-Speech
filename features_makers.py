import numpy as np
import pyworld as pw
from scipy.interpolate import interp1d


def get_pitch(mel_spec: np.array, audio: np.array, sr: int=22050):
    """Makes pitch feature for audio by MEL spectrogram.

    Args:
        mel_spec: array of MEL spectrogram
        audio: audio wave

    Returns:
        array of pitch
    """
    frame_period = (audio.shape[0] / sr * 1000) / mel_spec.shape[0]  # ms per frame

    # took from py world https://pypi.org/project/pyworld/
    _f0, t = pw.dio(audio, sr, frame_period=frame_period)
    f0 = pw.stonemask(audio, _f0, t, sr)[:mel_spec.shape[0]]

    # smooth pitch values
    nonzeros = np.nonzero(f0)
    x = np.arange(f0.shape[0])[nonzeros]
    y = f0[nonzeros]
    values = (y[0], y[-1])
    f = interp1d(x, y, bounds_error=False, fill_value=values)

    f0 = f(np.arange(f0.shape[0]))

    return f0


def get_energy(mel_spec: np.array):
    """Makes energy feature for audio by MEL spectrogram.

    Args:
        mel_spec: array of MEL spectrogram

    Returns:
        energy array
    """
    energy = np.linalg.norm(mel_spec, axis=-1)
    return energy