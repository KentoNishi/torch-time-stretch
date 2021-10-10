import torch
import numpy as np
from scipy.io import wavfile
from torch_time_stretch import *

# read an audio file
SAMPLE_RATE, sample = wavfile.read("./wavs/test.wav")

# convert to tensor of shape (batch_size, channels, samples)
dtype = sample.dtype
sample = torch.tensor(
    [np.swapaxes(sample, 0, 1)],  # (samples, channels) --> (channels, samples)
    dtype=torch.float32,
    device="cuda" if torch.cuda.is_available() else "cpu",
)


def test_time_stretch_2_up():
    # speed up by 2 times
    up = time_stretch(sample, Fraction(2, 1), SAMPLE_RATE)
    assert up.shape == sample.shape
    wavfile.write(
        "./wavs/shifted_octave_+1.wav",
        SAMPLE_RATE,
        np.swapaxes(up.cpu()[0].numpy(), 0, 1).astype(dtype),
    )


def test_time_stretch_2_down():
    # slow down by 2 times
    down = time_stretch(sample, Fraction(1, 2), SAMPLE_RATE)
    assert down.shape == sample.shape
    wavfile.write(
        "./wavs/shifted_octave_-1.wav",
        SAMPLE_RATE,
        np.swapaxes(down.cpu()[0].numpy(), 0, 1).astype(dtype),
    )


def test_time_stretch_to_fast_ratios():
    # get stretch ratios that are fast (between +1 and -1 octaves)
    for ratio in get_fast_stretches(SAMPLE_RATE):
        print("Stretching", ratio)
        shifted = time_stretch(sample, ratio, SAMPLE_RATE)
        assert shifted.shape == sample.shape
        wavfile.write(
            f"./wavs/stretched_ratio_{ratio.numerator}-{ratio.denominator}.wav",
            SAMPLE_RATE,
            np.swapaxes(shifted.cpu()[0].numpy(), 0, 1).astype(dtype),
        )
