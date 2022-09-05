from collections import Counter
from fractions import Fraction
from functools import reduce
from itertools import chain, count, islice, repeat
from typing import Union, Callable, List, Optional
from torch.nn.functional import pad
import torch
import torchaudio.transforms as T
from primePy import primes
from math import log2
import warnings

warnings.simplefilter("ignore")

# https://stackoverflow.com/a/46623112/9325832
def _combinations_without_repetition(r, iterable=None, values=None, counts=None):
    if iterable:
        values, counts = zip(*Counter(iterable).items())

    f = lambda i, c: chain.from_iterable(map(repeat, i, c))
    n = len(counts)
    indices = list(islice(f(count(), counts), r))
    if len(indices) < r:
        return
    while True:
        yield tuple(values[i] for i in indices)
        for i, j in zip(reversed(range(r)), f(reversed(range(n)), reversed(counts))):
            if indices[i] != j:
                break
        else:
            return
        j = indices[i] + 1
        for i, j in zip(range(i, r), f(count(j), counts[j:])):
            indices[i] = j


def get_fast_stretches(
    sample_rate: int,
    condition: Optional[Callable] = lambda x: x >= 0.5 and x <= 2 and x != 1,
) -> List[Fraction]:
    """
    Search for time-stretch targets that can be computed quickly for a given sample rate.

    Parameters
    ----------
    sample_rate: int
        The sample rate of an audio clip.
    condition: Callable [optional]
        A function to validate fast stretch ratios.
        Default is `lambda x: x >= 0.5 and x <= 2 and x != 1` (between 50% and 200% speed).

    Returns
    -------
    output: List[Fraction]
        A list of fast time-stretch target ratios
    """
    fast_shifts = set()
    factors = primes.factors(sample_rate)
    products = []
    for i in range(1, len(factors) + 1):
        products.extend(
            [
                reduce(lambda x, y: x * y, x)
                for x in _combinations_without_repetition(i, iterable=factors)
            ]
        )
    for i in products:
        for j in products:
            f = Fraction(i, j)
            if condition(f):
                fast_shifts.add(f)
    return list(fast_shifts)


def time_stretch(
    input: torch.Tensor,
    stretch: Union[float, Fraction],
    sample_rate: int,
    n_fft: Optional[int] = 0,
    hop_length: Optional[int] = 0,
) -> torch.Tensor:
    """
    Stretch a batch of waveforms by a given amount without altering the pitch.

    Parameters
    ----------
    input: torch.Tensor [shape=(batch_size, channels, samples)]
        Input audio clips of shape (batch_size, channels, samples)
    stretch: float OR Fraction
        Indicates the stretch ratio. Usually an element in `get_fast_stretches()`.
    sample_rate: int
        The sample rate of the input audio clips.
    n_fft: int [optional]
        Size of FFT. Default is `sample_rate // 64`.
    hop_length: int [optional]
        Size of hop length. Default is `n_fft // 32`.

    Returns
    -------
    output: torch.Tensor [shape=(batch_size, channels, samples)]
        The time-stretched batch of audio clips
    """

    if not n_fft:
        n_fft = sample_rate // 64
    if not hop_length:
        hop_length = n_fft // 32
    batch_size, channels, samples = input.shape
    # resampler = T.Resample(sample_rate, int(sample_rate / stretch)).to(input.device)
    output = input
    output = output.reshape(batch_size * channels, samples)
    output = torch.stft(output, n_fft, hop_length, return_complex=True)[None, ...]
    stretcher = T.TimeStretch(
        fixed_rate=float(1 / stretch), n_freq=output.shape[2], hop_length=hop_length
    ).to(input.device)
    output = stretcher(output)
    output = torch.istft(output[0], n_fft, hop_length)
    # output = resampler(output)
    del stretcher  # , resampler

    output = output.reshape(batch_size, channels, output.shape[1])
    return output
