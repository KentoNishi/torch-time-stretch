# Usage

## Installation
You can install `torch-time-stretch` from [PyPI](https://pypi.org/project/torch-time-stretch/).

```bash
pip install torch-time-stretch
```

To upgrade an existing installation of `torch-time-stretch`, use the following command:

```bash
pip install --upgrade --no-cache-dir torch-time-stretch
```

## Importing

First, import `torch-time-stretch`.

```python
# import all functions
from torch_time_stretch import *

# ... or import them manually
from torch_time_stretch import get_fast_stretches, time_stretch
```

## What's included
`torch-time-stretch` includes the following:

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Name</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Function</td>
      <td><code>time_stretch</code></td>
      <td>Stretch a batch of waveforms by a given amount without altering the pitch.</td>
    </tr>
    <tr>
      <td>Function</td>
      <td><code>get_fast_stretches</code></td>
      <td>Utility function for calculating time-stretches that can be executed quickly.</td>
    </tr>
  </tbody>
</table>

## Methods

### `time_stretch`
Stretch a batch of waveforms by a given amount without altering the pitch.

#### Arguments

<table>
  <thead>
    <tr>
      <th>Argument</th>
      <th>Required</th>
      <th>Default Value</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>input</code></td>
      <td>Yes</td>
      <td></td>
<td><pre><code>torch.Tensor [
    shape=(
        batch_size,
        channels,
        samples
    )
]</code></pre></td>
      <td>Input audio clips of shape (batch_size, channels, samples)</td>
    </tr>
    <tr>
      <td><code>stretch</code></td>
      <td>Yes</td>
      <td></td>
      <td><code>float</code> or <code>Fraction</code></td>
      <td>Indicates the stretch ratio (usually an element in <code>get_fast_stretches()</code>).</td>
    </tr>
    <tr>
      <td><code>sample_rate</code></td>
      <td>Yes</td>
      <td></td>
      <td><code>int</code></td>
      <td>The sample rate of the input audio clips.</td>
    </tr>
    <tr>
      <td><code>n_fft</code></td>
      <td>No</td>
      <td><code>sample_rate // 64</code></td>
      <td><code>int</code></td>
      <td>Size of FFT. Default <code>sample_rate // 64</code>. Smaller is faster.</td>
    </tr>
    <tr>
      <td><code>bins_per_octave</code></td>
      <td>No</td>
      <td><code>12</code></td>
      <td><code>int</code></td>
      <td>Number of bins per octave. Default is 12.</td>
    </tr>
  </tbody>
</table>

#### Return value

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
<td><pre><code>torch.Tensor [
    shape=(
        batch_size,
        channels,
        samples
    )
]</code></pre></td>
      <td>The time-stretched batch of audio clips</td>
    </tr>
  </tbody>
</table>

### `get_fast_stretches`
Search for time-stretch targets that can be computed quickly for a given sample rate.

<table>
  <thead>
    <tr>
      <th>Argument</th>
      <th>Required</th>
      <th>Default Value</th>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>sample_rate</code></td>
      <td>Yes</td>
      <td></td>
      <td><code>int</code></td>
      <td>The sample rate of an audio clip.</td>
    </tr>
    <tr>
      <td><code>condition</code></td>
      <td>No</td>
      <td>
<pre><code>lambda x: (
    x &gt;= 0.5 and x &lt;= 2 and x != 1
)</code></pre>
      </td>
      <td><code>Callable</code></td>
      <td>A function to validate fast stretch ratios. Default value limits computed targets to values between <code>-1</code> and <code>+1</code> octaves.</td>
    </tr>
  </tbody>
</table>

#### Return value

<table>
  <thead>
    <tr>
      <th>Type</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td><code>List[Fraction]</code></td>
      <td>A list of fast time-stretch target ratios that satisfy the given conditions.</td>
    </tr>
  </tbody>
</table>

## Example

See [example.py](https://github.com/KentoNishi/torch-time-stretch/blob/master/example.py) to see an example of `torch-time-stretch` in action!
