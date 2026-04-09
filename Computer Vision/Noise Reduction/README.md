# Noise Reduction Script

> A Python implementation of spectral subtraction for reducing background noise in WAV audio files.

## Overview

This script implements the spectral subtraction method for audio noise reduction, similar to noise reduction features in tools like Audacity. It reads a WAV file, applies Short-Time Fourier Transform (STFT) with Hamming windowing, estimates and subtracts the noise spectrum, then reconstructs the cleaned audio via inverse FFT. The script produces both a filtered audio file and a comparison graph.

## Features

- Spectral subtraction noise reduction on WAV audio files
- Hamming window-based frame segmentation (frame length: 400 samples)
- FFT-based spectral analysis with magnitude and phase separation
- Noise estimation via mean spectral magnitude across all frames
- Overlap-add reconstruction for the cleaned signal
- Generates a comparison plot (original vs. filtered waveform)
- Saves the filtered audio as a new WAV file

## Project Structure

```
Noise Reduction Script/
├── Noise Reduction Script.py    # Main noise reduction script
├── requirements.txt             # Python dependencies
├── Audio/                       # Sample audio files
│   ├── test_noise.wav           # Sample noisy input
│   └── (Filtered_Audio)test_noise .wav  # Sample filtered output
└── Graph/                       # Output graphs
    └── test_noise.wav(Spectral Subtraction graph).jpg
```

## Requirements

- Python 3.x
- matplotlib == 3.2.2
- numpy == 1.18.5
- scipy == 1.5.0

## Installation

```bash
cd "Noise Reduction Script"
pip install -r requirements.txt
```

## Usage

```bash
python "Noise Reduction Script.py"
```

When prompted, enter the path to a WAV audio file:

```
Enter the file path: Audio/test_noise.wav
```

The script outputs:
- `(Filtered_Audio)<filename>` — the noise-reduced WAV file
- `<filename>(Spectral Subtraction graph).jpg` — a comparison plot

## How It Works

1. **Read audio**: Uses `scipy.io.wavfile.read()` to load the WAV file and its sample rate
2. **Framing**: Splits the audio into overlapping frames of 400 samples with 50% overlap (hop size of 200)
3. **Windowing**: Applies a Hamming window (`np.hamming(400)`) to each frame
4. **FFT**: Computes the FFT of each windowed frame, separating magnitude and phase spectra
5. **Noise estimation**: Calculates the mean magnitude spectrum across all frames as the noise estimate
6. **Spectral subtraction**: Subtracts 2× the noise estimate magnitude from each frame's magnitude spectrum; negative values are floored to zero
7. **Reconstruction**: Combines the cleaned magnitude with the original phase, applies inverse FFT, and uses overlap-add to reconstruct the time-domain signal
8. **Output**: Writes the cleaned audio as a 16-bit WAV file and saves a matplotlib comparison plot

### Algorithm

$$\hat{S}(f) = \max(|X(f)| - 2 \cdot \overline{|N(f)|}, \; 0) \cdot e^{j\angle X(f)}$$

Where $X(f)$ is the noisy signal spectrum, $\overline{|N(f)|}$ is the mean noise magnitude, and $\hat{S}(f)$ is the estimated clean signal.

## Configuration

| Parameter | Value | Location | Description |
|-----------|-------|----------|-------------|
| `fl` | 400 | Line 7 | Frame length in samples |
| Noise multiplier | 2 | Line 22 | Scale factor for noise estimate subtraction |
| Plot size | (8, 5) | Line 30 | Matplotlib figure size |
| Plot x-axis | 64000 | Line 33 | Hardcoded x-axis range for plotting |

## Limitations

- **Bug**: Uses `os.path.basename()` but `os` is never imported — the script will raise a `NameError` at the output stage
- The plot x-axis is hardcoded to 64000 samples — will produce incorrect axis labels for audio files of different lengths
- Only supports WAV format input
- The noise estimate assumes stationary noise (mean across all frames) — not effective for non-stationary noise
- The noise multiplier (2×) is hardcoded and not tunable via parameters
- Frame length (400) is fixed — no option to adjust for different sample rates
- No command-line arguments — file path is entered interactively
- Output files are written to the current working directory, not alongside the input file

## License

Not specified.
