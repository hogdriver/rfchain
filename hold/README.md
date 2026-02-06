# RF Chain Framework for Anti-Jamming Analysis

A modular Python framework for simulating RF communication systems under various jamming scenarios.

## Features

- **Multiple Modulation Schemes**: BPSK, QPSK, 8PSK, 16PSK, 4/16/64/256-QAM, OOK, FSK
- **Pulse Shaping**: Root Raised Cosine (RRC), Raised Cosine (RC), Gaussian, Rectangular
- **Forward Error Correction**: Reed-Solomon, Convolutional, LDPC, Turbo (simplified)
- **Channel Models**: AWGN, Rayleigh, Rician, Nakagami, Flat fading, Frequency-selective
- **Jamming Types**: Barrage, Spot, Swept, Pulsed, Follower, Partial-band, Tone
- **Antenna Processing**: Omnidirectional, Directional, Beamforming
- **Receiver Processing**: PLL/Costas sync, LMS/RLS/CMA equalization
- **Validation**: BER, PER, SINR, EVM metrics with constellation and spectrum plots

## Installation

```bash
pip install numpy scipy matplotlib tomli tomli-w
```

Note: `tomli` and `tomli-w` are only needed if using TOML config files. The framework works with defaults if these are not available.

## Quick Start

```python use Python version 3.12.9 or greater
from main import RFChainFramework

# Run with defaults
rf_chain = RFChainFramework()
results = rf_chain.run_simulation(verbose=True)

# Run with custom config
rf_chain = RFChainFramework("configs/default_config.toml")
results = rf_chain.run_simulation()

# Parameter sweep
snr_values = [0, 5, 10, 15, 20, 25]
sweep_results = rf_chain.run_parameter_sweep(
    parameter='channel.snr_db',
    values=snr_values,
    metric='ber'
)
```

## Project Structure

```
rfchain_framework/
├── main.py                 # Main orchestration
├── requirements.txt        # Dependencies
├── configs/
│   └── default_config.toml # Default configuration
└── modules/
    ├── __init__.py
    ├── config.py           # Configuration management
    ├── source.py           # Signal generation
    ├── pulse_shaping.py    # Pulse shaping filters
    ├── fec.py              # Forward error correction
    ├── channel.py          # Channel models
    ├── jamming.py          # Jamming models
    ├── antenna.py          # Antenna processing
    ├── receiver.py         # Receiver processing
    └── validation.py       # Performance metrics
```

## Configuration Options

### Jamming Types
- `none` - No jamming
- `barrage` - Wideband noise jamming
- `spot` - Narrowband jamming at carrier
- `swept` - Frequency-swept jamming
- `pulsed` - Intermittent jamming
- `follower` - Repeater/follower jammer
- `partial_band` - Jams portion of bandwidth
- `tone` - Multi-tone jamming

### Channel Models
- `awgn` - Additive White Gaussian Noise only
- `rayleigh` - Rayleigh fading (no line-of-sight)
- `rician` - Rician fading (with line-of-sight)
- `nakagami` - Nakagami-m fading
- `flat` - Flat fading
- `frequency_selective` - Multipath fading

## Known Issues

1. FEC encoding/decoding with non-binary modulations may have symbol mapping issues
2. For best results, use `code_type = "none"` or `"convolutional"` with BPSK

## Fixes Applied

- Fixed `scipy.signal` import conflict in `jamming.py`
- Made TOML libraries optional in `config.py`
- Fixed constellation plot index bounds in `validation.py`
