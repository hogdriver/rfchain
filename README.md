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
pip install numpy scipy matplotlib

# Optional (for TOML config files):
pip install tomli tomli-w
```

Note: `tomli` and `tomli-w` are only needed if using TOML config files. The framework works with defaults if these are not available.

## Quick Start

```python
from main import RFChainFramework

# Run with defaults (outputs saved to current directory)
rf_chain = RFChainFramework()
results = rf_chain.run_simulation(verbose=True)

# Specify output directory for plots
rf_chain = RFChainFramework(output_dir="./output")
results = rf_chain.run_simulation()

# Run with custom config file
rf_chain = RFChainFramework(
    config_path="configs/default_config.toml",
    output_dir="./results"
)
results = rf_chain.run_simulation()

# Parameter sweep
import numpy as np
snr_values = np.arange(0, 25, 5)
sweep_results = rf_chain.run_parameter_sweep(
    parameter='channel.snr_db',
    values=snr_values,
    metric='ber'
)
```

## Running from Command Line

```bash
# From the rfchain_framework directory:
python main.py

# Or run examples:
python examples/anti_jamming_comparison_fixed.py
```

## Project Structure

```
rfchain_framework/
├── main.py                 # Main orchestration
├── requirements.txt        # Dependencies
├── configs/
│   └── default_config.toml # Default configuration
├── examples/
│   ├── anti_jamming_comparison.py        # Original comparison script
│   └── anti_jamming_comparison_fixed.py  # Fixed version using BPSK
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

## Output Files

When running simulations, the following files are saved to the output directory:
- `constellation.png` - Constellation diagram showing received symbols
- `spectrum.png` - Power spectral density of transmitted and received signals

## Known Limitations

1. FEC encoding/decoding with non-binary modulations (QPSK, QAM) may have symbol mapping issues. For best FEC results, use BPSK.
2. The Viterbi decoder can fail (produce worse BER) when input error rate exceeds ~15%.

## Troubleshooting

**"No config file found, using defaults"**
This is normal - the framework uses sensible defaults when no config file is provided.

**High BER (~50%)**
This usually means the jammer is overwhelming the signal. Try:
- Setting `jamming.jammer_type = 'none'`
- Reducing `jamming.jnr_db`
- Increasing `channel.snr_db`
