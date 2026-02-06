#!/usr/bin/env python3
"""
Example: Anti-Jamming Technique Comparison (Fixed Version)

This script compares the performance of different anti-jamming techniques:
1. No protection (baseline)
2. Strong FEC (convolutional with BPSK)  
3. Higher SNR margin (simulating directional antenna gain)
4. Combined approach

Key fixes from original:
- Uses BPSK to avoid FEC symbol mapping issues
- Tests across a range of SINR conditions
- Properly accounts for antenna gain effects
"""

import numpy as np
import sys
from pathlib import Path
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from modules.config import ConfigurationManager
from modules.source import SourceModule
from modules.pulse_shaping import PulseShapingModule
from modules.fec import FECModule
from modules.channel import ChannelModule
from modules.jamming import JammingModule
from modules.antenna import AntennaModule
from modules.receiver import ReceiverModule
from modules.validation import ValidationModule


def run_single_simulation(config_overrides: dict, verbose: bool = False) -> dict:
    """Run a single simulation with specified configuration overrides"""
    
    # Create default config
    config = ConfigurationManager()
    
    # Apply overrides
    for module_name, params in config_overrides.items():
        module = getattr(config, module_name)
        for param_name, value in params.items():
            setattr(module, param_name, value)
    
    # Initialize modules
    source = SourceModule(config.source)
    pulse_shaping = PulseShapingModule(config.pulse_shaping)
    fec = FECModule(config.fec)
    channel = ChannelModule(config.channel, config.source.sample_rate)
    jamming = JammingModule(config.jamming, config.source.sample_rate, config.source.carrier_freq)
    wavelength = 3e8 / config.source.carrier_freq
    antenna = AntennaModule(config.antenna, wavelength)
    receiver = ReceiverModule(config.receiver, config.source.signal_type)
    validation = ValidationModule(config.validation)
    
    # Run simulation chain
    source_out = source.generate()
    tx_symbols = source_out.symbols
    baseband = source_out.baseband_signal
    
    # FEC encode
    if config.fec.code_type != 'none':
        fec_out = fec.encode(tx_symbols)
        encoded_symbols = fec_out.encoded_data
        # Re-modulate (for BPSK, this is simple)
        if config.source.signal_type == 'bpsk':
            encoded_signal = 2 * encoded_symbols.astype(float) - 1
            # Apply power scaling
            power_watts = 10 ** ((config.source.power_dbm - 30) / 10)
            encoded_signal = encoded_signal * np.sqrt(power_watts)
        else:
            encoded_signal = baseband  # Fallback
    else:
        encoded_symbols = tx_symbols
        encoded_signal = baseband
    
    # Pulse shaping
    pulse_out = pulse_shaping.apply(encoded_signal)
    shaped = pulse_out.shaped_signal
    
    # Antenna TX (apply gain)
    ant_tx = antenna.transmit(shaped)
    tx_signal = ant_tx.signal
    
    # Channel
    chan_out = channel.apply(tx_signal)
    chan_signal = chan_out.signal
    
    # Jamming
    jam_out = jamming.apply(chan_signal)
    jammed = jam_out.jammed_signal
    jammer_sig = jam_out.jammer_signal
    
    # Antenna RX
    ant_rx = antenna.receive(jammed)
    rx_signal = ant_rx.signal
    
    # Matched filter and downsample
    matched = pulse_shaping.matched_filter(rx_signal)
    downsampled = pulse_shaping.downsample(matched)
    
    # Receiver
    recv_out = receiver.process(downsampled)
    rx_symbols = recv_out.recovered_symbols
    
    # FEC decode
    if config.fec.code_type != 'none':
        decoded = fec.decode(rx_symbols)
        decoded = decoded[:len(tx_symbols)]
    else:
        decoded = rx_symbols[:len(tx_symbols)]
    
    # Calculate BER
    min_len = min(len(tx_symbols), len(decoded))
    tx_bits = tx_symbols[:min_len].astype(int)
    rx_bits = decoded[:min_len].astype(int)
    errors = np.sum(tx_bits != rx_bits)
    ber = errors / min_len
    
    # Calculate SINR
    sig_power = np.mean(np.abs(tx_signal)**2)
    jam_power = np.mean(np.abs(jammer_sig)**2)
    sinr = sig_power / jam_power if jam_power > 0 else np.inf
    sinr_db = 10 * np.log10(sinr) if sinr < np.inf else np.inf
    
    return {
        'ber': ber,
        'sinr_db': sinr_db,
        'errors': errors,
        'total_bits': min_len
    }


def main():
    """Run comprehensive anti-jamming comparison"""
    
    print("=" * 80)
    print("ANTI-JAMMING TECHNIQUE COMPARISON")
    print("Using BPSK modulation for reliable FEC performance")
    print("=" * 80)
    
    # Base configuration
    base_config = {
        'source': {
            'signal_type': 'bpsk',
            'sample_rate': 1e6,
            'symbol_rate': 1e5,
            'num_symbols': 5000,
            'power_dbm': 0.0
        },
        'pulse_shaping': {
            'filter_type': 'rrc',
            'sps': 4,
            'span': 6,
            'beta': 0.35
        },
        'channel': {
            'fading_type': 'awgn',
            'snr_db': 20.0,
            'awgn_enabled': True
        },
        'receiver': {
            'sync_method': 'none',
            'equalizer_type': 'none'
        },
        'validation': {
            'save_constellation': False,
            'save_spectrum': False
        }
    }
    
    # Test scenarios with varying jammer strength
    jnr_values = np.array([-10, -5, 0, 5, 10, 15, 20])
    
    scenarios = {
        'No Protection': {
            'fec': {'code_type': 'none'},
            'antenna': {'num_elements': 1, 'beamforming_enabled': False, 'gain_dbi': 0},
            'jamming': {}
        },
        'FEC (Conv R=1/2)': {
            'fec': {'code_type': 'convolutional', 'code_rate': 0.5, 'interleaver': True},
            'antenna': {'num_elements': 1, 'beamforming_enabled': False, 'gain_dbi': 0},
            'jamming': {}
        },
        'Antenna Gain (+6dB)': {
            'fec': {'code_type': 'none'},
            'antenna': {'num_elements': 4, 'beamforming_enabled': True, 'gain_dbi': 6},
            'jamming': {}
        },
        'FEC + Antenna Gain': {
            'fec': {'code_type': 'convolutional', 'code_rate': 0.5, 'interleaver': True},
            'antenna': {'num_elements': 4, 'beamforming_enabled': True, 'gain_dbi': 6},
            'jamming': {}
        }
    }
    
    # Store results
    results = {name: [] for name in scenarios}
    
    print("\nRunning JNR sweep...")
    print("-" * 80)
    
    for jnr in jnr_values:
        print(f"JNR = {jnr:+3.0f} dB: ", end='', flush=True)
        
        for scenario_name, scenario_config in scenarios.items():
            # Merge configs
            config = {}
            for key, val in base_config.items():
                config[key] = val.copy()
            
            for key, val in scenario_config.items():
                if key in config:
                    config[key].update(val)
                else:
                    config[key] = val
            
            # Set jammer power
            config['jamming']['jnr_db'] = jnr
            config['jamming']['jammer_type'] = 'barrage' if jnr > -20 else 'none'
            
            # Run simulation
            result = run_single_simulation(config)
            results[scenario_name].append(result['ber'])
            
            print(f"{result['ber']:.2e} ", end='', flush=True)
        
        print()
    
    # Print summary table
    print("\n" + "=" * 80)
    print("RESULTS SUMMARY")
    print("=" * 80)
    print(f"\n{'JNR (dB)':<12}", end='')
    for name in scenarios:
        print(f"{name:>20}", end='')
    print("\n" + "-" * 92)
    
    for i, jnr in enumerate(jnr_values):
        print(f"{jnr:>+8.0f} dB ", end='')
        for name in scenarios:
            ber = results[name][i]
            print(f"{ber:>20.2e}", end='')
        print()
    
    # Calculate improvement at JNR = 10 dB
    print("\n" + "=" * 80)
    print("IMPROVEMENT ANALYSIS (at JNR = 10 dB)")
    print("=" * 80)
    
    jnr_10_idx = np.where(jnr_values == 10)[0][0]
    baseline_ber = results['No Protection'][jnr_10_idx]
    
    for name, ber_list in results.items():
        ber = ber_list[jnr_10_idx]
        if name == 'No Protection':
            print(f"\n{name}: BER = {ber:.2e} (baseline)")
        else:
            if ber > 0 and baseline_ber > 0:
                improvement = baseline_ber / ber
                print(f"\n{name}:")
                print(f"  BER = {ber:.2e}")
                print(f"  Improvement = {improvement:.1f}x ({10*np.log10(improvement):.1f} dB)")
    
    # Create plot
    plt.figure(figsize=(10, 6))
    
    markers = ['o-', 's-', '^-', 'd-']
    colors = ['#e41a1c', '#377eb8', '#4daf4a', '#984ea3']
    
    for (name, ber_list), marker, color in zip(results.items(), markers, colors):
        ber_array = np.array(ber_list)
        # Clip BER for log scale
        ber_array = np.clip(ber_array, 1e-6, 1)
        plt.semilogy(jnr_values, ber_array, marker, label=name, color=color, 
                     linewidth=2, markersize=8)
    
    plt.xlabel('Jammer-to-Noise Ratio (JNR) [dB]', fontsize=12)
    plt.ylabel('Bit Error Rate (BER)', fontsize=12)
    plt.title('Anti-Jamming Performance Comparison\n(BPSK, SNR=20dB, AWGN Channel)', fontsize=14)
    plt.legend(loc='upper left', fontsize=10)
    plt.grid(True, alpha=0.3, which='both')
    plt.xlim([jnr_values[0]-1, jnr_values[-1]+1])
    plt.ylim([1e-4, 1])
    
    plt.tight_layout()
    output_path = Path.cwd() / 'anti_jamming_comparison.png'
    plt.savefig(str(output_path), dpi=150, bbox_inches='tight')
    plt.close()
    
    print("\n" + "=" * 80)
    print(f"Plot saved to: {output_path}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = main()
