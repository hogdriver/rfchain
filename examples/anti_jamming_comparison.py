#!/usr/bin/env python3
"""
Example: Anti-Jamming Technique Comparison

This script compares the performance of different anti-jamming techniques:
1. No protection (baseline)
2. FEC only
3. Beamforming only
4. FEC + Beamforming combined
"""

import numpy as np
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from main import RFChainFramework


def create_config_scenarios():
    """Create different configuration scenarios"""
    
    base_config = {
        'source': {
            'signal_type': 'qpsk',
            'sample_rate': 1e6,
            'carrier_freq': 2.4e9,
            'symbol_rate': 1e5,
            'num_symbols': 5000,
            'power_dbm': 0.0
        },
        'pulse_shaping': {
            'filter_type': 'rrc',
            'span': 10,
            'sps': 8,
            'beta': 0.35
        },
        'channel': {
            'fading_type': 'rayleigh',
            'doppler_freq': 50.0,
            'path_delays': [0.0, 1e-6],
            'path_gains': [0.0, -3.0],
            'awgn_enabled': True,
            'snr_db': 15.0
        },
        'jamming': {
            'jammer_type': 'barrage',
            'jammer_power_dbm': 15.0,
            'jammer_bandwidth': 2e6,
            'jnr_db': 12.0,
            'sweep_rate': 1e6,
            'hop_period': 1e-3
        },
        'receiver': {
            'noise_figure_db': 5.0,
            'sync_method': 'pll',
            'equalizer_type': 'lms',
            'equalizer_taps': 11,
            'equalizer_step_size': 0.01
        },
        'validation': {
            'calculate_ber': True,
            'calculate_per': True,
            'calculate_sinr': True,
            'calculate_evm': True,
            'save_constellation': False,
            'save_spectrum': False
        }
    }
    
    scenarios = {
        'Baseline (No Protection)': {
            **base_config,
            'fec': {
                'code_type': 'none',
                'code_rate': 1.0,
                'interleaver': False,
                'interleaver_depth': 1
            },
            'antenna': {
                'antenna_type': 'omnidirectional',
                'num_elements': 1,
                'element_spacing': 0.5,
                'gain_dbi': 0.0,
                'beamforming_enabled': False,
                'steering_angle': 0.0
            }
        },
        'FEC Only': {
            **base_config,
            'fec': {
                'code_type': 'convolutional',
                'code_rate': 0.5,
                'interleaver': True,
                'interleaver_depth': 10
            },
            'antenna': {
                'antenna_type': 'omnidirectional',
                'num_elements': 1,
                'element_spacing': 0.5,
                'gain_dbi': 0.0,
                'beamforming_enabled': False,
                'steering_angle': 0.0
            }
        },
        'Beamforming Only': {
            **base_config,
            'fec': {
                'code_type': 'none',
                'code_rate': 1.0,
                'interleaver': False,
                'interleaver_depth': 1
            },
            'antenna': {
                'antenna_type': 'directional',
                'num_elements': 4,
                'element_spacing': 0.5,
                'gain_dbi': 3.0,
                'beamforming_enabled': True,
                'steering_angle': 0.0
            }
        },
        'FEC + Beamforming': {
            **base_config,
            'fec': {
                'code_type': 'convolutional',
                'code_rate': 0.5,
                'interleaver': True,
                'interleaver_depth': 10
            },
            'antenna': {
                'antenna_type': 'directional',
                'num_elements': 4,
                'element_spacing': 0.5,
                'gain_dbi': 3.0,
                'beamforming_enabled': True,
                'steering_angle': 0.0
            }
        }
    }
    
    return scenarios


def run_comparison():
    """Run comparison of anti-jamming techniques"""
    
    print("="*80)
    print("ANTI-JAMMING TECHNIQUE COMPARISON")
    print("="*80)
    print()
    
    scenarios = create_config_scenarios()
    results_summary = []
    
    for scenario_name, scenario_config in scenarios.items():
        print(f"\n{'='*80}")
        print(f"Testing: {scenario_name}")
        print(f"{'='*80}")
        
        # Create framework instance
        rf_chain = RFChainFramework()
        
        # Apply scenario configuration
        for module_name, module_config in scenario_config.items():
            module = getattr(rf_chain.config_manager, module_name)
            for param_name, param_value in module_config.items():
                setattr(module, param_name, param_value)
            rf_chain._reinitialize_module(module_name)
        
        # Run simulation
        sim_results = rf_chain.run_simulation(verbose=False)
        
        # Extract metrics
        metrics = sim_results['metrics']
        
        results_summary.append({
            'scenario': scenario_name,
            'ber': metrics.ber,
            'per': metrics.per,
            'sinr_db': metrics.sinr_db,
            'evm_db': metrics.evm_db
        })
        
        print(f"\nResults for {scenario_name}:")
        print(f"  BER:  {metrics.ber:.2e}")
        print(f"  PER:  {metrics.per:.4f}")
        print(f"  SINR: {metrics.sinr_db:.2f} dB")
        print(f"  EVM:  {metrics.evm_db:.2f} dB")
    
    # Print comparison table
    print("\n" + "="*80)
    print("COMPARISON SUMMARY")
    print("="*80)
    print(f"\n{'Technique':<25} {'BER':>12} {'PER':>10} {'SINR (dB)':>12} {'EVM (dB)':>12}")
    print("-"*80)
    
    for result in results_summary:
        print(f"{result['scenario']:<25} {result['ber']:>12.2e} {result['per']:>10.4f} "
              f"{result['sinr_db']:>12.2f} {result['evm_db']:>12.2f}")
    
    print("\n" + "="*80)
    print("ANALYSIS")
    print("="*80)
    
    baseline_ber = results_summary[0]['ber']
    
    for i, result in enumerate(results_summary[1:], 1):
        improvement_factor = baseline_ber / result['ber'] if result['ber'] > 0 else np.inf
        improvement_db = 10 * np.log10(improvement_factor) if improvement_factor < np.inf else np.inf
        
        print(f"\n{result['scenario']}:")
        print(f"  BER Improvement: {improvement_factor:.2f}x ({improvement_db:.2f} dB)")
        print(f"  SINR Gain: {result['sinr_db'] - results_summary[0]['sinr_db']:.2f} dB")


def run_jnr_sweep():
    """Sweep Jammer-to-Noise Ratio for different techniques"""
    
    print("\n" + "="*80)
    print("JAMMING RESISTANCE ANALYSIS (JNR Sweep)")
    print("="*80)
    
    jnr_values = np.arange(0, 25, 5)
    
    techniques = {
        'No Protection': {'fec': 'none', 'beamforming': False},
        'FEC Only': {'fec': 'convolutional', 'beamforming': False},
        'Beamforming Only': {'fec': 'none', 'beamforming': True},
        'FEC + Beamforming': {'fec': 'convolutional', 'beamforming': True}
    }
    
    print(f"\n{'JNR (dB)':<10}", end='')
    for tech in techniques:
        print(f"{tech:>20}", end='')
    print()
    print("-"*90)
    
    for jnr in jnr_values:
        print(f"{jnr:<10.1f}", end='')
        
        for tech_name, tech_config in techniques.items():
            rf_chain = RFChainFramework()
            
            # Configure
            rf_chain.config_manager.jamming.jnr_db = jnr
            rf_chain.config_manager.source.num_symbols = 3000  # Faster simulation
            
            if tech_config['fec'] == 'none':
                rf_chain.config_manager.fec.code_type = 'none'
            else:
                rf_chain.config_manager.fec.code_type = 'convolutional'
                rf_chain.config_manager.fec.code_rate = 0.5
            
            if tech_config['beamforming']:
                rf_chain.config_manager.antenna.num_elements = 4
                rf_chain.config_manager.antenna.beamforming_enabled = True
            else:
                rf_chain.config_manager.antenna.num_elements = 1
                rf_chain.config_manager.antenna.beamforming_enabled = False
            
            # Reinitialize
            rf_chain._reinitialize_module('jamming')
            rf_chain._reinitialize_module('fec')
            rf_chain._reinitialize_module('antenna')
            rf_chain._reinitialize_module('source')
            
            # Run
            results = rf_chain.run_simulation(verbose=False)
            ber = results['metrics'].ber
            
            print(f"{ber:>20.2e}", end='')
        
        print()


if __name__ == "__main__":
    # Run technique comparison
    run_comparison()
    
    # Run JNR sweep
    run_jnr_sweep()
    
    print("\n" + "="*80)
    print("Analysis complete!")
    print("="*80)
