"""
RF Chain Framework Main
Integrates all modules for anti-jamming analysis
"""

import numpy as np
from pathlib import Path
import sys

# Add modules to path
sys.path.insert(0, str(Path(__file__).parent))

from modules.config import ConfigurationManager
from modules.source import SourceModule
from modules.pulse_shaping import PulseShapingModule
from modules.fec import FECModule
from modules.channel import ChannelModule
from modules.jamming import JammingModule
from modules.antenna import AntennaModule
from modules.receiver import ReceiverModule
from modules.validation import ValidationModule, ValidationMetrics


class RFChainFramework:
    """
    Complete RF Chain Framework for Anti-Jamming Analysis
    
    This framework provides a modular environment for simulating and analyzing
    RF communication systems under various jamming scenarios.
    """
    
    def __init__(self, config_path: str = None, output_dir: str = None):
        """
        Initialize RF Chain Framework
        
        Args:
            config_path: Path to TOML configuration file
            output_dir: Directory for output files (plots, etc.). Defaults to current directory.
        """
        # Set output directory
        if output_dir is None:
            self.output_dir = str(Path.cwd())
        else:
            self.output_dir = str(Path(output_dir))
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        
        # Load configuration
        self.config_manager = ConfigurationManager(config_path)
        
        if not self.config_manager.validate_config():
            raise ValueError("Invalid configuration")
        
        # Get all configurations
        configs = self.config_manager.get_all_configs()
        
        # Initialize modules
        self.source = SourceModule(configs['source'])
        self.pulse_shaping = PulseShapingModule(configs['pulse_shaping'])
        self.fec = FECModule(configs['fec'])
        
        # Channel and jamming need sample rate
        sample_rate = configs['source'].sample_rate
        self.channel = ChannelModule(configs['channel'], sample_rate)
        self.jamming = JammingModule(
            configs['jamming'], 
            sample_rate, 
            configs['source'].carrier_freq
        )
        
        # Antenna needs wavelength
        c = 3e8  # Speed of light
        wavelength = c / configs['source'].carrier_freq
        self.antenna = AntennaModule(configs['antenna'], wavelength)
        
        # Receiver
        self.receiver = ReceiverModule(
            configs['receiver'],
            configs['source'].signal_type
        )
        
        # Validation - pass output directory
        self.validation = ValidationModule(configs['validation'], output_dir=self.output_dir)
        
        # Storage for results
        self.results = {}
    
    def run_simulation(self, verbose: bool = True) -> dict:
        """
        Run complete RF chain simulation
        
        Args:
            verbose: Print progress messages
            
        Returns:
            Dictionary containing simulation results and metrics
        """
        if verbose:
            print("=" * 70)
            print("RF CHAIN SIMULATION - Anti-Jamming Analysis")
            print("=" * 70)
        
        # Step 1: Source signal generation
        if verbose:
            print("\n[1/8] Generating source signal...")
        source_output = self.source.generate()
        tx_symbols = source_output.symbols
        baseband_signal = source_output.baseband_signal
        
        # Step 2: FEC encoding
        if verbose:
            print("[2/8] Applying Forward Error Correction...")
        fec_output = self.fec.encode(tx_symbols)
        encoded_symbols = fec_output.encoded_data
        
        # Convert encoded symbols back to baseband for modulation
        # Re-modulate encoded symbols
        temp_source = SourceModule(self.config_manager.source)
        temp_source.num_symbols = len(encoded_symbols)
        encoded_signal = temp_source._modulate(encoded_symbols)
        
        # Step 3: Pulse shaping
        if verbose:
            print("[3/8] Applying pulse shaping...")
        pulse_output = self.pulse_shaping.apply(encoded_signal)
        shaped_signal = pulse_output.shaped_signal
        
        # Step 4: Transmit antenna
        if verbose:
            print("[4/8] Processing through transmit antenna...")
        tx_antenna_output = self.antenna.transmit(shaped_signal)
        tx_signal = tx_antenna_output.signal
        
        # Step 5: Channel propagation
        if verbose:
            print("[5/8] Simulating channel propagation...")
        channel_output = self.channel.apply(tx_signal)
        channel_signal = channel_output.signal
        
        # Step 6: Jamming
        if verbose:
            print("[6/8] Applying jamming...")
        jamming_output = self.jamming.apply(channel_signal)
        jammed_signal = jamming_output.jammed_signal
        jammer_signal = jamming_output.jammer_signal
        
        # Step 7: Receive antenna
        if verbose:
            print("[7/8] Processing through receive antenna...")
        rx_antenna_output = self.antenna.receive(jammed_signal)
        rx_signal = rx_antenna_output.signal
        
        # Step 8: Receiver processing
        if verbose:
            print("[8/8] Receiver processing and demodulation...")
        
        # Apply matched filter (part of pulse shaping)
        matched_signal = self.pulse_shaping.matched_filter(rx_signal)
        
        # Downsample to symbol rate
        downsampled = self.pulse_shaping.downsample(matched_signal)
        
        # Receiver processing
        receiver_output = self.receiver.process(downsampled)
        rx_symbols = receiver_output.recovered_symbols
        
        # FEC decoding
        decoded_symbols = self.fec.decode(rx_symbols)
        
        # Trim to original length
        decoded_symbols = decoded_symbols[:len(tx_symbols)]
        
        # Validation
        if verbose:
            print("\n" + "=" * 70)
            print("VALIDATION AND METRICS")
            print("=" * 70)
        
        metrics = self.validation.validate(
            transmitted_symbols=tx_symbols,
            received_symbols=decoded_symbols,
            transmitted_signal=baseband_signal,
            received_signal=downsampled,
            jammer_signal=jammer_signal,
            sample_rate=source_output.sample_rate
        )
        
        # Print summary
        if verbose:
            summary = self.validation.generate_summary_report(metrics)
            print("\n" + summary)
        
        # Store results
        self.results = {
            'transmitted_symbols': tx_symbols,
            'received_symbols': decoded_symbols,
            'transmitted_signal': baseband_signal,
            'received_signal': downsampled,
            'shaped_signal': shaped_signal,
            'channel_signal': channel_signal,
            'jammed_signal': jammed_signal,
            'jammer_signal': jammer_signal,
            'metrics': metrics,
            'source_metadata': source_output.metadata,
            'fec_metadata': fec_output.metadata,
            'pulse_shaping_metadata': pulse_output.metadata,
            'channel_metadata': channel_output.metadata,
            'jamming_metadata': jamming_output.metadata,
            'receiver_metadata': receiver_output.metadata
        }
        
        return self.results
    
    def run_parameter_sweep(self, parameter: str, values: list, 
                           metric: str = 'ber', verbose: bool = False) -> dict:
        """
        Run simulation sweep over parameter values
        
        Args:
            parameter: Parameter to sweep (e.g., 'snr_db', 'jnr_db')
            values: List of values to test
            metric: Metric to track ('ber', 'per', 'sinr_db', 'evm_db')
            verbose: Print progress
            
        Returns:
            Dictionary with parameter values and corresponding metrics
        """
        results = {'parameter': parameter, 'values': [], 'metrics': []}
        
        for i, value in enumerate(values):
            if verbose:
                print(f"\n{'='*70}")
                print(f"Sweep {i+1}/{len(values)}: {parameter} = {value}")
                print(f"{'='*70}")
            
            # Set parameter
            self._set_parameter(parameter, value)
            
            # Run simulation
            sim_results = self.run_simulation(verbose=verbose)
            
            # Extract metric
            metric_value = self._extract_metric(sim_results['metrics'], metric)
            
            results['values'].append(value)
            results['metrics'].append(metric_value)
        
        return results
    
    def _set_parameter(self, parameter: str, value):
        """Set configuration parameter"""
        # Parse parameter path (e.g., 'channel.snr_db')
        if '.' in parameter:
            module_name, param_name = parameter.split('.')
            module_config = getattr(self.config_manager, module_name)
            setattr(module_config, param_name, value)
            
            # Reinitialize affected module
            self._reinitialize_module(module_name)
        else:
            raise ValueError(f"Parameter must be in format 'module.parameter'")
    
    def _reinitialize_module(self, module_name: str):
        """Reinitialize module after config change"""
        configs = self.config_manager.get_all_configs()
        sample_rate = configs['source'].sample_rate
        carrier_freq = configs['source'].carrier_freq
        wavelength = 3e8 / carrier_freq
        
        if module_name == 'source':
            self.source = SourceModule(configs['source'])
        elif module_name == 'pulse_shaping':
            self.pulse_shaping = PulseShapingModule(configs['pulse_shaping'])
        elif module_name == 'fec':
            self.fec = FECModule(configs['fec'])
        elif module_name == 'channel':
            self.channel = ChannelModule(configs['channel'], sample_rate)
        elif module_name == 'jamming':
            self.jamming = JammingModule(configs['jamming'], sample_rate, carrier_freq)
        elif module_name == 'antenna':
            self.antenna = AntennaModule(configs['antenna'], wavelength)
        elif module_name == 'receiver':
            self.receiver = ReceiverModule(configs['receiver'], configs['source'].signal_type)
    
    def _extract_metric(self, metrics: ValidationMetrics, metric_name: str):
        """Extract specific metric value"""
        return getattr(metrics, metric_name, None)
    
    def save_configuration(self, filepath: str):
        """Save current configuration to file"""
        self.config_manager.save_config(filepath)
        print(f"Configuration saved to: {filepath}")
    
    def export_results(self, filepath: str):
        """Export simulation results to file"""
        import pickle
        
        with open(filepath, 'wb') as f:
            pickle.dump(self.results, f)
        
        print(f"Results exported to: {filepath}")


def main():
    """Example usage of RF Chain Framework"""
    
    # Create framework with default or custom config
    # Look for config in the same directory as this script
    config_path = Path(__file__).parent / "configs" / "default_config.toml"
    
    if Path(config_path).exists():
        rf_chain = RFChainFramework(config_path)
    else:
        print("No config file found, using defaults")
        rf_chain = RFChainFramework()
    
    # Run single simulation
    print("\nRunning single simulation...")
    results = rf_chain.run_simulation(verbose=True)
    
    # Example: Run SNR sweep
    print("\n\nRunning SNR sweep...")
    snr_values = np.arange(0, 25, 5)
    sweep_results = rf_chain.run_parameter_sweep(
        parameter='channel.snr_db',
        values=snr_values,
        metric='ber',
        verbose=False
    )
    
    print("\nSNR Sweep Results:")
    print("-" * 40)
    for snr, ber in zip(sweep_results['values'], sweep_results['metrics']):
        print(f"SNR = {snr:5.1f} dB  →  BER = {ber:.2e}")


if __name__ == "__main__":
    main()
