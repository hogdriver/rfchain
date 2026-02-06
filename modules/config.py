"""
Configuration Module
Handles TOML-based configuration for the RF chain framework
"""

from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, asdict

# Try to import TOML libraries, fall back gracefully
try:
    import tomli
    import tomli_w
    TOML_AVAILABLE = True
except ImportError:
    TOML_AVAILABLE = False


@dataclass
class SourceConfig:
    """Source signal configuration"""
    signal_type: str = "qpsk"
    sample_rate: float = 1e6
    carrier_freq: float = 2.4e9
    symbol_rate: float = 1e5
    num_symbols: int = 10000
    power_dbm: float = 0.0


@dataclass
class PulseShapingConfig:
    """Pulse shaping filter configuration"""
    filter_type: str = "rrc"
    span: int = 10
    sps: int = 8
    beta: float = 0.35


@dataclass
class FECConfig:
    """Forward Error Correction configuration"""
    code_type: str = "reed_solomon"
    code_rate: float = 0.5
    interleaver: bool = True
    interleaver_depth: int = 10


@dataclass
class ChannelConfig:
    """Channel model configuration"""
    fading_type: str = "rayleigh"
    doppler_freq: float = 100.0
    path_delays: list = None
    path_gains: list = None
    awgn_enabled: bool = True
    snr_db: float = 20.0
    
    def __post_init__(self):
        if self.path_delays is None:
            self.path_delays = [0.0, 1e-6, 2e-6]
        if self.path_gains is None:
            self.path_gains = [0.0, -3.0, -6.0]


@dataclass
class JammingConfig:
    """Jamming model configuration"""
    jammer_type: str = "barrage"
    jammer_power_dbm: float = 10.0
    jammer_bandwidth: float = 5e6
    jnr_db: float = 10.0
    sweep_rate: float = 1e6
    hop_period: float = 1e-3


@dataclass
class AntennaConfig:
    """Antenna configuration"""
    antenna_type: str = "omnidirectional"
    num_elements: int = 1
    element_spacing: float = 0.5
    gain_dbi: float = 0.0
    beamforming_enabled: bool = False
    steering_angle: float = 0.0


@dataclass
class ReceiverConfig:
    """Receiver configuration"""
    noise_figure_db: float = 5.0
    sync_method: str = "pll"
    equalizer_type: str = "lms"
    equalizer_taps: int = 11
    equalizer_step_size: float = 0.01


@dataclass
class ValidationConfig:
    """Validation and metrics configuration"""
    calculate_ber: bool = True
    calculate_per: bool = True
    calculate_sinr: bool = True
    calculate_evm: bool = True
    save_constellation: bool = True
    save_spectrum: bool = True


class ConfigurationManager:
    """Manages configuration loading, saving, and validation"""
    
    def __init__(self, config_path: str = None):
        self.config_path = Path(config_path) if config_path else None
        self.source = SourceConfig()
        self.pulse_shaping = PulseShapingConfig()
        self.fec = FECConfig()
        self.channel = ChannelConfig()
        self.jamming = JammingConfig()
        self.antenna = AntennaConfig()
        self.receiver = ReceiverConfig()
        self.validation = ValidationConfig()
        
        if self.config_path and self.config_path.exists():
            self.load_config()
    
    def load_config(self, config_path: str = None):
        """Load configuration from TOML file"""
        if not TOML_AVAILABLE:
            print("Warning: TOML libraries not available. Using default configuration.")
            return
            
        path = Path(config_path) if config_path else self.config_path
        
        if not path or not path.exists():
            raise FileNotFoundError(f"Configuration file not found: {path}")
        
        with open(path, 'rb') as f:
            config_dict = tomli.load(f)
        
        # Update configurations
        if 'source' in config_dict:
            self.source = SourceConfig(**config_dict['source'])
        if 'pulse_shaping' in config_dict:
            self.pulse_shaping = PulseShapingConfig(**config_dict['pulse_shaping'])
        if 'fec' in config_dict:
            self.fec = FECConfig(**config_dict['fec'])
        if 'channel' in config_dict:
            self.channel = ChannelConfig(**config_dict['channel'])
        if 'jamming' in config_dict:
            self.jamming = JammingConfig(**config_dict['jamming'])
        if 'antenna' in config_dict:
            self.antenna = AntennaConfig(**config_dict['antenna'])
        if 'receiver' in config_dict:
            self.receiver = ReceiverConfig(**config_dict['receiver'])
        if 'validation' in config_dict:
            self.validation = ValidationConfig(**config_dict['validation'])
    
    def save_config(self, config_path: str = None):
        """Save current configuration to TOML file"""
        if not TOML_AVAILABLE:
            print("Warning: TOML libraries not available. Cannot save configuration.")
            return
            
        path = Path(config_path) if config_path else self.config_path
        
        if not path:
            raise ValueError("No configuration path specified")
        
        config_dict = {
            'source': asdict(self.source),
            'pulse_shaping': asdict(self.pulse_shaping),
            'fec': asdict(self.fec),
            'channel': asdict(self.channel),
            'jamming': asdict(self.jamming),
            'antenna': asdict(self.antenna),
            'receiver': asdict(self.receiver),
            'validation': asdict(self.validation)
        }
        
        with open(path, 'wb') as f:
            tomli_w.dump(config_dict, f)
    
    def get_all_configs(self) -> Dict[str, Any]:
        """Get all configurations as a dictionary"""
        return {
            'source': self.source,
            'pulse_shaping': self.pulse_shaping,
            'fec': self.fec,
            'channel': self.channel,
            'jamming': self.jamming,
            'antenna': self.antenna,
            'receiver': self.receiver,
            'validation': self.validation
        }
    
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        errors = []
        
        # Source validation
        if self.source.sample_rate <= 0:
            errors.append("Sample rate must be positive")
        if self.source.symbol_rate > self.source.sample_rate:
            errors.append("Symbol rate cannot exceed sample rate")
        
        # Pulse shaping validation
        if self.pulse_shaping.sps < 1:
            errors.append("Samples per symbol must be >= 1")
        if not 0 <= self.pulse_shaping.beta <= 1:
            errors.append("Beta (roll-off) must be between 0 and 1")
        
        # FEC validation
        if not 0 < self.fec.code_rate <= 1:
            errors.append("Code rate must be between 0 and 1")
        
        # Channel validation
        if len(self.channel.path_delays) != len(self.channel.path_gains):
            errors.append("Path delays and gains must have same length")
        
        if errors:
            print("Configuration validation errors:")
            for error in errors:
                print(f"  - {error}")
            return False
        
        return True
