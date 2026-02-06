"""
Source Module
Generates various modulated signals for RF chain simulation
"""

import numpy as np
from typing import Tuple, Optional
from dataclasses import dataclass


@dataclass
class SourceOutput:
    """Container for source module output"""
    symbols: np.ndarray
    baseband_signal: np.ndarray
    sample_rate: float
    carrier_freq: float
    metadata: dict


class SourceModule:
    """
    Source signal generator supporting multiple modulation schemes
    """
    
    def __init__(self, config):
        self.config = config
        self.signal_type = config.signal_type.lower()
        self.sample_rate = config.sample_rate
        self.carrier_freq = config.carrier_freq
        self.symbol_rate = config.symbol_rate
        self.num_symbols = config.num_symbols
        self.power_dbm = config.power_dbm
        
    def generate(self) -> SourceOutput:
        """Generate modulated signal based on configuration"""
        
        # Generate random symbols
        symbols = self._generate_symbols()
        
        # Modulate symbols
        baseband_signal = self._modulate(symbols)
        
        # Apply power scaling
        baseband_signal = self._apply_power(baseband_signal)
        
        metadata = {
            'signal_type': self.signal_type,
            'num_symbols': self.num_symbols,
            'power_dbm': self.power_dbm,
            'symbol_rate': self.symbol_rate
        }
        
        return SourceOutput(
            symbols=symbols,
            baseband_signal=baseband_signal,
            sample_rate=self.sample_rate,
            carrier_freq=self.carrier_freq,
            metadata=metadata
        )
    
    def _generate_symbols(self) -> np.ndarray:
        """Generate random symbols based on modulation type"""
        
        if self.signal_type in ['bpsk', 'qpsk', '8psk', '16psk']:
            # PSK modulations
            M = {'bpsk': 2, 'qpsk': 4, '8psk': 8, '16psk': 16}[self.signal_type]
            symbols = np.random.randint(0, M, self.num_symbols)
        
        elif self.signal_type in ['4qam', '16qam', '64qam', '256qam']:
            # QAM modulations
            M = {'4qam': 4, '16qam': 16, '64qam': 64, '256qam': 256}[self.signal_type]
            symbols = np.random.randint(0, M, self.num_symbols)
        
        elif self.signal_type == 'ook':
            # On-Off Keying
            symbols = np.random.randint(0, 2, self.num_symbols)
        
        elif self.signal_type in ['2fsk', '4fsk', '8fsk']:
            # FSK modulations
            M = {'2fsk': 2, '4fsk': 4, '8fsk': 8}[self.signal_type]
            symbols = np.random.randint(0, M, self.num_symbols)
        
        else:
            raise ValueError(f"Unsupported signal type: {self.signal_type}")
        
        return symbols
    
    def _modulate(self, symbols: np.ndarray) -> np.ndarray:
        """Modulate symbols to complex baseband signal"""
        
        if self.signal_type == 'bpsk':
            return self._modulate_bpsk(symbols)
        elif self.signal_type == 'qpsk':
            return self._modulate_qpsk(symbols)
        elif self.signal_type == '8psk':
            return self._modulate_psk(symbols, 8)
        elif self.signal_type == '16psk':
            return self._modulate_psk(symbols, 16)
        elif self.signal_type in ['4qam', '16qam', '64qam', '256qam']:
            M = {'4qam': 4, '16qam': 16, '64qam': 64, '256qam': 256}[self.signal_type]
            return self._modulate_qam(symbols, M)
        elif self.signal_type == 'ook':
            return self._modulate_ook(symbols)
        elif self.signal_type in ['2fsk', '4fsk', '8fsk']:
            M = {'2fsk': 2, '4fsk': 4, '8fsk': 8}[self.signal_type]
            return self._modulate_fsk(symbols, M)
        else:
            raise ValueError(f"Unsupported modulation: {self.signal_type}")
    
    def _modulate_bpsk(self, symbols: np.ndarray) -> np.ndarray:
        """BPSK modulation"""
        return 2 * symbols - 1
    
    def _modulate_qpsk(self, symbols: np.ndarray) -> np.ndarray:
        """QPSK modulation"""
        constellation = np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        return constellation[symbols]
    
    def _modulate_psk(self, symbols: np.ndarray, M: int) -> np.ndarray:
        """M-PSK modulation"""
        angles = 2 * np.pi * symbols / M
        return np.exp(1j * angles)
    
    def _modulate_qam(self, symbols: np.ndarray, M: int) -> np.ndarray:
        """M-QAM modulation"""
        # Generate square QAM constellation
        m = int(np.sqrt(M))
        
        if m * m != M:
            raise ValueError(f"{M}-QAM requires M to be a perfect square")
        
        # Create constellation
        I = np.arange(-m+1, m, 2)
        Q = np.arange(-m+1, m, 2)
        constellation = np.array([i + 1j*q for q in Q for i in I])
        
        # Normalize to unit average power
        constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
        
        return constellation[symbols]
    
    def _modulate_ook(self, symbols: np.ndarray) -> np.ndarray:
        """On-Off Keying modulation"""
        return symbols.astype(float)
    
    def _modulate_fsk(self, symbols: np.ndarray, M: int) -> np.ndarray:
        """M-FSK modulation (frequency shift keying)"""
        samples_per_symbol = int(self.sample_rate / self.symbol_rate)
        total_samples = len(symbols) * samples_per_symbol
        signal = np.zeros(total_samples, dtype=complex)
        
        # Frequency deviation
        freq_dev = self.symbol_rate / 2
        
        t = np.arange(samples_per_symbol) / self.sample_rate
        
        for i, sym in enumerate(symbols):
            freq_offset = (sym - (M-1)/2) * freq_dev
            phase = 2 * np.pi * freq_offset * t
            start_idx = i * samples_per_symbol
            end_idx = start_idx + samples_per_symbol
            signal[start_idx:end_idx] = np.exp(1j * phase)
        
        return signal
    
    def _apply_power(self, signal: np.ndarray) -> np.ndarray:
        """Scale signal to desired power level"""
        # Convert dBm to linear power (assuming 1 ohm impedance)
        power_watts = 10 ** ((self.power_dbm - 30) / 10)
        
        # Current signal power
        current_power = np.mean(np.abs(signal)**2)
        
        # Scale signal
        if current_power > 0:
            scale_factor = np.sqrt(power_watts / current_power)
            signal = signal * scale_factor
        
        return signal
    
    def get_constellation(self, num_points: int = 1000) -> np.ndarray:
        """Get theoretical constellation points for the modulation scheme"""
        symbols = np.arange(min(num_points, 2**10))
        
        if self.signal_type in ['bpsk', 'qpsk', '8psk', '16psk']:
            M = {'bpsk': 2, 'qpsk': 4, '8psk': 8, '16psk': 16}[self.signal_type]
            symbols = symbols % M
        elif self.signal_type in ['4qam', '16qam', '64qam', '256qam']:
            M = {'4qam': 4, '16qam': 16, '64qam': 64, '256qam': 256}[self.signal_type]
            symbols = symbols % M
        
        return self._modulate(symbols)
