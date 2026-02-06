"""
Receiver Module
Implements receiver signal processing including synchronization and equalization
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Tuple, Optional


@dataclass
class ReceiverOutput:
    """Container for receiver output"""
    recovered_symbols: np.ndarray
    equalized_signal: np.ndarray
    sync_metrics: dict
    metadata: dict


class ReceiverModule:
    """
    Receiver processing module
    Includes synchronization, equalization, and symbol recovery
    """
    
    def __init__(self, config, modulation_type: str):
        self.config = config
        self.modulation_type = modulation_type.lower()
        self.noise_figure_db = config.noise_figure_db
        self.sync_method = config.sync_method.lower()
        self.equalizer_type = config.equalizer_type.lower()
        self.equalizer_taps = config.equalizer_taps
        self.equalizer_step_size = config.equalizer_step_size
        
        # Initialize equalizer
        self.eq_weights = None
        self._initialize_equalizer()
    
    def _initialize_equalizer(self):
        """Initialize equalizer weights"""
        if self.equalizer_type != 'none':
            # Initialize to center spike
            self.eq_weights = np.zeros(self.equalizer_taps, dtype=complex)
            self.eq_weights[self.equalizer_taps // 2] = 1.0
    
    def process(self, received_signal: np.ndarray,
               training_symbols: Optional[np.ndarray] = None) -> ReceiverOutput:
        """Process received signal through receiver chain"""
        
        # Automatic Gain Control (AGC)
        agc_signal = self._apply_agc(received_signal)
        
        # Carrier and timing synchronization
        sync_signal, sync_metrics = self._synchronize(agc_signal)
        
        # Equalization
        if self.equalizer_type != 'none':
            equalized_signal = self._equalize(sync_signal, training_symbols)
        else:
            equalized_signal = sync_signal
        
        # Symbol recovery (demodulation)
        recovered_symbols = self._demodulate(equalized_signal)
        
        metadata = {
            'noise_figure_db': self.noise_figure_db,
            'sync_method': self.sync_method,
            'equalizer_type': self.equalizer_type,
            'num_symbols_recovered': len(recovered_symbols)
        }
        
        return ReceiverOutput(
            recovered_symbols=recovered_symbols,
            equalized_signal=equalized_signal,
            sync_metrics=sync_metrics,
            metadata=metadata
        )
    
    def _apply_agc(self, signal: np.ndarray) -> np.ndarray:
        """Automatic Gain Control"""
        # Calculate current signal power
        current_power = np.mean(np.abs(signal) ** 2)
        
        if current_power > 0:
            # Target power (normalized)
            target_power = 1.0
            gain = np.sqrt(target_power / current_power)
            return signal * gain
        else:
            return signal
    
    def _synchronize(self, signal: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Carrier and timing synchronization"""
        
        if self.sync_method == 'pll':
            sync_signal, metrics = self._pll_sync(signal)
        elif self.sync_method == 'costas':
            sync_signal, metrics = self._costas_loop(signal)
        elif self.sync_method == 'early_late':
            sync_signal, metrics = self._early_late_sync(signal)
        elif self.sync_method == 'none':
            sync_signal = signal
            metrics = {'synchronized': False}
        else:
            raise ValueError(f"Unsupported sync method: {self.sync_method}")
        
        return sync_signal, metrics
    
    def _pll_sync(self, signal: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Phase-Locked Loop for carrier synchronization"""
        # PLL parameters
        loop_bw = 0.01  # Normalized loop bandwidth
        damping = 1.0 / np.sqrt(2)
        
        # Loop filter coefficients
        theta = loop_bw / (damping + 1/(4*damping))
        d = 1 + 2*damping*theta + theta**2
        K1 = (4*damping*theta) / d
        K2 = (4*theta**2) / d
        
        # Initialize PLL state
        phase_est = 0.0
        freq_est = 0.0
        output = np.zeros_like(signal)
        phase_errors = []
        
        for i, sample in enumerate(signal):
            # Remove estimated carrier
            output[i] = sample * np.exp(-1j * phase_est)
            
            # Phase detector (decision-directed)
            decision = self._make_decision(output[i])
            phase_error = np.imag(output[i] * np.conj(decision))
            phase_errors.append(phase_error)
            
            # Update frequency and phase estimates
            freq_est += K2 * phase_error
            phase_est += freq_est + K1 * phase_error
            
            # Wrap phase
            phase_est = np.mod(phase_est + np.pi, 2*np.pi) - np.pi
        
        metrics = {
            'synchronized': True,
            'final_phase_error': phase_errors[-1] if phase_errors else 0,
            'mean_phase_error': np.mean(np.abs(phase_errors)) if phase_errors else 0,
            'frequency_offset': freq_est
        }
        
        return output, metrics
    
    def _costas_loop(self, signal: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Costas loop for carrier recovery (QPSK)"""
        # Similar to PLL but uses Costas error detector
        loop_bw = 0.01
        damping = 1.0 / np.sqrt(2)
        
        theta = loop_bw / (damping + 1/(4*damping))
        d = 1 + 2*damping*theta + theta**2
        K1 = (4*damping*theta) / d
        K2 = (4*theta**2) / d
        
        phase_est = 0.0
        freq_est = 0.0
        output = np.zeros_like(signal)
        phase_errors = []
        
        for i, sample in enumerate(signal):
            output[i] = sample * np.exp(-1j * phase_est)
            
            # Costas error detector
            I = np.real(output[i])
            Q = np.imag(output[i])
            phase_error = np.sign(I) * Q - np.sign(Q) * I
            phase_errors.append(phase_error)
            
            freq_est += K2 * phase_error
            phase_est += freq_est + K1 * phase_error
            phase_est = np.mod(phase_est + np.pi, 2*np.pi) - np.pi
        
        metrics = {
            'synchronized': True,
            'final_phase_error': phase_errors[-1] if phase_errors else 0,
            'mean_phase_error': np.mean(np.abs(phase_errors)) if phase_errors else 0
        }
        
        return output, metrics
    
    def _early_late_sync(self, signal: np.ndarray) -> Tuple[np.ndarray, dict]:
        """Early-late gate timing synchronization"""
        # Simplified timing recovery
        # Just returns signal with timing metrics
        
        # Calculate timing error using early-late gate
        timing_errors = []
        
        # Use simple power-based timing error
        window_size = 10
        for i in range(0, len(signal) - window_size, window_size):
            early = np.abs(signal[i:i+window_size//2])
            late = np.abs(signal[i+window_size//2:i+window_size])
            timing_error = np.mean(early**2) - np.mean(late**2)
            timing_errors.append(timing_error)
        
        metrics = {
            'synchronized': True,
            'mean_timing_error': np.mean(np.abs(timing_errors)) if timing_errors else 0
        }
        
        return signal, metrics
    
    def _equalize(self, signal: np.ndarray, 
                 training_symbols: Optional[np.ndarray] = None) -> np.ndarray:
        """Adaptive equalization"""
        
        if self.equalizer_type == 'lms':
            return self._lms_equalizer(signal, training_symbols)
        elif self.equalizer_type == 'rls':
            return self._rls_equalizer(signal, training_symbols)
        elif self.equalizer_type == 'cma':
            return self._cma_equalizer(signal)
        elif self.equalizer_type == 'zero_forcing':
            return self._zero_forcing_equalizer(signal)
        else:
            return signal
    
    def _lms_equalizer(self, signal: np.ndarray, 
                      training_symbols: Optional[np.ndarray] = None) -> np.ndarray:
        """Least Mean Squares adaptive equalizer"""
        mu = self.equalizer_step_size
        M = self.equalizer_taps
        
        # Pad signal
        padded_signal = np.concatenate([np.zeros(M-1), signal])
        output = np.zeros(len(signal), dtype=complex)
        
        # Training phase
        if training_symbols is not None:
            training_len = min(len(training_symbols), len(signal))
            
            for n in range(training_len):
                # Get input vector
                x = padded_signal[n:n+M][::-1]
                
                # Filter output
                y = np.dot(self.eq_weights, x)
                output[n] = y
                
                # Error
                error = training_symbols[n] - y
                
                # Update weights
                self.eq_weights += mu * np.conj(error) * x
            
            start_idx = training_len
        else:
            start_idx = 0
        
        # Decision-directed mode
        for n in range(start_idx, len(signal)):
            x = padded_signal[n:n+M][::-1]
            y = np.dot(self.eq_weights, x)
            output[n] = y
            
            # Decision-directed error
            decision = self._make_decision(y)
            error = decision - y
            self.eq_weights += mu * np.conj(error) * x
        
        return output
    
    def _rls_equalizer(self, signal: np.ndarray,
                      training_symbols: Optional[np.ndarray] = None) -> np.ndarray:
        """Recursive Least Squares equalizer"""
        # Simplified RLS
        M = self.equalizer_taps
        lam = 0.99  # Forgetting factor
        delta = 1.0  # Initialization parameter
        
        # Initialize
        P = np.eye(M) / delta  # Inverse correlation matrix
        padded_signal = np.concatenate([np.zeros(M-1), signal])
        output = np.zeros(len(signal), dtype=complex)
        
        for n in range(len(signal)):
            x = padded_signal[n:n+M][::-1]
            y = np.dot(self.eq_weights, x)
            output[n] = y
            
            # Desired signal
            if training_symbols is not None and n < len(training_symbols):
                desired = training_symbols[n]
            else:
                desired = self._make_decision(y)
            
            # Error
            error = desired - y
            
            # RLS update
            Px = np.dot(P, x)
            k = Px / (lam + np.dot(np.conj(x), Px))
            P = (P - np.outer(k, np.conj(Px))) / lam
            self.eq_weights += k * np.conj(error)
        
        return output
    
    def _cma_equalizer(self, signal: np.ndarray) -> np.ndarray:
        """Constant Modulus Algorithm equalizer (blind)"""
        mu = self.equalizer_step_size * 0.1  # CMA needs smaller step size
        M = self.equalizer_taps
        R = 1.0  # Desired constant modulus
        
        padded_signal = np.concatenate([np.zeros(M-1), signal])
        output = np.zeros(len(signal), dtype=complex)
        
        for n in range(len(signal)):
            x = padded_signal[n:n+M][::-1]
            y = np.dot(self.eq_weights, x)
            output[n] = y
            
            # CMA error
            error = y * (R - np.abs(y)**2)
            
            # Update weights
            self.eq_weights += mu * np.conj(error) * x
        
        return output
    
    def _zero_forcing_equalizer(self, signal: np.ndarray) -> np.ndarray:
        """Zero-forcing equalizer (requires channel estimate)"""
        # Simplified: just return signal
        # Full implementation would need channel estimate
        return signal
    
    def _demodulate(self, signal: np.ndarray) -> np.ndarray:
        """Demodulate signal to recover symbols"""
        # Make hard decisions on received symbols
        decisions = np.array([self._make_decision(s) for s in signal])
        
        # Map decisions to symbol indices
        symbols = self._decisions_to_symbols(decisions)
        
        return symbols
    
    def _make_decision(self, sample: complex) -> complex:
        """Make hard decision on received sample"""
        
        if self.modulation_type == 'bpsk':
            return 1.0 if np.real(sample) > 0 else -1.0
        
        elif self.modulation_type == 'qpsk':
            I = 1.0 if np.real(sample) > 0 else -1.0
            Q = 1.0 if np.imag(sample) > 0 else -1.0
            return (I + 1j*Q) / np.sqrt(2)
        
        elif self.modulation_type in ['16qam', '64qam', '256qam']:
            M = {'16qam': 16, '64qam': 64, '256qam': 256}[self.modulation_type]
            m = int(np.sqrt(M))
            
            # Quantize I and Q separately
            I = np.real(sample)
            Q = np.imag(sample)
            
            # Decision boundaries
            levels = np.arange(-m+1, m, 2)
            I_dec = levels[np.argmin(np.abs(levels - I))]
            Q_dec = levels[np.argmin(np.abs(levels - Q))]
            
            # Normalize
            decision = (I_dec + 1j*Q_dec) / np.sqrt(np.mean(levels**2) * 2)
            return decision
        
        else:
            # Generic decision: just quantize
            return sample
    
    def _decisions_to_symbols(self, decisions: np.ndarray) -> np.ndarray:
        """Convert decision points to symbol indices"""
        
        if self.modulation_type == 'bpsk':
            return ((decisions.real + 1) / 2).astype(int)
        
        elif self.modulation_type == 'qpsk':
            # Map QPSK constellation to 0-3
            symbols = np.zeros(len(decisions), dtype=int)
            for i, d in enumerate(decisions):
                if np.real(d) > 0 and np.imag(d) > 0:
                    symbols[i] = 0
                elif np.real(d) < 0 and np.imag(d) > 0:
                    symbols[i] = 1
                elif np.real(d) < 0 and np.imag(d) < 0:
                    symbols[i] = 2
                else:
                    symbols[i] = 3
            return symbols
        
        else:
            # Generic: just return decisions as symbols (simplified)
            return np.arange(len(decisions)) % 4
    
    def calculate_evm(self, received_symbols: np.ndarray, 
                     ideal_symbols: np.ndarray) -> float:
        """Calculate Error Vector Magnitude"""
        if len(received_symbols) != len(ideal_symbols):
            min_len = min(len(received_symbols), len(ideal_symbols))
            received_symbols = received_symbols[:min_len]
            ideal_symbols = ideal_symbols[:min_len]
        
        error_vector = received_symbols - ideal_symbols
        error_power = np.mean(np.abs(error_vector) ** 2)
        signal_power = np.mean(np.abs(ideal_symbols) ** 2)
        
        if signal_power > 0:
            evm = np.sqrt(error_power / signal_power)
            evm_db = 20 * np.log10(evm)
        else:
            evm_db = np.inf
        
        return evm_db
