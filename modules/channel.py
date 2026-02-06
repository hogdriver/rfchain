"""
Channel Model Module
Implements various wireless channel models including fading and AWGN
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass
from typing import Optional


@dataclass
class ChannelOutput:
    """Container for channel output"""
    signal: np.ndarray
    channel_response: Optional[np.ndarray]
    metadata: dict


class ChannelModule:
    """
    Wireless channel model supporting multiple fading types
    - Rayleigh fading
    - Rician fading
    - Nakagami fading
    - AWGN (Additive White Gaussian Noise)
    - Flat fading
    - Frequency-selective fading
    """
    
    def __init__(self, config, sample_rate: float):
        self.config = config
        self.sample_rate = sample_rate
        self.fading_type = config.fading_type.lower()
        self.doppler_freq = config.doppler_freq
        self.path_delays = np.array(config.path_delays)
        self.path_gains_db = np.array(config.path_gains)
        self.awgn_enabled = config.awgn_enabled
        self.snr_db = config.snr_db
        
        # Convert path gains from dB to linear
        self.path_gains = 10 ** (self.path_gains_db / 20)
        
        # Normalize path gains
        self.path_gains /= np.sqrt(np.sum(self.path_gains ** 2))
        
    def apply(self, signal: np.ndarray) -> ChannelOutput:
        """Apply channel model to input signal"""
        
        # Apply fading if not flat AWGN
        if self.fading_type != 'awgn':
            faded_signal, channel_response = self._apply_fading(signal)
        else:
            faded_signal = signal.copy()
            channel_response = None
        
        # Apply AWGN if enabled
        if self.awgn_enabled:
            noisy_signal = self._add_awgn(faded_signal)
        else:
            noisy_signal = faded_signal
        
        metadata = {
            'fading_type': self.fading_type,
            'doppler_freq': self.doppler_freq,
            'num_paths': len(self.path_delays),
            'snr_db': self.snr_db if self.awgn_enabled else None,
            'awgn_enabled': self.awgn_enabled
        }
        
        return ChannelOutput(
            signal=noisy_signal,
            channel_response=channel_response,
            metadata=metadata
        )
    
    def _apply_fading(self, signal: np.ndarray) -> tuple:
        """Apply fading based on channel type"""
        
        if self.fading_type == 'rayleigh':
            return self._rayleigh_fading(signal)
        elif self.fading_type == 'rician':
            return self._rician_fading(signal)
        elif self.fading_type == 'nakagami':
            return self._nakagami_fading(signal)
        elif self.fading_type == 'flat':
            return self._flat_fading(signal)
        elif self.fading_type == 'frequency_selective':
            return self._frequency_selective_fading(signal)
        else:
            raise ValueError(f"Unsupported fading type: {self.fading_type}")
    
    def _rayleigh_fading(self, signal: np.ndarray) -> tuple:
        """Rayleigh fading channel (no LOS component)"""
        # Generate complex Gaussian samples for fading
        num_samples = len(signal)
        
        # Generate fading coefficient using Jakes model
        fading_coeff = self._generate_fading_coefficients(num_samples)
        
        # Apply multipath if multiple paths
        if len(self.path_delays) > 1:
            output = self._apply_multipath(signal, fading_coeff)
        else:
            output = signal * fading_coeff
        
        return output, fading_coeff
    
    def _rician_fading(self, signal: np.ndarray, K_factor_db: float = 10.0) -> tuple:
        """Rician fading channel (with LOS component)"""
        num_samples = len(signal)
        K = 10 ** (K_factor_db / 10)  # Convert K-factor from dB
        
        # LOS component (constant)
        los_component = np.sqrt(K / (K + 1))
        
        # Scattered component (Rayleigh)
        scattered = self._generate_fading_coefficients(num_samples) * np.sqrt(1 / (K + 1))
        
        # Total fading coefficient
        fading_coeff = los_component + scattered
        
        # Apply multipath
        if len(self.path_delays) > 1:
            output = self._apply_multipath(signal, fading_coeff)
        else:
            output = signal * fading_coeff
        
        return output, fading_coeff
    
    def _nakagami_fading(self, signal: np.ndarray, m: float = 2.0) -> tuple:
        """Nakagami-m fading channel"""
        num_samples = len(signal)
        
        # Generate Nakagami fading using Gamma distribution
        # Nakagami amplitude follows Nakagami distribution
        # Related to Gamma: r² follows Gamma(m, omega/m)
        omega = 1.0  # Average power
        
        # Generate Gamma random variables
        r_squared = np.random.gamma(m, omega/m, num_samples)
        r = np.sqrt(r_squared)
        
        # Random phase
        phase = np.random.uniform(0, 2*np.pi, num_samples)
        fading_coeff = r * np.exp(1j * phase)
        
        # Apply time correlation if Doppler specified
        if self.doppler_freq > 0:
            fading_coeff = self._apply_doppler_filtering(fading_coeff)
        
        # Apply multipath
        if len(self.path_delays) > 1:
            output = self._apply_multipath(signal, fading_coeff)
        else:
            output = signal * fading_coeff
        
        return output, fading_coeff
    
    def _flat_fading(self, signal: np.ndarray) -> tuple:
        """Flat fading (frequency-nonselective)"""
        num_samples = len(signal)
        fading_coeff = self._generate_fading_coefficients(num_samples)
        output = signal * fading_coeff
        return output, fading_coeff
    
    def _frequency_selective_fading(self, signal: np.ndarray) -> tuple:
        """Frequency-selective fading with multiple paths"""
        output = self._apply_multipath(signal, None)
        
        # Create channel impulse response
        max_delay_samples = int(np.max(self.path_delays) * self.sample_rate)
        channel_response = np.zeros(max_delay_samples + 1, dtype=complex)
        
        for delay, gain in zip(self.path_delays, self.path_gains):
            delay_samples = int(delay * self.sample_rate)
            if delay_samples < len(channel_response):
                # Add time-varying fading for each path
                fading = self._generate_fading_coefficients(1)[0]
                channel_response[delay_samples] = gain * fading
        
        return output, channel_response
    
    def _generate_fading_coefficients(self, num_samples: int) -> np.ndarray:
        """Generate time-varying fading coefficients using Jakes model"""
        
        if self.doppler_freq == 0:
            # Static channel
            phase = np.random.uniform(0, 2*np.pi)
            return np.ones(num_samples) * np.exp(1j * phase)
        
        # Generate complex Gaussian samples
        i_component = np.random.randn(num_samples)
        q_component = np.random.randn(num_samples)
        fading = (i_component + 1j * q_component) / np.sqrt(2)
        
        # Apply Doppler filtering
        fading = self._apply_doppler_filtering(fading)
        
        return fading
    
    def _apply_doppler_filtering(self, fading: np.ndarray) -> np.ndarray:
        """Apply Doppler spectrum filtering"""
        # Create Doppler filter (Jakes spectrum)
        # Simplified: low-pass filter with cutoff at Doppler frequency
        
        # Normalize Doppler frequency
        fd_normalized = self.doppler_freq / self.sample_rate
        
        # Design low-pass filter
        if fd_normalized > 0 and fd_normalized < 0.5:
            # Butterworth filter
            b, a = signal.butter(4, fd_normalized * 2, btype='low')
            filtered = signal.lfilter(b, a, fading)
            
            # Normalize to unit power
            filtered = filtered / np.sqrt(np.mean(np.abs(filtered)**2))
        else:
            filtered = fading
        
        return filtered
    
    def _apply_multipath(self, signal: np.ndarray, 
                         fading_coeff: Optional[np.ndarray] = None) -> np.ndarray:
        """Apply multipath propagation"""
        output = np.zeros(len(signal), dtype=complex)
        
        for i, (delay, gain) in enumerate(zip(self.path_delays, self.path_gains)):
            delay_samples = int(delay * self.sample_rate)
            
            # Generate fading for this path
            if fading_coeff is not None:
                path_fading = fading_coeff
            else:
                path_fading = self._generate_fading_coefficients(len(signal))
            
            # Delayed and scaled signal
            if delay_samples < len(signal):
                delayed_signal = np.concatenate([
                    np.zeros(delay_samples),
                    signal[:-delay_samples] if delay_samples > 0 else signal
                ])
                output += gain * path_fading * delayed_signal
        
        return output
    
    def _add_awgn(self, signal: np.ndarray) -> np.ndarray:
        """Add Additive White Gaussian Noise"""
        # Calculate signal power
        signal_power = np.mean(np.abs(signal) ** 2)
        
        # Calculate noise power from SNR
        snr_linear = 10 ** (self.snr_db / 10)
        noise_power = signal_power / snr_linear
        
        # Generate complex Gaussian noise
        noise_std = np.sqrt(noise_power / 2)  # Divide by 2 for complex noise
        noise = noise_std * (np.random.randn(len(signal)) + 
                            1j * np.random.randn(len(signal)))
        
        return signal + noise
    
    def get_channel_impulse_response(self) -> np.ndarray:
        """Get the channel impulse response"""
        max_delay_samples = int(np.max(self.path_delays) * self.sample_rate) + 1
        h = np.zeros(max_delay_samples, dtype=complex)
        
        for delay, gain in zip(self.path_delays, self.path_gains):
            delay_samples = int(delay * self.sample_rate)
            if delay_samples < len(h):
                fading = self._generate_fading_coefficients(1)[0]
                h[delay_samples] = gain * fading
        
        return h
    
    def get_frequency_response(self, num_points: int = 512) -> tuple:
        """Get channel frequency response"""
        h = self.get_channel_impulse_response()
        w, H = signal.freqz(h, 1, worN=num_points, fs=self.sample_rate)
        return w, H
    
    def calculate_coherence_bandwidth(self) -> float:
        """Calculate coherence bandwidth from delay spread"""
        if len(self.path_delays) < 2:
            return np.inf
        
        # RMS delay spread
        mean_delay = np.sum(self.path_delays * self.path_gains**2)
        rms_delay_spread = np.sqrt(
            np.sum(((self.path_delays - mean_delay)**2) * self.path_gains**2)
        )
        
        # Coherence bandwidth (approximate)
        coherence_bw = 1 / (5 * rms_delay_spread) if rms_delay_spread > 0 else np.inf
        
        return coherence_bw
    
    def calculate_coherence_time(self) -> float:
        """Calculate coherence time from Doppler frequency"""
        if self.doppler_freq == 0:
            return np.inf
        
        # Coherence time (approximate)
        coherence_time = 9 / (16 * np.pi * self.doppler_freq)
        
        return coherence_time
