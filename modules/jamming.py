"""
Jamming Model Module
Implements various jamming techniques for anti-jamming analysis

Supported jamming types:
- Barrage (wideband noise) jamming
- Spot (narrowband) jamming  
- Swept frequency jamming
- Pulsed jamming
- Follower (repeater) jamming
- Partial-band jamming
- Tone (multi-tone) jamming
- Smart jamming (adaptive)

Fixes applied:
- Fixed scipy.signal import conflict (renamed to sp_signal)
- Added proper handling for zero-power signals
- Added input validation
- Fixed divide-by-zero warnings in log calculations
- Added smart jammer type
- Improved documentation
- Added configurable parameters for pulsed and tone jammers
"""

import numpy as np
from scipy import signal as sp_signal
from dataclasses import dataclass, field
from typing import Optional, Tuple, Dict, Any
import warnings


@dataclass
class JammingOutput:
    """Container for jamming output"""
    jammed_signal: np.ndarray
    jammer_signal: np.ndarray
    metadata: Dict[str, Any] = field(default_factory=dict)


class JammingModule:
    """
    Jamming signal generator supporting multiple jamming types.
    
    This module simulates various electronic warfare jamming techniques
    commonly used in anti-jamming analysis and countermeasure development.
    
    Parameters
    ----------
    config : JammingConfig
        Configuration object containing jamming parameters
    sample_rate : float
        Signal sample rate in Hz
    carrier_freq : float
        Carrier frequency in Hz
        
    Attributes
    ----------
    jammer_type : str
        Type of jammer ('barrage', 'spot', 'swept', 'pulsed', 
                        'follower', 'partial_band', 'tone', 'smart', 'none')
    jammer_power_dbm : float
        Jammer transmit power in dBm
    jammer_bandwidth : float
        Jammer bandwidth in Hz
    jnr_db : float
        Jammer-to-Signal ratio in dB (positive = jammer stronger)
    sweep_rate : float
        Frequency sweep rate for swept jammer in Hz/s
    hop_period : float
        Hopping/pulsing period in seconds
    """
    
    # Supported jammer types
    SUPPORTED_JAMMERS = [
        'none', 'barrage', 'spot', 'swept', 'pulsed', 
        'follower', 'partial_band', 'tone', 'smart'
    ]
    
    def __init__(self, config, sample_rate: float, carrier_freq: float):
        """Initialize jamming module with configuration."""
        self.config = config
        self.sample_rate = float(sample_rate)
        self.carrier_freq = float(carrier_freq)
        
        # Extract config parameters with validation
        self.jammer_type = config.jammer_type.lower().strip()
        if self.jammer_type not in self.SUPPORTED_JAMMERS:
            raise ValueError(
                f"Unsupported jammer type: '{self.jammer_type}'. "
                f"Supported types: {self.SUPPORTED_JAMMERS}"
            )
        
        self.jammer_power_dbm = float(config.jammer_power_dbm)
        self.jammer_bandwidth = float(config.jammer_bandwidth)
        self.jnr_db = float(config.jnr_db)  # Jammer-to-Signal Ratio
        self.sweep_rate = float(config.sweep_rate)
        self.hop_period = float(config.hop_period)
        
        # Optional advanced parameters (with defaults)
        self.duty_cycle = getattr(config, 'duty_cycle', 0.5)
        self.num_tones = getattr(config, 'num_tones', 5)
        self.partial_band_fraction = getattr(config, 'partial_band_fraction', 0.3)
        
    def apply(self, signal: np.ndarray) -> JammingOutput:
        """
        Apply jamming to input signal.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal to be jammed (complex baseband)
            
        Returns
        -------
        JammingOutput
            Container with jammed signal, jammer signal, and metadata
        """
        # Input validation
        if signal is None or len(signal) == 0:
            raise ValueError("Input signal cannot be None or empty")
        
        signal = np.asarray(signal, dtype=complex)
        num_samples = len(signal)
        
        # Generate jammer signal
        jammer = self._generate_jammer(num_samples)
        
        # Scale jammer to desired power relative to signal
        jammer = self._scale_jammer_power(jammer, signal)
        
        # Combine signal and jammer
        jammed_signal = signal + jammer
        
        # Calculate metrics safely
        signal_power = np.mean(np.abs(signal) ** 2)
        jammer_power = np.mean(np.abs(jammer) ** 2)
        
        # Safe logarithm calculations
        if signal_power > 0 and jammer_power > 0:
            js_ratio_db = 10 * np.log10(jammer_power / signal_power)
        elif jammer_power > 0:
            js_ratio_db = np.inf
        else:
            js_ratio_db = -np.inf
            
        signal_power_db = 10 * np.log10(signal_power) if signal_power > 0 else -np.inf
        jammer_power_db = 10 * np.log10(jammer_power) if jammer_power > 0 else -np.inf
        
        metadata = {
            'jammer_type': self.jammer_type,
            'jammer_power_dbm': self.jammer_power_dbm,
            'jammer_bandwidth': self.jammer_bandwidth,
            'target_jnr_db': self.jnr_db,
            'actual_js_ratio_db': js_ratio_db,
            'signal_power_db': signal_power_db,
            'jammer_power_db': jammer_power_db,
            'num_samples': num_samples
        }
        
        return JammingOutput(
            jammed_signal=jammed_signal,
            jammer_signal=jammer,
            metadata=metadata
        )
    
    def _generate_jammer(self, num_samples: int) -> np.ndarray:
        """
        Generate jammer signal based on configured type.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate
            
        Returns
        -------
        np.ndarray
            Complex jammer signal
        """
        generators = {
            'barrage': self._barrage_jammer,
            'spot': self._spot_jammer,
            'swept': self._swept_jammer,
            'pulsed': self._pulsed_jammer,
            'follower': self._follower_jammer,
            'partial_band': self._partial_band_jammer,
            'tone': self._tone_jammer,
            'smart': self._smart_jammer,
            'none': lambda n: np.zeros(n, dtype=complex)
        }
        
        generator = generators.get(self.jammer_type)
        if generator is None:
            raise ValueError(f"Unknown jammer type: {self.jammer_type}")
            
        return generator(num_samples)
    
    def _barrage_jammer(self, num_samples: int) -> np.ndarray:
        """
        Generate barrage (wideband noise) jamming signal.
        
        Barrage jamming spreads energy across a wide bandwidth,
        effective against spread spectrum and frequency hopping systems.
        """
        # Generate complex Gaussian noise
        noise = (np.random.randn(num_samples) + 
                 1j * np.random.randn(num_samples)) / np.sqrt(2)
        
        # Bandlimit if bandwidth is less than Nyquist
        if self.jammer_bandwidth < self.sample_rate:
            jammer = self._bandlimit_signal(noise, self.jammer_bandwidth)
        else:
            jammer = noise
        
        return jammer
    
    def _spot_jammer(self, num_samples: int) -> np.ndarray:
        """
        Generate spot (narrowband) jamming signal.
        
        Spot jamming concentrates energy in a narrow band around
        the target frequency, effective against fixed-frequency systems.
        """
        # Generate complex Gaussian noise
        noise = (np.random.randn(num_samples) + 
                 1j * np.random.randn(num_samples)) / np.sqrt(2)
        
        # Filter to narrow bandwidth (10% of configured bandwidth)
        narrow_bw = max(self.jammer_bandwidth * 0.1, 100)  # Minimum 100 Hz
        jammer = self._bandlimit_signal(noise, narrow_bw)
        
        return jammer
    
    def _swept_jammer(self, num_samples: int) -> np.ndarray:
        """
        Generate swept frequency jamming signal.
        
        Swept jamming sweeps across the frequency band,
        effective against narrowband systems across multiple frequencies.
        """
        t = np.arange(num_samples) / self.sample_rate
        
        # Calculate sweep parameters
        sweep_bw = self.jammer_bandwidth
        sweep_period = sweep_bw / self.sweep_rate if self.sweep_rate > 0 else 1.0
        
        # Generate linear frequency sweep (sawtooth)
        # Instantaneous frequency varies linearly
        f_inst = (self.sweep_rate * t) % sweep_bw - sweep_bw / 2
        
        # Integrate frequency to get phase
        phase = 2 * np.pi * np.cumsum(f_inst) / self.sample_rate
        
        # Generate swept tone
        jammer = np.exp(1j * phase)
        
        return jammer
    
    def _pulsed_jammer(self, num_samples: int) -> np.ndarray:
        """
        Generate pulsed jamming signal.
        
        Pulsed jamming transmits in bursts, which can be effective
        against time-division systems and reduces average power consumption.
        """
        # Generate continuous noise
        noise = (np.random.randn(num_samples) + 
                 1j * np.random.randn(num_samples)) / np.sqrt(2)
        
        # Calculate pulse parameters
        pulse_samples = max(int(self.hop_period * self.sample_rate), 1)
        on_samples = max(int(pulse_samples * self.duty_cycle), 1)
        
        # Create pulse mask
        pulse_mask = np.zeros(num_samples, dtype=float)
        for i in range(0, num_samples, pulse_samples):
            end_idx = min(i + on_samples, num_samples)
            pulse_mask[i:end_idx] = 1.0
        
        # Apply pulsing
        jammer = noise * pulse_mask
        
        # Scale to maintain average power (compensate for duty cycle)
        if self.duty_cycle > 0:
            jammer *= np.sqrt(1.0 / self.duty_cycle)
        
        return jammer
    
    def _follower_jammer(self, num_samples: int) -> np.ndarray:
        """
        Generate follower (repeater) jamming signal.
        
        Follower jamming attempts to mimic or track the target signal,
        creating correlated interference. This simplified version uses
        filtered noise to create temporal correlation.
        """
        # Generate complex noise
        noise = (np.random.randn(num_samples) + 
                 1j * np.random.randn(num_samples))
        
        # Apply smoothing filter to create correlation (simulates processing delay)
        filter_len = max(int(0.01 * self.sample_rate), 10)  # 10ms or minimum 10 samples
        smoothing_filter = np.ones(filter_len) / filter_len
        
        # Filter real and imaginary parts separately
        real_part = np.convolve(noise.real, smoothing_filter, mode='same')
        imag_part = np.convolve(noise.imag, smoothing_filter, mode='same')
        
        jammer = (real_part + 1j * imag_part) / np.sqrt(2)
        
        return jammer
    
    def _partial_band_jammer(self, num_samples: int) -> np.ndarray:
        """
        Generate partial-band jamming signal.
        
        Partial-band jamming concentrates energy in a fraction of the
        total bandwidth, potentially more effective against some
        spread spectrum systems than barrage jamming.
        """
        # Generate wideband noise
        noise = (np.random.randn(num_samples) + 
                 1j * np.random.randn(num_samples)) / np.sqrt(2)
        
        # Bandlimit to partial bandwidth
        partial_bw = self.jammer_bandwidth * self.partial_band_fraction
        jammer = self._bandlimit_signal(noise, partial_bw)
        
        # Scale up to concentrate power in partial band
        if self.partial_band_fraction > 0:
            jammer *= np.sqrt(1.0 / self.partial_band_fraction)
        
        return jammer
    
    def _tone_jammer(self, num_samples: int) -> np.ndarray:
        """
        Generate multi-tone jamming signal.
        
        Tone jamming places discrete tones across the band,
        effective against systems with known channel frequencies.
        """
        t = np.arange(num_samples) / self.sample_rate
        jammer = np.zeros(num_samples, dtype=complex)
        
        # Generate multiple tones evenly spaced across bandwidth
        num_tones = max(self.num_tones, 1)
        tone_spacing = self.jammer_bandwidth / (num_tones + 1)
        
        for i in range(num_tones):
            # Frequency offset from center
            freq_offset = (i - (num_tones - 1) / 2) * tone_spacing
            
            # Random initial phase for each tone
            phase = np.random.uniform(0, 2 * np.pi)
            
            # Generate tone
            tone = np.exp(1j * (2 * np.pi * freq_offset * t + phase))
            jammer += tone
        
        # Normalize power
        jammer /= np.sqrt(num_tones)
        
        return jammer
    
    def _smart_jammer(self, num_samples: int) -> np.ndarray:
        """
        Generate smart (adaptive) jamming signal.
        
        Smart jamming combines multiple techniques and can adapt
        its strategy. This implementation alternates between
        spot and swept jamming.
        """
        # Divide signal into segments and apply different jamming
        segment_len = max(int(self.hop_period * self.sample_rate), 100)
        jammer = np.zeros(num_samples, dtype=complex)
        
        segment_start = 0
        use_spot = True  # Alternate between spot and swept
        
        while segment_start < num_samples:
            segment_end = min(segment_start + segment_len, num_samples)
            segment_samples = segment_end - segment_start
            
            if use_spot:
                segment_jammer = self._spot_jammer(segment_samples)
            else:
                segment_jammer = self._swept_jammer(segment_samples)
            
            jammer[segment_start:segment_end] = segment_jammer
            segment_start = segment_end
            use_spot = not use_spot
        
        return jammer
    
    def _bandlimit_signal(self, sig: np.ndarray, bandwidth: float) -> np.ndarray:
        """
        Apply bandlimiting filter to signal.
        
        Parameters
        ----------
        sig : np.ndarray
            Input signal
        bandwidth : float
            Desired bandwidth in Hz
            
        Returns
        -------
        np.ndarray
            Bandlimited signal
        """
        # Handle edge cases
        if bandwidth <= 0:
            return np.zeros_like(sig)
        
        nyquist = self.sample_rate / 2
        
        # Ensure cutoff is valid
        cutoff = min(bandwidth / 2, nyquist * 0.99)
        cutoff = max(cutoff, nyquist * 0.01)  # Minimum cutoff
        
        normalized_cutoff = cutoff / nyquist
        
        # Design Butterworth lowpass filter
        try:
            b, a = sp_signal.butter(6, normalized_cutoff, btype='low')
            filtered = sp_signal.lfilter(b, a, sig)
        except Exception as e:
            warnings.warn(f"Filter design failed: {e}. Returning unfiltered signal.")
            filtered = sig
        
        return filtered
    
    def _scale_jammer_power(self, jammer: np.ndarray, 
                           signal: np.ndarray) -> np.ndarray:
        """
        Scale jammer to achieve desired Jammer-to-Signal ratio.
        
        Parameters
        ----------
        jammer : np.ndarray
            Raw jammer signal
        signal : np.ndarray
            Target signal (for power reference)
            
        Returns
        -------
        np.ndarray
            Scaled jammer signal
        """
        # Calculate signal power
        signal_power = np.mean(np.abs(signal) ** 2)
        
        if signal_power <= 0:
            # Can't scale relative to zero-power signal
            warnings.warn("Signal power is zero. Jammer not scaled.")
            return jammer
        
        # Calculate desired jammer power from J/S ratio
        js_ratio_linear = 10 ** (self.jnr_db / 10)
        desired_jammer_power = signal_power * js_ratio_linear
        
        # Current jammer power
        current_jammer_power = np.mean(np.abs(jammer) ** 2)
        
        if current_jammer_power <= 0:
            # Jammer has no power (e.g., 'none' type)
            return jammer
        
        # Scale jammer
        scale_factor = np.sqrt(desired_jammer_power / current_jammer_power)
        jammer = jammer * scale_factor
        
        return jammer
    
    def get_jammer_spectrum(self, num_samples: int = 4096, 
                           nfft: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Get jammer power spectral density.
        
        Parameters
        ----------
        num_samples : int
            Number of samples to generate for spectrum estimation
        nfft : int
            FFT size for spectrum calculation
            
        Returns
        -------
        freqs : np.ndarray
            Frequency array in Hz
        psd : np.ndarray
            Power spectral density
        """
        jammer = self._generate_jammer(num_samples)
        
        # Compute PSD using Welch's method
        freqs, psd = sp_signal.welch(
            jammer, 
            fs=self.sample_rate, 
            nperseg=min(nfft, len(jammer)),
            return_onesided=False
        )
        
        # Shift to center DC component
        freqs = np.fft.fftshift(freqs)
        psd = np.fft.fftshift(psd)
        
        # Ensure no zeros for log plotting
        psd = np.maximum(psd, 1e-20)
        
        return freqs, psd
    
    def calculate_jamming_margin(self, signal_power_dbm: float) -> float:
        """
        Calculate jamming margin in dB.
        
        Jamming margin is the difference between signal power and jammer power.
        Positive margin means signal is stronger than jammer.
        
        Parameters
        ----------
        signal_power_dbm : float
            Signal power in dBm
            
        Returns
        -------
        float
            Jamming margin in dB
        """
        jamming_margin_db = signal_power_dbm - self.jammer_power_dbm
        return jamming_margin_db
    
    def get_jammer_effectiveness(self, ber_without_jam: float, 
                                ber_with_jam: float) -> float:
        """
        Calculate jammer effectiveness metric.
        
        Effectiveness is defined as the relative increase in BER
        caused by the jammer.
        
        Parameters
        ----------
        ber_without_jam : float
            Bit error rate without jamming
        ber_with_jam : float
            Bit error rate with jamming
            
        Returns
        -------
        float
            Jammer effectiveness (ratio of BER increase)
        """
        if ber_without_jam > 0:
            effectiveness = (ber_with_jam - ber_without_jam) / ber_without_jam
        elif ber_with_jam > 0:
            effectiveness = np.inf
        else:
            effectiveness = 0.0
        
        return effectiveness
    
    def get_info(self) -> Dict[str, Any]:
        """
        Get jammer configuration information.
        
        Returns
        -------
        dict
            Dictionary of jammer parameters
        """
        return {
            'jammer_type': self.jammer_type,
            'sample_rate': self.sample_rate,
            'carrier_freq': self.carrier_freq,
            'jammer_power_dbm': self.jammer_power_dbm,
            'jammer_bandwidth': self.jammer_bandwidth,
            'jnr_db': self.jnr_db,
            'sweep_rate': self.sweep_rate,
            'hop_period': self.hop_period,
            'duty_cycle': self.duty_cycle,
            'num_tones': self.num_tones,
            'partial_band_fraction': self.partial_band_fraction
        }
    
    def __repr__(self) -> str:
        return (f"JammingModule(type='{self.jammer_type}', "
                f"jnr={self.jnr_db}dB, bw={self.jammer_bandwidth/1e6:.2f}MHz)")
