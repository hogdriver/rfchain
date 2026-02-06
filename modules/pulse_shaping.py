"""
Pulse Shaping Module
Implements various pulse shaping filters for bandwidth control
"""

import numpy as np
from scipy import signal
from dataclasses import dataclass


@dataclass
class PulseShapingOutput:
    """Container for pulse shaping output"""
    shaped_signal: np.ndarray
    filter_taps: np.ndarray
    metadata: dict


class PulseShapingModule:
    """
    Pulse shaping filter implementation
    Supports RRC, RC, Gaussian, and rectangular filters
    """
    
    def __init__(self, config):
        self.config = config
        self.filter_type = config.filter_type.lower()
        self.span = config.span
        self.sps = config.sps  # samples per symbol
        self.beta = config.beta  # roll-off factor
        
        # Generate filter taps
        self.filter_taps = self._generate_filter()
    
    def apply(self, baseband_signal: np.ndarray) -> PulseShapingOutput:
        """Apply pulse shaping filter to baseband signal"""
        
        # Upsample signal
        upsampled = self._upsample(baseband_signal)
        
        # Apply filter
        shaped_signal = signal.lfilter(self.filter_taps, 1.0, upsampled)
        
        # Compensate for filter delay
        delay = len(self.filter_taps) // 2
        shaped_signal = np.concatenate([shaped_signal[delay:], np.zeros(delay)])
        
        metadata = {
            'filter_type': self.filter_type,
            'span': self.span,
            'sps': self.sps,
            'beta': self.beta,
            'filter_length': len(self.filter_taps)
        }
        
        return PulseShapingOutput(
            shaped_signal=shaped_signal,
            filter_taps=self.filter_taps,
            metadata=metadata
        )
    
    def _upsample(self, signal: np.ndarray) -> np.ndarray:
        """Upsample signal by inserting zeros"""
        upsampled = np.zeros(len(signal) * self.sps, dtype=signal.dtype)
        upsampled[::self.sps] = signal
        return upsampled
    
    def _generate_filter(self) -> np.ndarray:
        """Generate pulse shaping filter based on type"""
        
        if self.filter_type == 'rrc':
            return self._rrc_filter()
        elif self.filter_type == 'rc':
            return self._rc_filter()
        elif self.filter_type == 'gaussian':
            return self._gaussian_filter()
        elif self.filter_type == 'rectangular':
            return self._rectangular_filter()
        else:
            raise ValueError(f"Unsupported filter type: {self.filter_type}")
    
    def _rrc_filter(self) -> np.ndarray:
        """Root Raised Cosine filter"""
        N = self.span * self.sps
        t = np.arange(-N//2, N//2 + 1) / self.sps
        
        # Handle special cases
        h = np.zeros(len(t))
        
        for i, time in enumerate(t):
            if time == 0:
                h[i] = (1 + self.beta * (4/np.pi - 1))
            elif abs(abs(time) - 1/(4*self.beta)) < 1e-10:
                h[i] = (self.beta / np.sqrt(2)) * (
                    (1 + 2/np.pi) * np.sin(np.pi/(4*self.beta)) +
                    (1 - 2/np.pi) * np.cos(np.pi/(4*self.beta))
                )
            else:
                numerator = np.sin(np.pi * time * (1 - self.beta)) + \
                           4 * self.beta * time * np.cos(np.pi * time * (1 + self.beta))
                denominator = np.pi * time * (1 - (4 * self.beta * time)**2)
                h[i] = numerator / denominator
        
        # Normalize
        h = h / np.sqrt(np.sum(h**2))
        
        return h
    
    def _rc_filter(self) -> np.ndarray:
        """Raised Cosine filter"""
        N = self.span * self.sps
        t = np.arange(-N//2, N//2 + 1) / self.sps
        
        h = np.zeros(len(t))
        
        for i, time in enumerate(t):
            if time == 0:
                h[i] = 1.0
            elif abs(abs(time) - 1/(2*self.beta)) < 1e-10:
                h[i] = np.pi / 4 * np.sinc(1/(2*self.beta))
            else:
                h[i] = np.sinc(time) * np.cos(np.pi * self.beta * time) / \
                      (1 - (2 * self.beta * time)**2)
        
        # Normalize
        h = h / np.sum(h)
        
        return h
    
    def _gaussian_filter(self) -> np.ndarray:
        """Gaussian pulse shaping filter"""
        # BT product (bandwidth * symbol time)
        BT = self.beta  # Using beta as BT product
        
        N = self.span * self.sps
        t = np.arange(-N//2, N//2 + 1) / self.sps
        
        # Gaussian filter formula
        alpha = np.sqrt(np.log(2) / 2) / BT
        h = np.exp(-((alpha * t)**2))
        
        # Normalize
        h = h / np.sum(h)
        
        return h
    
    def _rectangular_filter(self) -> np.ndarray:
        """Rectangular (brick-wall) filter"""
        h = np.ones(self.sps) / self.sps
        return h
    
    def get_frequency_response(self, num_points: int = 512) -> tuple:
        """Get frequency response of the filter"""
        w, h = signal.freqz(self.filter_taps, 1, worN=num_points)
        return w, h
    
    def matched_filter(self, received_signal: np.ndarray) -> np.ndarray:
        """Apply matched filter (time-reversed conjugate of tx filter)"""
        matched_taps = np.conj(self.filter_taps[::-1])
        filtered = signal.lfilter(matched_taps, 1.0, received_signal)
        
        # Compensate for filter delay
        delay = len(matched_taps) // 2
        filtered = np.concatenate([filtered[delay:], np.zeros(delay)])
        
        return filtered
    
    def downsample(self, signal: np.ndarray, offset: int = 0) -> np.ndarray:
        """Downsample signal by taking every sps-th sample"""
        return signal[offset::self.sps]
