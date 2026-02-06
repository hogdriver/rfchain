"""
Antenna Module
Implements antenna patterns and beamforming
"""

import numpy as np
from dataclasses import dataclass
from typing import Tuple


@dataclass
class AntennaOutput:
    """Container for antenna output"""
    signal: np.ndarray
    array_response: np.ndarray
    metadata: dict


class AntennaModule:
    """
    Antenna and array processing module
    Supports various antenna types and beamforming
    """
    
    def __init__(self, config, wavelength: float):
        self.config = config
        self.wavelength = wavelength
        self.antenna_type = config.antenna_type.lower()
        self.num_elements = config.num_elements
        self.element_spacing = config.element_spacing  # in wavelengths
        self.gain_dbi = config.gain_dbi
        self.beamforming_enabled = config.beamforming_enabled
        self.steering_angle = config.steering_angle  # in degrees
        
        # Initialize array geometry
        self._initialize_array()
        
    def _initialize_array(self):
        """Initialize antenna array geometry"""
        if self.num_elements == 1:
            self.element_positions = np.array([[0.0]])
        else:
            # Linear array along x-axis
            spacing_m = self.element_spacing * self.wavelength
            positions = np.arange(self.num_elements) * spacing_m
            positions -= np.mean(positions)  # Center the array
            self.element_positions = positions.reshape(-1, 1)
    
    def transmit(self, signal: np.ndarray, 
                angle_of_arrival: float = 0.0) -> AntennaOutput:
        """Process signal through transmit antenna/array"""
        
        if self.num_elements == 1:
            # Single antenna - just apply gain
            output_signal = self._apply_antenna_gain(signal, angle_of_arrival)
            array_response = np.ones(1, dtype=complex)
        else:
            # Array processing with beamforming
            weights = self._calculate_beamforming_weights()
            output_signal, array_response = self._apply_array_weights(signal, weights)
        
        metadata = {
            'antenna_type': self.antenna_type,
            'num_elements': self.num_elements,
            'gain_dbi': self.gain_dbi,
            'beamforming': self.beamforming_enabled,
            'steering_angle': self.steering_angle if self.beamforming_enabled else None
        }
        
        return AntennaOutput(
            signal=output_signal,
            array_response=array_response,
            metadata=metadata
        )
    
    def receive(self, signal: np.ndarray, 
               angle_of_arrival: float = 0.0,
               interference_signals: list = None,
               interference_angles: list = None) -> AntennaOutput:
        """Process signal through receive antenna/array"""
        
        if self.num_elements == 1:
            # Single antenna
            output_signal = self._apply_antenna_gain(signal, angle_of_arrival)
            array_response = np.ones(1, dtype=complex)
        else:
            # Multi-element array
            if self.beamforming_enabled:
                weights = self._calculate_beamforming_weights()
            else:
                weights = np.ones(self.num_elements, dtype=complex) / np.sqrt(self.num_elements)
            
            # Receive signal from desired direction
            received_signal = self._receive_from_direction(signal, angle_of_arrival, weights)
            
            # Add interference if present
            if interference_signals and interference_angles:
                for int_sig, int_angle in zip(interference_signals, interference_angles):
                    interference = self._receive_from_direction(int_sig, int_angle, weights)
                    received_signal += interference
            
            array_response = self._get_array_response(angle_of_arrival)
            output_signal = received_signal
        
        metadata = {
            'antenna_type': self.antenna_type,
            'num_elements': self.num_elements,
            'angle_of_arrival': angle_of_arrival,
            'beamforming': self.beamforming_enabled
        }
        
        return AntennaOutput(
            signal=output_signal,
            array_response=array_response,
            metadata=metadata
        )
    
    def _apply_antenna_gain(self, signal: np.ndarray, angle: float) -> np.ndarray:
        """Apply antenna gain pattern"""
        # Convert gain from dBi to linear
        gain_linear = 10 ** (self.gain_dbi / 10)
        
        # Get directional gain based on antenna type
        directional_gain = self._get_directional_gain(angle)
        
        # Apply total gain
        total_gain = np.sqrt(gain_linear * directional_gain)
        return signal * total_gain
    
    def _get_directional_gain(self, angle: float) -> float:
        """Get directional gain based on antenna type"""
        angle_rad = np.deg2rad(angle)
        
        if self.antenna_type == 'omnidirectional':
            return 1.0
        
        elif self.antenna_type == 'directional':
            # Simple cosine pattern
            return max(0, np.cos(angle_rad)) ** 2
        
        elif self.antenna_type == 'patch':
            # Patch antenna pattern (narrower beam)
            return max(0, np.cos(angle_rad)) ** 4
        
        elif self.antenna_type == 'dipole':
            # Dipole pattern in elevation
            return np.sin(angle_rad) ** 2 if abs(angle) < 90 else 0
        
        else:
            return 1.0
    
    def _calculate_beamforming_weights(self) -> np.ndarray:
        """Calculate beamforming weights"""
        
        if not self.beamforming_enabled or self.num_elements == 1:
            return np.ones(self.num_elements, dtype=complex) / np.sqrt(self.num_elements)
        
        # Phase shift beamforming (delay-and-sum)
        steering_angle_rad = np.deg2rad(self.steering_angle)
        k = 2 * np.pi / self.wavelength  # Wavenumber
        
        # Calculate phase shifts for each element
        phase_shifts = -k * self.element_positions.flatten() * np.sin(steering_angle_rad)
        weights = np.exp(1j * phase_shifts)
        
        # Normalize
        weights = weights / np.sqrt(np.sum(np.abs(weights) ** 2))
        
        return weights
    
    def _apply_array_weights(self, signal: np.ndarray, 
                            weights: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Apply array weights to signal"""
        # For transmit: signal is combined from all elements
        # Simplified: just apply weights
        output = signal * np.sum(weights)
        return output, weights
    
    def _receive_from_direction(self, signal: np.ndarray, 
                               angle: float, 
                               weights: np.ndarray) -> np.ndarray:
        """Receive signal from specific direction with array"""
        # Calculate array response for this direction
        array_response = self._get_array_response(angle)
        
        # Apply beamforming weights
        array_factor = np.sum(weights * array_response)
        
        # Receive signal with array gain
        received = signal * array_factor
        
        return received
    
    def _get_array_response(self, angle: float) -> np.ndarray:
        """Get array response vector for given angle"""
        angle_rad = np.deg2rad(angle)
        k = 2 * np.pi / self.wavelength
        
        # Phase at each element
        phases = k * self.element_positions.flatten() * np.sin(angle_rad)
        array_response = np.exp(1j * phases)
        
        return array_response
    
    def get_array_pattern(self, num_angles: int = 361) -> Tuple[np.ndarray, np.ndarray]:
        """Get antenna array pattern"""
        angles = np.linspace(-90, 90, num_angles)
        pattern = np.zeros(num_angles)
        
        if self.num_elements == 1:
            # Single antenna pattern
            for i, angle in enumerate(angles):
                pattern[i] = self._get_directional_gain(angle)
        else:
            # Array pattern with beamforming
            weights = self._calculate_beamforming_weights()
            
            for i, angle in enumerate(angles):
                array_response = self._get_array_response(angle)
                array_factor = np.abs(np.sum(weights * array_response)) ** 2
                directional_gain = self._get_directional_gain(angle)
                pattern[i] = array_factor * directional_gain
        
        # Normalize
        if np.max(pattern) > 0:
            pattern = pattern / np.max(pattern)
        
        return angles, pattern
    
    def calculate_beamwidth(self) -> float:
        """Calculate 3dB beamwidth"""
        angles, pattern = self.get_array_pattern(1801)
        
        # Find angles where pattern is -3dB (half power)
        half_power = 0.5
        above_half_power = pattern >= half_power
        
        if not np.any(above_half_power):
            return 180.0
        
        # Find first and last index above half power around peak
        peak_idx = np.argmax(pattern)
        
        # Search left
        left_idx = peak_idx
        while left_idx > 0 and pattern[left_idx] >= half_power:
            left_idx -= 1
        
        # Search right
        right_idx = peak_idx
        while right_idx < len(pattern) - 1 and pattern[right_idx] >= half_power:
            right_idx += 1
        
        beamwidth = abs(angles[right_idx] - angles[left_idx])
        
        return beamwidth
    
    def calculate_directivity(self) -> float:
        """Calculate antenna directivity in dBi"""
        angles, pattern = self.get_array_pattern(361)
        angles_rad = np.deg2rad(angles)
        
        # Numerical integration of pattern
        # Directivity = 4*pi / integral of pattern over sphere
        # For linear array (2D pattern), simplified calculation
        
        integral = np.trapz(pattern * np.cos(angles_rad), angles_rad)
        
        if integral > 0:
            directivity_linear = np.max(pattern) * np.pi / integral
            directivity_dbi = 10 * np.log10(directivity_linear)
        else:
            directivity_dbi = 0
        
        return directivity_dbi
    
    def calculate_gain(self) -> float:
        """Calculate total antenna gain including losses"""
        # Directivity + element gain - losses (simplified)
        directivity = self.calculate_directivity()
        total_gain = directivity + self.gain_dbi
        
        # Array gain for multi-element arrays
        if self.num_elements > 1:
            array_gain = 10 * np.log10(self.num_elements)
            total_gain += array_gain
        
        return total_gain
