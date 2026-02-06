"""
Validation Module
Performance metrics and analysis for RF chain validation

This module provides comprehensive validation capabilities including:
- Bit Error Rate (BER) calculation
- Packet Error Rate (PER) calculation  
- Signal-to-Interference-plus-Noise Ratio (SINR)
- Error Vector Magnitude (EVM)
- Constellation diagram plotting
- Spectrum analysis plotting

Fixes applied:
- Fixed index out of bounds in _plot_constellation when received and ideal have different lengths
- Added input validation and error handling
- Improved constellation plot with better visualization options
- Added modulation type detection for proper ideal constellation display
- Configurable output directory for plots
- Removed debug print statements
- Added proper docstrings
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for server use
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple, List, Any
from pathlib import Path
import warnings


@dataclass
class ValidationMetrics:
    """Container for validation metrics"""
    ber: Optional[float] = None
    per: Optional[float] = None
    sinr_db: Optional[float] = None
    evm_db: Optional[float] = None
    constellation_plot: Optional[str] = None
    spectrum_plot: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


class ValidationModule:
    """
    Validation and performance analysis module.
    
    Calculates various communication system metrics and generates
    diagnostic plots for RF chain analysis.
    
    Parameters
    ----------
    config : ValidationConfig
        Configuration object with validation settings
        
    Attributes
    ----------
    calculate_ber : bool
        Whether to calculate Bit Error Rate
    calculate_per : bool
        Whether to calculate Packet Error Rate
    calculate_sinr : bool
        Whether to calculate SINR
    calculate_evm : bool
        Whether to calculate Error Vector Magnitude
    save_constellation : bool
        Whether to save constellation plot
    save_spectrum : bool
        Whether to save spectrum plot
    output_dir : str
        Directory for saving plots
    """
    
    def __init__(self, config, output_dir: str = '/home/claude'):
        """Initialize validation module."""
        self.config = config
        self.calculate_ber = getattr(config, 'calculate_ber', True)
        self.calculate_per = getattr(config, 'calculate_per', True)
        self.calculate_sinr = getattr(config, 'calculate_sinr', True)
        self.calculate_evm = getattr(config, 'calculate_evm', True)
        self.save_constellation = getattr(config, 'save_constellation', True)
        self.save_spectrum = getattr(config, 'save_spectrum', True)
        self.output_dir = output_dir
        
        # Ensure output directory exists
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def validate(self, 
                transmitted_symbols: np.ndarray,
                received_symbols: np.ndarray,
                transmitted_signal: np.ndarray,
                received_signal: np.ndarray,
                jammer_signal: Optional[np.ndarray] = None,
                noise_power: Optional[float] = None,
                sample_rate: float = 1e6,
                modulation_type: str = 'qpsk') -> ValidationMetrics:
        """
        Perform validation and calculate metrics.
        
        Parameters
        ----------
        transmitted_symbols : np.ndarray
            Original transmitted symbol indices
        received_symbols : np.ndarray
            Recovered symbol indices after demodulation
        transmitted_signal : np.ndarray
            Complex baseband transmitted signal
        received_signal : np.ndarray
            Complex baseband received signal
        jammer_signal : np.ndarray, optional
            Jammer signal for SINR calculation
        noise_power : float, optional
            Noise power for SINR calculation
        sample_rate : float
            Signal sample rate in Hz
        modulation_type : str
            Modulation type for constellation reference
            
        Returns
        -------
        ValidationMetrics
            Container with all calculated metrics
        """
        metrics = ValidationMetrics(metadata={})
        
        # Calculate BER
        if self.calculate_ber:
            metrics.ber = self._calculate_ber(transmitted_symbols, received_symbols)
            metrics.metadata['ber'] = metrics.ber
        
        # Calculate PER
        if self.calculate_per:
            metrics.per = self._calculate_per(transmitted_symbols, received_symbols)
            metrics.metadata['per'] = metrics.per
        
        # Calculate SINR
        if self.calculate_sinr and jammer_signal is not None:
            metrics.sinr_db = self._calculate_sinr(
                transmitted_signal, jammer_signal, noise_power
            )
            metrics.metadata['sinr_db'] = metrics.sinr_db
        
        # Calculate EVM
        if self.calculate_evm:
            metrics.evm_db = self._calculate_evm(transmitted_signal, received_signal)
            metrics.metadata['evm_db'] = metrics.evm_db
        
        # Generate constellation plot
        if self.save_constellation:
            metrics.constellation_plot = self._plot_constellation(
                received_signal=received_signal, 
                ideal_signal=transmitted_signal,
                modulation_type=modulation_type
            )
        
        # Generate spectrum plot
        if self.save_spectrum:
            metrics.spectrum_plot = self._plot_spectrum(
                transmitted_signal, received_signal, sample_rate
            )
        
        return metrics
    
    def _calculate_ber(self, tx_symbols: np.ndarray, 
                      rx_symbols: np.ndarray) -> float:
        """
        Calculate Bit Error Rate.
        
        Parameters
        ----------
        tx_symbols : np.ndarray
            Transmitted symbol indices
        rx_symbols : np.ndarray
            Received symbol indices
            
        Returns
        -------
        float
            Bit error rate (0.0 to 1.0)
        """
        if tx_symbols is None or rx_symbols is None:
            return 0.0
            
        if len(tx_symbols) == 0 or len(rx_symbols) == 0:
            return 0.0
        
        # Ensure same length
        min_len = min(len(tx_symbols), len(rx_symbols))
        tx_symbols = np.asarray(tx_symbols[:min_len])
        rx_symbols = np.asarray(rx_symbols[:min_len])
        
        # Convert symbols to bits
        tx_bits = self._symbols_to_bits(tx_symbols)
        rx_bits = self._symbols_to_bits(rx_symbols)
        
        # Count bit errors
        bit_errors = np.sum(tx_bits != rx_bits)
        total_bits = len(tx_bits)
        
        ber = bit_errors / total_bits if total_bits > 0 else 0.0
        
        return float(ber)
    
    def _calculate_per(self, tx_symbols: np.ndarray, 
                      rx_symbols: np.ndarray,
                      packet_size: int = 100) -> float:
        """
        Calculate Packet Error Rate.
        
        Parameters
        ----------
        tx_symbols : np.ndarray
            Transmitted symbol indices
        rx_symbols : np.ndarray
            Received symbol indices
        packet_size : int
            Number of symbols per packet
            
        Returns
        -------
        float
            Packet error rate (0.0 to 1.0)
        """
        if tx_symbols is None or rx_symbols is None:
            return 0.0
            
        min_len = min(len(tx_symbols), len(rx_symbols))
        tx_symbols = np.asarray(tx_symbols[:min_len])
        rx_symbols = np.asarray(rx_symbols[:min_len])
        
        # Split into packets
        num_packets = min_len // packet_size
        
        if num_packets == 0:
            return 0.0
        
        packet_errors = 0
        for i in range(num_packets):
            start_idx = i * packet_size
            end_idx = start_idx + packet_size
            
            tx_packet = tx_symbols[start_idx:end_idx]
            rx_packet = rx_symbols[start_idx:end_idx]
            
            # Packet is in error if any symbol is wrong
            if not np.array_equal(tx_packet, rx_packet):
                packet_errors += 1
        
        per = packet_errors / num_packets
        
        return float(per)
    
    def _calculate_sinr(self, signal: np.ndarray,
                       jammer: np.ndarray,
                       noise_power: Optional[float] = None) -> float:
        """
        Calculate Signal-to-Interference-plus-Noise Ratio.
        
        Parameters
        ----------
        signal : np.ndarray
            Desired signal
        jammer : np.ndarray
            Interference/jammer signal
        noise_power : float, optional
            Additional noise power
            
        Returns
        -------
        float
            SINR in dB
        """
        signal_power = np.mean(np.abs(signal) ** 2)
        jammer_power = np.mean(np.abs(jammer) ** 2)
        
        # Total interference power
        interference_power = jammer_power
        if noise_power is not None and noise_power > 0:
            interference_power += noise_power
        
        if interference_power > 0 and signal_power > 0:
            sinr = signal_power / interference_power
            sinr_db = 10 * np.log10(sinr)
        elif signal_power > 0:
            sinr_db = np.inf
        else:
            sinr_db = -np.inf
        
        return float(sinr_db)
    
    def _calculate_evm(self, tx_signal: np.ndarray, 
                      rx_signal: np.ndarray) -> float:
        """
        Calculate Error Vector Magnitude.
        
        Parameters
        ----------
        tx_signal : np.ndarray
            Transmitted signal (reference)
        rx_signal : np.ndarray
            Received signal
            
        Returns
        -------
        float
            EVM in dB
        """
        if tx_signal is None or rx_signal is None:
            return np.inf
            
        # Normalize signals to same length
        min_len = min(len(tx_signal), len(rx_signal))
        tx_signal = np.asarray(tx_signal[:min_len], dtype=complex)
        rx_signal = np.asarray(rx_signal[:min_len], dtype=complex)
        
        # Calculate powers
        tx_power = np.mean(np.abs(tx_signal) ** 2)
        rx_power = np.mean(np.abs(rx_signal) ** 2)
        
        if tx_power <= 0:
            return np.inf
        
        # Normalize received signal to match transmit power
        if rx_power > 0:
            rx_signal = rx_signal * np.sqrt(tx_power / rx_power)
        
        # Calculate error vector
        error = rx_signal - tx_signal
        error_power = np.mean(np.abs(error) ** 2)
        
        # EVM as percentage, then convert to dB
        evm = np.sqrt(error_power / tx_power)
        evm_db = 20 * np.log10(evm) if evm > 0 else -np.inf
        
        return float(evm_db)
    
    def _symbols_to_bits(self, symbols: np.ndarray, 
                        bits_per_symbol: int = 2) -> np.ndarray:
        """
        Convert symbol indices to bit array.
        
        Parameters
        ----------
        symbols : np.ndarray
            Symbol indices
        bits_per_symbol : int
            Number of bits per symbol
            
        Returns
        -------
        np.ndarray
            Bit array
        """
        # Ensure symbols are integers
        symbols = np.asarray(symbols).astype(int)
        
        # Clip to valid range
        max_symbol = (1 << bits_per_symbol) - 1
        symbols = np.clip(symbols, 0, max_symbol)
        
        # Convert each symbol to bits
        bits = []
        for sym in symbols:
            for i in range(bits_per_symbol - 1, -1, -1):
                bits.append((sym >> i) & 1)
        
        return np.array(bits, dtype=int)
    
    def _plot_constellation(self, 
                           received_signal: np.ndarray,
                           ideal_signal: Optional[np.ndarray] = None,
                           modulation_type: str = 'qpsk',
                           max_points: int = 2000,
                           filename: Optional[str] = None) -> str:
        """
        Generate constellation diagram.
        
        This function plots the received signal constellation points
        along with ideal constellation points for reference.
        
        Parameters
        ----------
        received_signal : np.ndarray
            Complex received signal samples
        ideal_signal : np.ndarray, optional
            Complex ideal/transmitted signal for reference
        modulation_type : str
            Modulation type ('bpsk', 'qpsk', '16qam', '64qam', etc.)
        max_points : int
            Maximum number of points to plot (for performance)
        filename : str, optional
            Output filename (default: constellation.png in output_dir)
            
        Returns
        -------
        str
            Path to saved plot file
        """
        # Input validation
        if received_signal is None or len(received_signal) == 0:
            warnings.warn("Received signal is empty. Cannot generate constellation plot.")
            return ""
        
        # Convert to complex array
        received = np.asarray(received_signal, dtype=complex).flatten()
        
        # Handle ideal signal
        if ideal_signal is not None:
            ideal = np.asarray(ideal_signal, dtype=complex).flatten()
            # Match lengths BEFORE any indexing
            min_len = min(len(received), len(ideal))
            received = received[:min_len]
            ideal = ideal[:min_len]
        else:
            ideal = None
        
        # Subsample if too many points
        num_points = len(received)
        if num_points > max_points:
            # Use random indices for subsampling
            np.random.seed(42)  # Reproducible sampling
            indices = np.random.choice(num_points, max_points, replace=False)
            indices = np.sort(indices)  # Sort for consistency
            
            received_plot = received[indices]
            ideal_plot = ideal[indices] if ideal is not None else None
        else:
            received_plot = received
            ideal_plot = ideal
        
        # Create figure
        fig, ax = plt.subplots(figsize=(8, 8))
        
        # Plot received constellation points
        ax.scatter(
            np.real(received_plot), 
            np.imag(received_plot), 
            alpha=0.4, 
            s=15, 
            c='blue', 
            label='Received',
            edgecolors='none'
        )
        
        # Plot ideal constellation points
        if ideal_plot is not None:
            # Get unique ideal points for cleaner visualization
            unique_ideal = np.unique(np.round(ideal_plot, decimals=6))
            ax.scatter(
                np.real(unique_ideal), 
                np.imag(unique_ideal),
                s=150, 
                c='red', 
                marker='x', 
                linewidths=3, 
                label='Ideal',
                zorder=10
            )
        else:
            # Generate theoretical constellation if no ideal provided
            theoretical = self._get_theoretical_constellation(modulation_type)
            if theoretical is not None:
                ax.scatter(
                    np.real(theoretical), 
                    np.imag(theoretical),
                    s=150, 
                    c='red', 
                    marker='x', 
                    linewidths=3, 
                    label='Theoretical',
                    zorder=10
                )
        
        # Plot formatting
        ax.set_xlabel('In-Phase (I)', fontsize=12)
        ax.set_ylabel('Quadrature (Q)', fontsize=12)
        ax.set_title(f'Constellation Diagram ({modulation_type.upper()})', fontsize=14)
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_aspect('equal', adjustable='box')
        ax.legend(loc='upper right', fontsize=10)
        
        # Add axis lines through origin
        ax.axhline(y=0, color='k', linewidth=0.5, alpha=0.3)
        ax.axvline(x=0, color='k', linewidth=0.5, alpha=0.3)
        
        # Set symmetric limits
        max_val = max(
            np.max(np.abs(np.real(received_plot))),
            np.max(np.abs(np.imag(received_plot)))
        ) * 1.2
        if np.isfinite(max_val) and max_val > 0:
            ax.set_xlim(-max_val, max_val)
            ax.set_ylim(-max_val, max_val)
        
        # Save figure
        if filename is None:
            filename = str(Path(self.output_dir) / 'constellation.png')
        
        plt.tight_layout()
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return filename
    
    def _get_theoretical_constellation(self, modulation_type: str) -> Optional[np.ndarray]:
        """
        Get theoretical constellation points for a modulation type.
        
        Parameters
        ----------
        modulation_type : str
            Modulation type
            
        Returns
        -------
        np.ndarray or None
            Array of constellation points
        """
        mod_type = modulation_type.lower().strip()
        
        if mod_type == 'bpsk':
            return np.array([-1, 1], dtype=complex)
        
        elif mod_type == 'qpsk':
            return np.array([1+1j, -1+1j, -1-1j, 1-1j]) / np.sqrt(2)
        
        elif mod_type == '8psk':
            angles = np.arange(8) * 2 * np.pi / 8
            return np.exp(1j * angles)
        
        elif mod_type in ['4qam', '16qam']:
            if mod_type == '4qam':
                m = 2
            else:
                m = 4
            I = np.arange(-m+1, m, 2)
            Q = np.arange(-m+1, m, 2)
            constellation = np.array([i + 1j*q for q in Q for i in I])
            # Normalize
            constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
            return constellation
        
        elif mod_type == '64qam':
            m = 8
            I = np.arange(-m+1, m, 2)
            Q = np.arange(-m+1, m, 2)
            constellation = np.array([i + 1j*q for q in Q for i in I])
            constellation /= np.sqrt(np.mean(np.abs(constellation)**2))
            return constellation
        
        else:
            return None
    
    def _plot_spectrum(self, tx_signal: np.ndarray,
                      rx_signal: np.ndarray,
                      sample_rate: float,
                      filename: Optional[str] = None) -> str:
        """
        Generate power spectral density plot.
        
        Parameters
        ----------
        tx_signal : np.ndarray
            Transmitted signal
        rx_signal : np.ndarray
            Received signal
        sample_rate : float
            Sample rate in Hz
        filename : str, optional
            Output filename
            
        Returns
        -------
        str
            Path to saved plot file
        """
        # Input validation
        if tx_signal is None or len(tx_signal) == 0:
            warnings.warn("Transmitted signal is empty.")
            tx_signal = np.zeros(1024, dtype=complex)
            
        if rx_signal is None or len(rx_signal) == 0:
            warnings.warn("Received signal is empty.")
            rx_signal = np.zeros(1024, dtype=complex)
        
        # Calculate PSDs
        f_tx, psd_tx = self._calculate_psd(tx_signal, sample_rate)
        f_rx, psd_rx = self._calculate_psd(rx_signal, sample_rate)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))
        
        # Transmitted spectrum
        axes[0].plot(f_tx / 1e6, 10 * np.log10(psd_tx), 'b-', linewidth=1)
        axes[0].set_xlabel('Frequency (MHz)', fontsize=11)
        axes[0].set_ylabel('PSD (dB/Hz)', fontsize=11)
        axes[0].set_title('Transmitted Signal Spectrum', fontsize=12)
        axes[0].grid(True, alpha=0.3)
        
        # Received spectrum
        axes[1].plot(f_rx / 1e6, 10 * np.log10(psd_rx), 'r-', linewidth=1)
        axes[1].set_xlabel('Frequency (MHz)', fontsize=11)
        axes[1].set_ylabel('PSD (dB/Hz)', fontsize=11)
        axes[1].set_title('Received Signal Spectrum', fontsize=12)
        axes[1].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        if filename is None:
            filename = str(Path(self.output_dir) / 'spectrum.png')
        
        plt.savefig(filename, dpi=150, bbox_inches='tight', facecolor='white')
        plt.close(fig)
        
        return filename
    
    def _calculate_psd(self, signal: np.ndarray, 
                      sample_rate: float,
                      nfft: int = 1024) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate Power Spectral Density using Welch's method.
        
        Parameters
        ----------
        signal : np.ndarray
            Input signal
        sample_rate : float
            Sample rate in Hz
        nfft : int
            FFT size
            
        Returns
        -------
        freqs : np.ndarray
            Frequency array
        psd : np.ndarray
            Power spectral density
        """
        from scipy import signal as sp_signal
        
        signal = np.asarray(signal, dtype=complex)
        
        # Ensure we have enough samples
        nperseg = min(nfft, len(signal))
        if nperseg < 8:
            nperseg = len(signal)
        
        try:
            f, psd = sp_signal.welch(
                signal, 
                fs=sample_rate, 
                nperseg=nperseg,
                return_onesided=False,
                detrend=False
            )
            
            # Shift to center DC
            f = np.fft.fftshift(f)
            psd = np.fft.fftshift(psd)
            
        except Exception as e:
            warnings.warn(f"PSD calculation failed: {e}")
            f = np.linspace(-sample_rate/2, sample_rate/2, nfft)
            psd = np.ones(nfft) * 1e-20
        
        # Avoid log of zero
        psd = np.maximum(psd, 1e-20)
        
        return f, psd
    
    def generate_summary_report(self, metrics: ValidationMetrics) -> str:
        """
        Generate text summary report.
        
        Parameters
        ----------
        metrics : ValidationMetrics
            Calculated metrics
            
        Returns
        -------
        str
            Formatted report string
        """
        report = []
        report.append("=" * 60)
        report.append("RF CHAIN VALIDATION REPORT")
        report.append("=" * 60)
        report.append("")
        
        if metrics.ber is not None:
            report.append(f"Bit Error Rate (BER):        {metrics.ber:.2e}")
        
        if metrics.per is not None:
            report.append(f"Packet Error Rate (PER):     {metrics.per:.4f}")
        
        if metrics.sinr_db is not None:
            if np.isfinite(metrics.sinr_db):
                report.append(f"SINR:                        {metrics.sinr_db:.2f} dB")
            else:
                report.append(f"SINR:                        {metrics.sinr_db}")
        
        if metrics.evm_db is not None:
            if np.isfinite(metrics.evm_db):
                report.append(f"Error Vector Magnitude:      {metrics.evm_db:.2f} dB")
            else:
                report.append(f"Error Vector Magnitude:      {metrics.evm_db}")
        
        report.append("")
        
        if metrics.constellation_plot:
            report.append(f"Constellation plot saved:    {metrics.constellation_plot}")
        
        if metrics.spectrum_plot:
            report.append(f"Spectrum plot saved:         {metrics.spectrum_plot}")
        
        report.append("")
        report.append("=" * 60)
        
        return "\n".join(report)
    
    def calculate_theoretical_ber(self, snr_db: float, 
                                  modulation: str) -> float:
        """
        Calculate theoretical BER for given SNR and modulation.
        
        Parameters
        ----------
        snr_db : float
            Signal-to-noise ratio in dB
        modulation : str
            Modulation type
            
        Returns
        -------
        float
            Theoretical bit error rate
        """
        from scipy.special import erfc
        
        snr_linear = 10 ** (snr_db / 10)
        mod_type = modulation.lower().strip()
        
        if mod_type == 'bpsk':
            ber = 0.5 * erfc(np.sqrt(snr_linear))
        
        elif mod_type == 'qpsk':
            # QPSK has same BER as BPSK for same Eb/N0
            ber = 0.5 * erfc(np.sqrt(snr_linear))
        
        elif mod_type == '8psk':
            # Approximate 8-PSK BER
            ber = (1/3) * erfc(np.sqrt(snr_linear) * np.sin(np.pi/8))
        
        elif mod_type == '16qam':
            # Approximate 16-QAM BER
            ber = (3/8) * erfc(np.sqrt(snr_linear / 5))
        
        elif mod_type == '64qam':
            # Approximate 64-QAM BER
            ber = (7/24) * erfc(np.sqrt(snr_linear / 21))
        
        else:
            # Generic approximation
            ber = 0.5 * np.exp(-snr_linear)
        
        return float(max(ber, 0.0))
    
    def __repr__(self) -> str:
        return (f"ValidationModule(ber={self.calculate_ber}, per={self.calculate_per}, "
                f"sinr={self.calculate_sinr}, evm={self.calculate_evm})")
