"""
Forward Error Correction (FEC) Module
Implements various error correction codes
"""

import numpy as np
from typing import Tuple
from dataclasses import dataclass


@dataclass
class FECOutput:
    """Container for FEC output"""
    encoded_data: np.ndarray
    metadata: dict


class FECModule:
    """
    Forward Error Correction implementation
    Supports Reed-Solomon, Convolutional, LDPC, and Turbo codes
    """
    
    def __init__(self, config):
        self.config = config
        self.code_type = config.code_type.lower()
        self.code_rate = config.code_rate
        self.interleaver_enabled = config.interleaver
        self.interleaver_depth = config.interleaver_depth
        
        # Initialize encoder based on code type
        self._initialize_encoder()
    
    def _initialize_encoder(self):
        """Initialize encoder parameters"""
        
        if self.code_type == 'reed_solomon':
            # Reed-Solomon parameters
            self.n = 255  # Codeword length
            self.k = int(self.n * self.code_rate)  # Message length
            
        elif self.code_type == 'convolutional':
            # Convolutional code parameters
            self.constraint_length = 7
            self.generator_polynomials = [0o171, 0o133]  # Industry standard
            
        elif self.code_type == 'ldpc':
            # LDPC parameters (simplified)
            self.block_length = 1024
            self.info_bits = int(self.block_length * self.code_rate)
            
        elif self.code_type == 'turbo':
            # Turbo code parameters
            self.constraint_length = 4
            self.generator_polynomials = [0o13, 0o15]
            
        elif self.code_type == 'none':
            # No coding
            pass
        else:
            raise ValueError(f"Unsupported FEC type: {self.code_type}")
    
    def encode(self, data: np.ndarray) -> FECOutput:
        """Encode data with FEC"""
        
        if self.code_type == 'none':
            encoded = data
        elif self.code_type == 'reed_solomon':
            encoded = self._encode_reed_solomon(data)
        elif self.code_type == 'convolutional':
            encoded = self._encode_convolutional(data)
        elif self.code_type == 'ldpc':
            encoded = self._encode_ldpc(data)
        elif self.code_type == 'turbo':
            encoded = self._encode_turbo(data)
        else:
            raise ValueError(f"Unsupported encoding: {self.code_type}")
        
        # Apply interleaver if enabled
        if self.interleaver_enabled and self.code_type != 'none':
            encoded = self._interleave(encoded)
        
        metadata = {
            'code_type': self.code_type,
            'code_rate': self.code_rate,
            'interleaved': self.interleaver_enabled,
            'original_length': len(data),
            'encoded_length': len(encoded)
        }
        
        return FECOutput(encoded_data=encoded, metadata=metadata)
    
    def decode(self, received_data: np.ndarray, 
               llr: np.ndarray = None) -> np.ndarray:
        """Decode received data"""
        
        # Deinterleave if enabled
        if self.interleaver_enabled and self.code_type != 'none':
            received_data = self._deinterleave(received_data)
        
        if self.code_type == 'none':
            decoded = received_data
        elif self.code_type == 'reed_solomon':
            decoded = self._decode_reed_solomon(received_data)
        elif self.code_type == 'convolutional':
            decoded = self._decode_convolutional(received_data, llr)
        elif self.code_type == 'ldpc':
            decoded = self._decode_ldpc(received_data, llr)
        elif self.code_type == 'turbo':
            decoded = self._decode_turbo(received_data, llr)
        else:
            raise ValueError(f"Unsupported decoding: {self.code_type}")
        
        return decoded
    
    def _encode_reed_solomon(self, data: np.ndarray) -> np.ndarray:
        """Simplified Reed-Solomon encoding"""
        # Pad data to match block size
        padded = self._pad_to_blocks(data, self.k)
        
        # Simplified encoding: add parity symbols
        num_blocks = len(padded) // self.k
        encoded = np.zeros(num_blocks * self.n, dtype=np.uint8)
        
        for i in range(num_blocks):
            block_start = i * self.k
            block_end = block_start + self.k
            block = padded[block_start:block_end]
            
            # Copy data
            enc_start = i * self.n
            encoded[enc_start:enc_start + self.k] = block
            
            # Generate parity (simplified - XOR-based)
            parity = self._generate_parity(block, self.n - self.k)
            encoded[enc_start + self.k:enc_start + self.n] = parity
        
        return encoded
    
    def _decode_reed_solomon(self, received: np.ndarray) -> np.ndarray:
        """Simplified Reed-Solomon decoding"""
        num_blocks = len(received) // self.n
        decoded = np.zeros(num_blocks * self.k, dtype=np.uint8)
        
        for i in range(num_blocks):
            block_start = i * self.n
            block = received[block_start:block_start + self.n]
            
            # Extract data portion (simplified - no error correction)
            decoded[i * self.k:(i + 1) * self.k] = block[:self.k]
        
        return decoded
    
    def _encode_convolutional(self, data: np.ndarray) -> np.ndarray:
        """Convolutional encoding"""
        # Initialize state
        state = 0
        encoded = []
        
        # Add tail bits
        data_with_tail = np.concatenate([data, np.zeros(self.constraint_length - 1)])
        
        for bit in data_with_tail:
            # Shift in new bit
            state = ((state << 1) | int(bit)) & ((1 << self.constraint_length) - 1)
            
            # Generate output bits
            for gen_poly in self.generator_polynomials:
                output_bit = bin(state & gen_poly).count('1') % 2
                encoded.append(output_bit)
        
        return np.array(encoded, dtype=np.uint8)
    
    def _decode_convolutional(self, received: np.ndarray, 
                             llr: np.ndarray = None) -> np.ndarray:
        """Viterbi decoding for convolutional codes"""
        # Simplified hard-decision Viterbi
        num_states = 2 ** (self.constraint_length - 1)
        code_rate_inv = len(self.generator_polynomials)
        
        # Reshape received data
        num_symbols = len(received) // code_rate_inv
        received = received[:num_symbols * code_rate_inv].reshape(-1, code_rate_inv)
        
        # Viterbi algorithm
        path_metrics = np.full(num_states, np.inf)
        path_metrics[0] = 0
        paths = [[] for _ in range(num_states)]
        
        for symbol in received:
            new_metrics = np.full(num_states, np.inf)
            new_paths = [[] for _ in range(num_states)]
            
            for state in range(num_states):
                if path_metrics[state] == np.inf:
                    continue
                
                for input_bit in [0, 1]:
                    # Calculate next state
                    next_state = ((state << 1) | input_bit) & (num_states - 1)
                    
                    # Calculate expected output
                    temp_state = (state << 1) | input_bit
                    expected = [bin(temp_state & g).count('1') % 2 
                              for g in self.generator_polynomials]
                    
                    # Calculate Hamming distance
                    distance = np.sum(symbol != expected)
                    metric = path_metrics[state] + distance
                    
                    if metric < new_metrics[next_state]:
                        new_metrics[next_state] = metric
                        new_paths[next_state] = paths[state] + [input_bit]
            
            path_metrics = new_metrics
            paths = new_paths
        
        # Find best path
        best_state = np.argmin(path_metrics)
        decoded = np.array(paths[best_state], dtype=np.uint8)
        
        return decoded
    
    def _encode_ldpc(self, data: np.ndarray) -> np.ndarray:
        """Simplified LDPC encoding"""
        # Pad to block size
        padded = self._pad_to_blocks(data, self.info_bits)
        
        num_blocks = len(padded) // self.info_bits
        parity_bits = self.block_length - self.info_bits
        encoded = np.zeros(num_blocks * self.block_length, dtype=np.uint8)
        
        for i in range(num_blocks):
            info_start = i * self.info_bits
            block = padded[info_start:info_start + self.info_bits]
            
            # Copy information bits
            enc_start = i * self.block_length
            encoded[enc_start:enc_start + self.info_bits] = block
            
            # Generate parity bits (simplified)
            parity = self._generate_parity(block, parity_bits)
            encoded[enc_start + self.info_bits:enc_start + self.block_length] = parity
        
        return encoded
    
    def _decode_ldpc(self, received: np.ndarray, llr: np.ndarray = None) -> np.ndarray:
        """Simplified LDPC decoding"""
        num_blocks = len(received) // self.block_length
        decoded = np.zeros(num_blocks * self.info_bits, dtype=np.uint8)
        
        for i in range(num_blocks):
            block_start = i * self.block_length
            block = received[block_start:block_start + self.block_length]
            decoded[i * self.info_bits:(i + 1) * self.info_bits] = block[:self.info_bits]
        
        return decoded
    
    def _encode_turbo(self, data: np.ndarray) -> np.ndarray:
        """Turbo code encoding"""
        # Encode with first encoder
        encoded1 = self._encode_convolutional(data)
        
        # Interleave and encode with second encoder
        interleaved = self._interleave(data)
        encoded2 = self._encode_convolutional(interleaved)
        
        # Combine (systematic + parity1 + parity2)
        # Simplified: just concatenate
        return np.concatenate([data, encoded1, encoded2])
    
    def _decode_turbo(self, received: np.ndarray, llr: np.ndarray = None) -> np.ndarray:
        """Simplified turbo decoding"""
        # Just use first decoder (simplified)
        data_len = len(received) // 3
        return self._decode_convolutional(received[data_len:2*data_len])
    
    def _interleave(self, data: np.ndarray) -> np.ndarray:
        """Block interleaver"""
        depth = self.interleaver_depth
        
        # Pad to make divisible by depth
        padded_len = int(np.ceil(len(data) / depth) * depth)
        padded = np.zeros(padded_len, dtype=data.dtype)
        padded[:len(data)] = data
        
        # Reshape and interleave
        matrix = padded.reshape(-1, depth)
        interleaved = matrix.T.flatten()
        
        return interleaved[:len(data)]
    
    def _deinterleave(self, data: np.ndarray) -> np.ndarray:
        """Block deinterleaver"""
        depth = self.interleaver_depth
        
        # Pad to make divisible by depth
        padded_len = int(np.ceil(len(data) / depth) * depth)
        padded = np.zeros(padded_len, dtype=data.dtype)
        padded[:len(data)] = data
        
        # Reshape and deinterleave
        num_cols = padded_len // depth
        matrix = padded.reshape(depth, num_cols)
        deinterleaved = matrix.T.flatten()
        
        return deinterleaved[:len(data)]
    
    def _pad_to_blocks(self, data: np.ndarray, block_size: int) -> np.ndarray:
        """Pad data to multiple of block size"""
        remainder = len(data) % block_size
        if remainder != 0:
            padding = np.zeros(block_size - remainder, dtype=data.dtype)
            return np.concatenate([data, padding])
        return data
    
    def _generate_parity(self, data: np.ndarray, num_parity: int) -> np.ndarray:
        """Generate parity bits (simplified XOR-based)"""
        parity = np.zeros(num_parity, dtype=np.uint8)
        for i in range(num_parity):
            # XOR subset of data bits
            indices = np.arange(i, len(data), num_parity)
            parity[i] = np.sum(data[indices]) % 2
        return parity
