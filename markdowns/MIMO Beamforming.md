# TUTORIAL 2: MIMO BEAMFORMING AND SIGNAL PROCESSING

## Introduction

Multiple-Input Multiple-Output (MIMO) systems use multiple antennas at both transmitter and receiver to dramatically improve wireless communication performance. For your research on joint anti-jamming and anti-eavesdropping, MIMO provides the spatial degrees of freedom needed to simultaneously enhance legitimate communication while degrading adversarial channels.

## 1. MIMO System Fundamentals

### Basic MIMO Model

**System equation:**

```
y = Hx + n
```

where:

-   **x** ∈ ℂ^(N_t × 1) = transmitted signal vector (N_t transmit antennas)
-   **y** ∈ ℂ^(N_r × 1) = received signal vector (N_r receive antennas)
-   **H** ∈ ℂ^(N_r × N_t) = channel matrix
-   **n** ∈ ℂ^(N_r × 1) = additive white Gaussian noise (AWGN) vector

**Channel matrix H:**

```
H = [h₁₁  h₁₂  ...  h₁ₙₜ]
    [h₂₁  h₂₂  ...  h₂ₙₜ]
    [ ⋮    ⋮    ⋱    ⋮  ]
    [hₙᵣ₁ hₙᵣ₂ ...  hₙᵣₙₜ]
```

Each element h_ij represents the complex channel gain from transmit antenna j to receive antenna i.

### Why MIMO Matters for Security

**Spatial diversity advantages:**

1.  **Beamforming gain:** Focus energy toward intended receiver
2.  **Null steering:** Create nulls toward eavesdroppers/jammers
3.  **Spatial multiplexing:** Send multiple data streams simultaneously
4.  **Artificial noise injection:** Degrade eavesdropper's channel without affecting legitimate receiver

## 2. Channel Models

### Flat Fading MIMO Channel

**Rayleigh fading (rich scattering):**

```
H_ij ~ CN(0, 1)  (independent and identically distributed)
```

Each channel coefficient is a complex Gaussian random variable with zero mean and unit variance.

**Rician fading (line-of-sight component):**

```
H = √(K/(K+1)) H_LOS + √(1/(K+1)) H_NLOS
```

where:

-   K = Rician K-factor (ratio of LOS to scattered power)
-   H_LOS = deterministic line-of-sight component
-   H_NLOS = Rayleigh-distributed scattered component

### Channel State Information (CSI)

**Perfect CSI:** Transmitter and receiver know H exactly

-   Enables optimal beamforming and precoding
-   Unrealistic but provides performance bounds

**Imperfect CSI:** Estimation errors

```
Ĥ = H + E
```

where E is the estimation error matrix

**Statistical CSI:** Only channel statistics known (mean, covariance)

-   More realistic for practical systems
-   Requires robust design

## 3. MIMO Capacity

### Ergodic Capacity (average over channel realizations)

**With perfect CSI at receiver only:**

```
C = E[log₂ det(I_Nr + (ρ/N_t) HH^H)] bits/s/Hz
```

where:

-   ρ = SNR = P/σ²
-   P = total transmit power
-   σ² = noise variance per receive antenna
-   I_Nr = N_r × N_r identity matrix
-   H^H = conjugate transpose (Hermitian) of H

**With perfect CSI at transmitter and receiver:** Use water-filling power allocation across spatial modes (singular values of H).

### Spatial Multiplexing Gain

For N_t transmit and N_r receive antennas:

```
Multiplexing gain = min(N_t, N_r)
```

Capacity grows linearly with min(N_t, N_r) at high SNR!

## 4. Beamforming Fundamentals

Beamforming is the process of shaping the transmitted signal's spatial pattern to focus energy in desired directions.

### Transmit Beamforming

**Signal model:**

```
x = w s
```

where:

-   s = scalar information symbol, E[|s|²] = 1
-   w ∈ ℂ^(N_t × 1) = beamforming vector (weight vector)
-   Power constraint: ||w||² ≤ P

**Received signal:**

```
y = Hw s + n = (H w) s + n
```

The effective channel becomes the scalar h_eff = H w.

### Receive Beamforming (Combining)

**Signal model:**

```
z = v^H y
```

where:

-   v ∈ ℂ^(N_r × 1) = receive combining vector
-   z = combined scalar output

**Combined signal:**

```
z = v^H Hx + v^H n
```

### Joint Transmit-Receive Beamforming

**Effective channel:**

```
h_eff = v^H H w
```

**Received SNR:**

```
SNR = |v^H H w|² / (σ² ||v||²)
```

## 5. Classical Beamforming Techniques

### Maximum Ratio Transmission (MRT)

**Objective:** Maximize received signal power

**Transmit beamforming vector:**

```
w_MRT = √P · (H^H v) / ||H^H v||
```

For single receiver antenna (MISO): v = 1, so:

```
w_MRT = √P · h^H / ||h||
```

where h is the channel vector.

**Properties:**

-   Simple to implement
-   Optimal for single-user scenario without interference
-   Does NOT suppress interference or eavesdroppers

### Zero-Forcing (ZF) Beamforming

**Objective:** Null out interference to unintended receivers

For multi-user scenario with K users and channel matrix:

```
H = [h₁^H]
    [h₂^H]
    [ ⋮  ]
    [hₖ^H]
```

**ZF precoding matrix:**

```
W_ZF = H^H (H H^H)^(-1)
```

Then normalize columns to satisfy power constraint.

**Properties:**

-   Creates perfect nulls at unintended receivers
-   Useful for multi-user MIMO and anti-eavesdropping
-   Can suffer from noise enhancement when channels are poorly conditioned

### Minimum Mean Square Error (MMSE) Beamforming

**Objective:** Minimize mean square error between transmitted and received symbols

**Receive combining (MMSE receiver):**

```
v_MMSE = (H^H H + (σ²/P) I)^(-1) H^H
```

**Transmit beamforming (MMSE precoding):**

```
W_MMSE = H^H (H H^H + (σ²/P) I)^(-1)
```

**Properties:**

-   Balances signal enhancement and noise/interference suppression
-   Outperforms MRT and ZF in moderate SNR regimes
-   Requires knowledge of noise variance

## 6. Singular Value Decomposition (SVD) for MIMO

SVD decomposes the channel matrix:

```
H = U Σ V^H
```

where:

-   U ∈ ℂ^(N_r × N_r) = unitary matrix (left singular vectors)
-   V ∈ ℂ^(N_t × N_t) = unitary matrix (right singular vectors)
-   Σ = diagonal matrix of singular values [σ₁, σ₂, ..., σ_min(Nr,Nt)]

**Application:**

-   Transmit precoding: x = V s (where s is the data vector)
-   Receive combining: ỹ = U^H y

**Result:** Parallel independent channels (eigenbeams)

```
ỹ = Σ s + ñ
```

Each eigenbeam has gain σᵢ.

### Water-Filling Power Allocation

Allocate power across eigenbeams to maximize capacity:

```
Pᵢ = (μ - σ²/σᵢ²)⁺
```

where:

-   μ = water-filling level (chosen to satisfy power constraint)
-   (x)⁺ = max(0, x)
-   Allocate more power to stronger eigenbeams
-   Don't use eigenbeams where σᵢ² < σ²/μ

## 7. Beamforming for Physical Layer Security

### Artificial Noise (AN) Beamforming

**Concept:** Transmit information signal + artificial noise

-   Information beamformed toward Bob
-   AN beamformed to degrade Eve's channel without affecting Bob

**Signal model:**

```
x = w_s s + W_n z
```

where:

-   w_s = information beamforming vector
-   W_n = AN beamforming matrix
-   z = artificial noise vector (random, E[zz^H] = I)

**Design constraint:**

```
H_B W_n = 0  (AN doesn't affect Bob)
```

**Null space approach:** Bob's channel is H_B ∈ ℂ^(1 × N_t). Find null space N(H_B):

```
W_n spans null space of H_B
```

### Secrecy Rate with AN

**Bob's received signal:**

```
y_B = H_B w_s s + n_B
```

(AN component is nulled)

**Eve's received signal:**

```
y_E = H_E w_s s + H_E W_n z + n_E
```

(AN appears as additional interference)

**Achievable secrecy rate:**

```
R_s = [log₂(1 + SNR_B) - log₂(1 + SINR_E)]⁺
```

where:

-   SNR_B = |H_B w_s|² / σ²_B
-   SINR_E = |H_E w_s|² / (||H_E W_n||²_F + σ²_E)

### Power Allocation Between Signal and AN

Total power constraint:

```
||w_s||² + ||W_n||²_F ≤ P
```

**Optimization problem:**

```
max R_s
subject to: ||w_s||² + ||W_n||²_F ≤ P
            H_B W_n = 0
```

This is typically non-convex and requires iterative solutions.

## 8. Multi-User MIMO Beamforming

### Downlink (Broadcast Channel)

Transmitter serves K users simultaneously.

**Signal model:**

```
y_k = H_k ∑ᵢ wᵢ sᵢ + n_k
```

where:

-   H_k = channel to user k
-   wᵢ = beamforming vector for user i's data stream
-   sᵢ = data symbol for user i

**Zero-Forcing precoding:** Design {wᵢ} such that:

```
H_k wᵢ = 0  for all k ≠ i
```

Ensures no inter-user interference.

### Block Diagonalization (BD)

Extension of ZF for multi-antenna users.

**Objective:** Pre-cancel all inter-user interference

For user k, construct precoding matrix V_k such that:

```
H_j V_k = 0  for all j ≠ k
```

## 9. Robust Beamforming Under Imperfect CSI

### Uncertainty Models

**Bounded error model:**

```
H = Ĥ + E,  ||E||_F ≤ ε
```

**Stochastic error model:**

```
H = Ĥ + E,  E ~ CN(0, C_E)
```

### Worst-Case Robust Design

**Optimization:**

```
max  min  SNR(H, w)
 w   H∈U
```

where U is the uncertainty set for H.

**Result:** Typically leads to more conservative beamforming (wider beams, less aggressive nulling).

### Outage-Based Design

Design for certain outage probability:

```
Pr[SNR < threshold] ≤ ε_outage
```

Common for fading channels where perfect CSI is impossible.

## 10. Practical Implementation Considerations

### Antenna Array Geometries

**Uniform Linear Array (ULA):**

-   Antennas spaced λ/2 apart in a line
-   Simple, commonly used
-   Provides beamforming in one dimension (azimuth)

**Uniform Planar Array (UPA):**

-   2D grid of antennas
-   Enables beamforming in both azimuth and elevation
-   Used in massive MIMO base stations

**Array response vector for ULA:**

```
a(θ) = [1, e^(jπ sin θ), e^(j2π sin θ), ..., e^(j(N-1)π sin θ)]^T
```

where θ is the angle of arrival/departure.

### Computational Complexity

**Matrix operations dominate:**

-   Matrix inversion: O(N³)
-   SVD: O(min(M,N) · M · N) for M×N matrix
-   Eigenvalue decomposition: O(N³)

**Real-time constraints:**

-   Channel coherence time limits update rate
-   Fast algorithms and approximations often necessary
-   Hardware accelerators (FPGAs, ASICs) for massive MIMO

### Calibration Requirements

**Transmit-receive calibration:**

-   Required for TDD reciprocity-based beamforming
-   Compensate for RF chain mismatches

**Antenna mutual coupling:**

-   Adjacent antennas affect each other
-   Can degrade beamforming performance if not modeled

## 11. Python Implementation Examples

### Example 1: MRT Beamforming

python

```python
import numpy as np

# System parameters
Nt = 4  # Number of transmit antennas
Nr = 1  # Number of receive antennas (MISO)
P = 1.0  # Transmit power
sigma2 = 0.1  # Noise variance

# Generate random Rayleigh fading channel
h = (np.random.randn(Nr, Nt) + 1j*np.random.randn(Nr, Nt)) / np.sqrt(2)

# MRT beamforming vector
w_MRT = np.sqrt(P) * h.conj().T / np.linalg.norm(h)

# Effective channel
h_eff = h @ w_MRT

# Received SNR
SNR = np.abs(h_eff)**2 / sigma2
print(f"MRT SNR: {10*np.log10(SNR[0,0]):.2f} dB")
```

### Example 2: Zero-Forcing Multi-User Beamforming

python

```python
import numpy as np

# System parameters
Nt = 8   # Transmit antennas
K = 4    # Number of users (each with 1 antenna)
P = 1.0  # Total power

# Generate channels for K users (K × Nt matrix)
H = (np.random.randn(K, Nt) + 1j*np.random.randn(K, Nt)) / np.sqrt(2)

# Zero-Forcing precoding
W_ZF = H.conj().T @ np.linalg.inv(H @ H.conj().T)

# Power normalization
power_per_user = np.sum(np.abs(W_ZF)**2, axis=0)
W_ZF = W_ZF @ np.diag(np.sqrt(P/K / power_per_user))

# Verify nulling (should be near zero for off-diagonal)
interference = H @ W_ZF
print("Interference matrix (should be approximately diagonal):")
print(np.abs(interference))
```

### Example 3: Artificial Noise Beamforming

python

```python
import numpy as np
from scipy.linalg import null_space

# System parameters
Nt = 6   # Transmit antennas
P = 1.0  # Total power
alpha = 0.7  # Fraction of power for information signal

# Channels
h_B = (np.random.randn(1, Nt) + 1j*np.random.randn(1, Nt)) / np.sqrt(2)  # Bob
h_E = (np.random.randn(1, Nt) + 1j*np.random.randn(1, Nt)) / np.sqrt(2)  # Eve

# Information beamforming (MRT toward Bob)
w_s = np.sqrt(alpha * P) * h_B.conj().T / np.linalg.norm(h_B)

# Artificial noise in null space of Bob's channel
N_B = null_space(h_B)  # Null space basis
# Random coefficients for AN
z = (np.random.randn(N_B.shape[1], 1) + 1j*np.random.randn(N_B.shape[1], 1)) / np.sqrt(2)
W_n = np.sqrt((1-alpha) * P / N_B.shape[1]) * N_B @ z

# Verify Bob receives no AN
print(f"AN at Bob (should be ~0): {np.abs(h_B @ W_n)[0,0]:.6f}")

# Eve receives both signal and AN
signal_Eve = h_E @ w_s
AN_Eve = h_E @ W_n
print(f"Signal power at Eve: {np.abs(signal_Eve[0,0])**2:.4f}")
print(f"AN power at Eve: {np.abs(AN_Eve[0,0])**2:.4f}")
```

### Example 4: SVD-Based Spatial Multiplexing

python

```python
import numpy as np
import matplotlib.pyplot as plt

# System parameters
Nt = 4
Nr = 4
P = 1.0
sigma2 = 0.01

# Channel matrix
H = (np.random.randn(Nr, Nt) + 1j*np.random.randn(Nr, Nt)) / np.sqrt(2)

# SVD
U, S, Vh = np.linalg.svd(H)

# Number of spatial streams
num_streams = len(S)

# Water-filling power allocation
mu = 0  # Water-filling level (found iteratively)
for mu_candidate in np.linspace(sigma2/np.max(S)**2, sigma2/np.min(S)**2 * 10, 1000):
    P_allocated = np.sum(np.maximum(mu_candidate - sigma2/S**2, 0))
    if P_allocated <= P:
        mu = mu_candidate
        break

# Power per stream
P_streams = np.maximum(mu - sigma2/S**2, 0)
print(f"Power allocation: {P_streams}")

# Capacity with water-filling
capacity = np.sum(np.log2(1 + S**2 * P_streams / sigma2))
print(f"MIMO capacity: {capacity:.2f} bits/s/Hz")
```

## 12. Connection to Anti-Jamming and Anti-Eavesdropping

### Simultaneous Null Steering

**Challenge:** Create nulls toward both jammer and eavesdropper

**Approach:** Design beamforming vector w such that:
```
H_E w ≈ 0  (null toward Eve)
H_J w ≈ 0  (null toward jammer)
H_B w is maximized  (strong signal to Bob)
```

**Constraint:** Requires N_t ≥ 3 (at least one DoF for signal, one null for Eve, one null for jammer)

### Beamforming with Jamming

**Received signal at Bob:**
```
y_B = H_B w s + H_JB j + n_B
```

where H_JB is jammer-to-Bob channel, j is jamming signal.

**SINR at Bob:**
```
SINR_B = |H_B w|² / (|H_JB|² P_J + σ²_B)
```

**Design approaches:**
1. Maximize SINR_B while creating nulls toward Eve
2. Joint optimization of beamforming and power allocation
3. Adaptive beamforming that tracks jammer location

## 13. Advanced Topics

### Massive MIMO

**Definition:** Systems with very large antenna arrays (64, 128, 256+ antennas)

**Key properties:**
- Channel vectors to different users become nearly orthogonal
- Simple linear processing (MRT, ZF) approaches optimal
- Robustness to imperfect CSI improves

**Security implications:**
- Extremely narrow beams → inherent spatial security
- Large null space for artificial noise
- Eavesdropper localization becomes easier

### Hybrid Beamforming

**Motivation:** Reduce number of RF chains (cost, power)

**Architecture:**
- Analog beamforming (phase shifters) in RF domain
- Digital beamforming (baseband processing)

**Signal model:**
```
x = F_RF F_BB s
```

where F_RF is analog precoder (constant modulus constraint), F_BB is digital precoder.

### Intelligent Reflecting Surfaces (IRS)

**Concept:** Programmable metasurfaces that reflect signals with controllable phase shifts

**Combined with MIMO:**

-   Additional spatial degrees of freedom
-   Can create favorable propagation environments
-   Useful for anti-jamming (reflect around obstacles) and anti-eavesdropping (avoid Eve)

## 14. Key Takeaways

-   **MIMO provides spatial DoF** critical for simultaneously achieving anti-jamming and anti-eavesdropping
-   **Beamforming techniques:**
    -   MRT: Maximize signal strength (single user)
    -   ZF: Null interference (multi-user, anti-eavesdropping)
    -   MMSE: Balance signal and interference
-   **Artificial noise** exploits null space to degrade eavesdropper without affecting legitimate receiver
-   **SVD** decomposes MIMO channel into parallel scalar channels
-   **Practical challenges:** CSI acquisition, computational complexity, hardware impairments
-   **Security applications:** Spatial nulling toward adversaries, AN injection, robust design under uncertainty

## 15. Recommended Next Steps

1.  Study convex optimization for beamforming design
2.  Learn about secrecy rate maximization in MIMO wiretap channels
3.  Explore game-theoretic approaches to adversarial beamforming
4.  Implement simulations combining beamforming with spread spectrum (anti-jamming)
5.  Read papers on robust beamforming under imperfect eavesdropper CSI


> Written with [StackEdit](https://stackedit.io/).
