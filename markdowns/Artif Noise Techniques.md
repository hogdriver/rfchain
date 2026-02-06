# TUTORIAL 3: ARTIFICIAL NOISE TECHNIQUES FOR PHYSICAL LAYER SECURITY

## Introduction

Artificial Noise (AN) is a powerful physical layer security technique that exploits spatial dimensions to degrade the eavesdropper's channel while maintaining or even enhancing the legitimate receiver's signal quality. This tutorial covers the theory, design principles, optimization methods, and practical implementation of AN techniques—essential for your research on joint anti-jamming and anti-eavesdropping.

## 1. The Artificial Noise Concept

### Basic Principle

**Traditional approach (without AN):**

-   Transmit only information signal
-   Hope eavesdropper's channel is worse than Bob's
-   Limited control over secrecy

**Artificial Noise approach:**

-   Transmit information signal + carefully designed noise
-   Design AN to interfere with Eve but NOT with Bob
-   Proactively degrade eavesdropper's channel

### Why Artificial Noise Works

**Spatial selectivity through MIMO:**

-   Multiple transmit antennas create spatial degrees of freedom (DoF)
-   Information signal uses some DoF
-   Remaining DoF used for AN that:
    -   Lies in null space of Bob's channel (doesn't affect Bob)
    -   Appears as strong interference at Eve

**Key requirement:** Need more transmit antennas than receive antennas at Bob

```
N_t > N_r (Bob)
```

This ensures null space exists for AN injection.

## 2. System Model with Artificial Noise

### Single-Antenna Bob (MISO Wiretap Channel)

**Transmit signal:**

```
x = w_s s + W_n z
```

where:

-   s = information symbol, E[|s|²] = 1
-   w_s ∈ ℂ^(N_t × 1) = information beamforming vector
-   W_n ∈ ℂ^(N_t × N_a) = AN beamforming matrix (N_a = N_t - 1 for single-antenna Bob)
-   z ∈ ℂ^(N_a × 1) = AN symbols, E[zz^H] = I

**Bob's received signal:**

```
y_B = h_B^H x + n_B
     = h_B^H w_s s + h_B^H W_n z + n_B
```

**Design constraint (perfect AN nulling at Bob):**

```
h_B^H W_n = 0
```

Therefore:

```
y_B = h_B^H w_s s + n_B
```

(AN doesn't affect Bob!)

**Eve's received signal:**

```
y_E = h_E^H x + n_E
     = h_E^H w_s s + h_E^H W_n z + n_E
```

Eve receives both signal AND artificial noise.

### Multi-Antenna Bob (MIMO Wiretap Channel)

**Bob has N_B receive antennas:**

```
y_B = H_B x + n_B
     = H_B w_s s + H_B W_n z + n_B
```

**Design constraint:**

```
H_B W_n = 0  (N_B × N_a zero matrix)
```

**Null space dimension:**

```
N_a = N_t - rank(H_B)
```

For full-rank H_B: N_a = N_t - N_B

## 3. Artificial Noise Design Approaches

### Approach 1: Null Space Projection

**Objective:** Ensure AN lies entirely in null space of H_B

**Steps:**

1.  **Compute null space of Bob's channel:**

python

```python
N_B = null_space(H_B)  # N_t × N_a matrix
```

Columns of N_B form orthonormal basis for null space.

2. **Project AN onto null space:**
```
W_n = N_B V
```

where V ∈ ℂ^(N_a × N_a) is a design matrix (often V = I).

3. **Verify nulling:**
```
H_B W_n = H_B N_B V = 0
```

**Properties:**
- Guarantees zero AN leakage to Bob
- Simple to implement
- Doesn't optimize AN effectiveness at Eve

### Approach 2: Optimized AN Beamforming

**Objective:** Maximize AN power at Eve subject to nulling at Bob

**Optimization problem:**
```
max   ||H_E W_n||²_F
W_n

subject to: H_B W_n = 0
            ||W_n||²_F ≤ P_n
```

where P_n is power allocated to AN.

**Solution approach:**
1. Project onto null space: W_n = N_B V
2. Optimize V to maximize ||H_E N_B V||²_F
3. Normalize to satisfy power constraint

**Result:** AN beamformed toward Eve within the constraint of not affecting Bob.

### Approach 3: Imperfect AN (Partial Nulling)

**Motivation:** Perfect CSI of Bob's channel may not be available

**Relaxed constraint:**
```
||H_B W_n||²_F ≤ ε
```

Allows small AN leakage to Bob in exchange for:
- Robustness to CSI errors
- More aggressive AN toward Eve

**Tradeoff:** Bob's SINR decreases slightly, but Eve's SINR decreases more.

## 4. Power Allocation Between Signal and AN

### The Power Allocation Problem

**Total power constraint:**
```
P_s + P_n = P
```

where:
- P_s = ||w_s||² = power for information signal
- P_n = ||W_n||²_F = power for artificial noise
- P = total available power

**Key question:** How to split power between signal and AN to maximize secrecy rate?

### Performance Metrics

**Bob's SNR (with perfect AN nulling):**
```
SNR_B = |h_B^H w_s|² / σ²_B = P_s |h_B^H w_s/||w_s|||² / σ²_B
```

**Eve's SINR (signal-to-interference-plus-noise ratio):**
```
SINR_E = |h_E^H w_s|² / (||h_E^H W_n||² + σ²_E)
        = P_s |h_E^H ŵ_s|² / (P_n ||h_E^H Ŵ_n||² + σ²_E)
```

where ŵ_s = w_s/||w_s||, Ŵ_n = W_n/||W_n||_F are normalized beamformers.

**Secrecy rate:**
```
R_s = [log₂(1 + SNR_B) - log₂(1 + SINR_E)]⁺
```

### Optimal Power Allocation

**Problem formulation:**
```
max     R_s(P_s, P_n)
P_s,P_n

subject to: P_s + P_n = P
            P_s ≥ 0, P_n ≥ 0
```

**Key observations:**

1. **If Eve's channel is weak:** Less AN needed
   - More power to signal (P_s ↑)
   - Less power to AN (P_n ↓)

2. **If Eve's channel is strong:** More AN needed
   - May reduce signal power (P_s ↓)
   - Increase AN power (P_n ↑)

3. **Non-monotonic behavior:** Secrecy rate not always monotonically increasing with P_s

**Solution methods:**
- Exhaustive search over α ∈ [0,1] where P_s = αP, P_n = (1-α)P
- Gradient-based optimization
- Closed-form solutions exist for special cases

### Example: Fixed Beamforming with Power Allocation

Assume:
- w_s ∝ h_B^H (MRT toward Bob)
- W_n spans null space of h_B
- Optimize only power split α
```
P_s = αP
P_n = (1-α)P
```

**Bob's rate:**
```
R_B = log₂(1 + α P ||h_B||² / σ²_B)
```

**Eve's rate:**
```
R_E = log₂(1 + α P |h_E^H h_B|² / ||h_B||² / ((1-α) P ||h_E^H N_B||² + σ²_E))
```

**Secrecy rate:**
```
R_s(α) = [R_B(α) - R_E(α)]⁺
```

Find α* ∈ [0,1] that maximizes R_s(α).

## 5. Joint Optimization of Beamforming and Power

### Problem Formulation

**General optimization:**
```
max      [log₂(1 + SNR_B) - log₂(1 + SINR_E)]⁺
w_s,W_n

subject to: h_B^H W_n = 0  (for single-antenna Bob)
            ||w_s||² + ||W_n||²_F ≤ P
```

**Challenges:**
- Non-convex objective function
- Coupled optimization variables
- Requires iterative solution

### Alternating Optimization

**Algorithm:**

1. **Initialize:** Set W_n randomly (satisfying null space constraint)

2. **Optimize w_s given W_n:**
```
   max  log₂(1 + |h_B^H w_s|² / σ²_B)
   w_s
   subject to: ||w_s||² ≤ P - ||W_n||²_F
```
   
   **Solution:** w_s ∝ h_B^H (MRT toward Bob)

3. **Optimize W_n given w_s:**
```
   min  log₂(1 + |h_E^H w_s|² / (||h_E^H W_n||² + σ²_E))
   W_n
   subject to: h_B^H W_n = 0
               ||W_n||²_F ≤ P - ||w_s||²
```
   
   Equivalently, maximize ||h_E^H W_n||² (AN power at Eve)

4. **Repeat** steps 2-3 until convergence

**Convergence:** Guaranteed to converge to local optimum (objective increases monotonically).

### Semidefinite Relaxation (SDR)

**Reformulation using rank-1 constraint:**

Define:
- W_s = w_s w_s^H (rank-1 matrix)
- W_AN = W_n W_n^H

**Original problem becomes:**
```
max  log₂(det(I + H_B W_s H_B^H / σ²_B)) - log₂(det(I + H_E W_s H_E^H / (H_E W_AN H_E^H + σ²_E I)))

subject to: Tr(W_s) + Tr(W_AN) ≤ P
            W_s ⪰ 0, rank(W_s) = 1
            W_AN ⪰ 0
            H_B W_AN H_B^H = 0
```

**Relaxation:** Drop rank constraint → Semidefinite Program (SDP)

**Solution:**
- Solve SDP to get W_s*, W_AN*
- If rank(W_s*) = 1: extract w_s, optimal solution found
- If rank(W_s*) > 1: use randomization or Gaussian approximation

## 6. Robust Artificial Noise Design

### Imperfect CSI of Bob's Channel

**Channel uncertainty model:**
```
h_B = ĥ_B + e_B
```

where:
- ĥ_B = estimated channel
- e_B = estimation error

**Bounded error:**
```
||e_B|| ≤ ε_B
```

**Worst-case robust design:**
```
max     min         R_s(w_s, W_n, h_B)
w_s,W_n  h_B: ||h_B - ĥ_B|| ≤ ε_B

subject to: ||w_s||² + ||W_n||²_F ≤ P
```

**Challenge:** Inner minimization is difficult

**Approximation:** Ensure AN leakage to Bob is bounded:
```
||h_B^H W_n||² ≤ δ  for all ||h_B - ĥ_B|| ≤ ε_B
```

**Conservative constraint:**
```
||W_n||²_F ≤ δ / (||ĥ_B|| + ε_B)²
```

### Imperfect or Unknown CSI of Eve's Channel

**Scenarios:**

1. **Statistical CSI only:** Know distribution of h_E, not instantaneous realization
2. **No CSI:** Completely unknown Eve location/channel

**Approaches:**

**1. Isotropic AN (no Eve CSI):**
```
W_n = √(P_n / N_a) · N_B
```

Spreads AN uniformly in null space—doesn't favor any direction.

**2. Statistical AN design:**

If h_E ~ CN(0, C_E) where C_E is known covariance:
```
max  E[||h_E^H W_n||²] = Tr(C_E W_n W_n^H)
W_n
```

**3. Worst-case AN design:**

Assume Eve can be in any direction:
```
max  min ||h_E^H W_n||²
W_n  ||h_E||=1
     h_E ⊥ h_B
```

Ensures minimum AN power in all directions orthogonal to Bob.

### Outage-Based Design

**Definition:** Secrecy outage occurs when R_s < R_target

**Outage probability:**
```
P_out = Pr[R_s < R_target]
```

**Design objective:** Minimize P_out or ensure P_out ≤ ε_target

**Application:** When channels are time-varying/fading, design for acceptable outage probability rather than worst-case.

## 7. Multi-User Scenarios

### Broadcast Channel with Multiple Bobs

**System:** One transmitter, K legitimate receivers (Bobs), eavesdroppers

**Signal model:**
```
x = ∑ᵢ₌₁ᴷ wᵢ sᵢ + W_n z
```

where:
- wᵢ = information beamforming vector for user i
- sᵢ = data symbol for user i
- W_n = artificial noise beamforming matrix

**Null space constraint:**
```
H_B W_n = 0
```

where H_B = [h₁^H; h₂^H; ...; hₖ^H] (channels of all K Bobs).

**Null space dimension:**
```
N_a = N_t - rank(H_B) ≥ N_t - K
```

**Challenge:** As K increases, null space shrinks → less room for AN.

**Required:** N_t > K for AN to be possible.

### Multi-Antenna Eavesdropper

**Eve has N_E receive antennas:**
```
y_E = H_E x + n_E
     = H_E w_s s + H_E W_n z + n_E
```

**Eve's capacity (with optimal combining):**
```
C_E = log₂ det(I + H_E w_s w_s^H H_E^H (H_E W_n W_n^H H_E^H + σ²_E I)⁻¹)
```

**Key insight:** Multi-antenna Eve can:
- Use spatial filtering to suppress AN
- Achieve higher capacity than single-antenna Eve

**Countermeasure:** Need rank(W_n W_n^H) ≥ N_E to effectively jam all Eve's antennas.

**Requirement:** N_t - N_B ≥ N_E

### Cooperative Jamming

**Scenario:** Multiple transmitters cooperate to create AN

**Benefits:**
- Increased spatial DoF for AN
- Can jam Eve from multiple directions
- More flexibility in power allocation

**Signal model:**
```
x_j = W_n,j z_j  (transmitter j sends only AN, j ≠ source)
```

**Design:** Coordinate AN beamformers {W_n,j} across transmitters.

## 8. AN with Other Security Techniques

### AN + Channel Coding for Secrecy

**Wiretap codes:** Error-correcting codes designed for security

**Combined approach:**
- AN degrades Eve's channel
- Wiretap coding achieves secrecy over degraded channel

**Advantage:** Coding can handle residual information leakage.

### AN + Spread Spectrum (Anti-Jamming)

**System model:**
```
x = w_s (C ⊙ s) + W_n z
```

where:
- C = spreading code
- ⊙ = element-wise multiplication (spreading)

**Benefits:**
- Spreading provides anti-jamming
- AN provides anti-eavesdropping
- Joint design optimizes both

**Challenge:** Power allocation among signal, spreading gain, and AN.

### AN + Friendly Jamming

**Friendly jammer:** Helper node that transmits interference

**System:**
- Source transmits: x_S = w_s s + W_n,S z_S
- Jammer transmits: x_J = W_n,J z_J

**Design:** Coordinate source AN and friendly jamming:
```
h_B^H W_n,S = 0  and  h_B^H W_n,J = 0
```

**Benefit:** Double the AN power at Eve.

## 9. Performance Analysis

### Secrecy Rate Analysis

**Achievable secrecy rate:**
```
R_s = [C_B - C_E]⁺
```

where:
```
C_B = log₂(1 + P_s ||h_B||² / σ²_B)

C_E = log₂(1 + P_s |h_E^H w_s/||w_s|||² / (P_n ||h_E^H W_n/||W_n||_F||² + σ²_E))
```

**Asymptotic analysis (high SNR):**

As P → ∞ with fixed α (P_s = αP, P_n = (1-α)P):
```
R_s → log₂(||h_B||² / |h_E^H ĥ_B/||h_B|||²) + log₂((1-α)||h_E^H N_B/||N_B||_F||² / α)
```

**Insight:** Secrecy rate saturates at high SNR unless AN power grows with P.

### Secrecy Outage Probability

**Definition:**
```
P_out(R_target) = Pr[R_s < R_target]
```

**For Rayleigh fading:**

Closed-form expressions possible for special cases (isotropic AN, specific power allocation).

**Example (single-antenna Bob and Eve, isotropic AN):**
```
P_out ≈ exp(-R_target · σ²_B / (P_s ||h_B||²)) · (1 - exp(-σ²_E / (P_n)))
```

(approximate expression for illustration)

### Ergodic Secrecy Rate

**Definition:** Average secrecy rate over channel realizations
```
R̄_s = E[R_s] = E[[C_B - C_E]⁺]
```

**Computation:**
- Analytical: difficult except for special cases
- Monte Carlo simulation: generate many channel realizations, average

## 10. Practical Implementation Considerations

### AN Generation

**Random AN symbols:**
```
z ~ CN(0, I)
```

**Implementation:**

-   Use Gaussian random number generator
-   Independent across antennas and time
-   Refresh every symbol period

**Alternative:** Deterministic AN (structured interference)

-   Lower peak-to-average power ratio (PAPR)
-   Easier hardware implementation
-   May sacrifice some randomness

### Null Space Computation

**Numerical stability:**

For h_B ∈ ℂ^(1 × N_t):

**Method 1: SVD**

python

```python
U, S, Vh = np.linalg.svd(h_B)
N_B = Vh[1:, :].T.conj()  # Last N_t-1 right singular vectors
```

**Method 2: QR decomposition**

python

```python
Q, R = np.linalg.qr(h_B.T.conj())
N_B = Q[:, 1:]  # Orthogonal complement
```

**Numerical issues:**

-   Ill-conditioned channels (nearly rank-deficient)
-   Finite precision arithmetic
-   Regularization may be needed

### CSI Acquisition

**Bob's channel estimation:**

-   Pilot-based channel estimation
-   Uplink-downlink reciprocity (TDD systems)
-   Feedback (FDD systems)

**Eve's channel estimation:**

-   Pilot contamination (Eve sends fake pilots)
-   Statistical estimation
-   Assume worst-case

**Update rate:**

-   AN beamformer must track channel variations
-   Typically updated every coherence time

### Hardware Impairments

**Phase noise:**

-   Affects beamforming accuracy
-   Can cause AN leakage to Bob

**I/Q imbalance:**

-   Degrades null depth
-   Requires calibration

**Power amplifier nonlinearity:**

-   Distorts AN, reduces effectiveness
-   Beamforming may need to account for nonlinearity

### Computational Complexity

**Per-symbol operations:**

-   Matrix-vector multiplication: O(N_t N_a) for W_n z
-   Total: O(N_t²) for full beamforming computation

**Optimization (less frequent):**

-   SVD: O(N_t³)
-   SDP: Polynomial but can be slow for large N_t
-   Use fast algorithms or approximations

## 11. Python Implementation Examples

### Example 1: Basic AN with Null Space Projection

python

```python
import numpy as np
from scipy.linalg import null_space

# System parameters
Nt = 6    # Transmit antennas
P = 1.0   # Total power
alpha = 0.6  # Fraction for signal

# Generate channels (Rayleigh fading)
h_B = (np.random.randn(1, Nt) + 1j*np.random.randn(1, Nt)) / np.sqrt(2)
h_E = (np.random.randn(1, Nt) + 1j*np.random.randn(1, Nt)) / np.sqrt(2)

# Information beamforming (MRT)
w_s = np.sqrt(alpha * P) * h_B.conj().T / np.linalg.norm(h_B)

# Null space of Bob's channel
N_B = null_space(h_B)

# Artificial noise beamforming (isotropic in null space)
z = (np.random.randn(N_B.shape[1], 1) + 1j*np.random.randn(N_B.shape[1], 1)) / np.sqrt(2)
W_n = np.sqrt((1-alpha) * P / N_B.shape[1]) * N_B

# Verify perfect nulling
print(f"AN leakage to Bob: {np.abs(h_B @ W_n @ z)[0,0]:.8f}")

# Bob's SNR
SNR_B = np.abs(h_B @ w_s)[0,0]**2 / 0.1  # Assume sigma2_B = 0.1
print(f"Bob's SNR: {10*np.log10(SNR_B):.2f} dB")

# Eve's SINR
signal_E = np.abs(h_E @ w_s)[0,0]**2
AN_E = np.linalg.norm(h_E @ W_n @ z)**2
SINR_E = signal_E / (AN_E + 0.1)  # Assume sigma2_E = 0.1
print(f"Eve's SINR: {10*np.log10(SINR_E):.2f} dB")

# Secrecy rate
R_s = np.maximum(np.log2(1 + SNR_B) - np.log2(1 + SINR_E), 0)
print(f"Secrecy rate: {R_s:.3f} bits/s/Hz")
```

### Example 2: Power Allocation Optimization

python

```python
import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import null_space

# System setup
Nt = 8
h_B = (np.random.randn(1, Nt) + 1j*np.random.randn(1, Nt)) / np.sqrt(2)
h_E = (np.random.randn(1, Nt) + 1j*np.random.randn(1, Nt)) / np.sqrt(2)
sigma2_B = 0.1
sigma2_E = 0.1
P = 1.0

# Null space
N_B = null_space(h_B)

# Sweep power allocation
alphas = np.linspace(0, 1, 100)
R_s_values = []

for alpha in alphas:
    # Information beamforming
    w_s = np.sqrt(alpha * P) * h_B.conj().T / np.linalg.norm(h_B)
    
    # AN beamforming
    P_n = (1 - alpha) * P
    
    # Bob's rate (no AN interference)
    SNR_B = np.abs(h_B @ w_s)[0,0]**2 / sigma2_B
    R_B = np.log2(1 + SNR_B)
    
    # Eve's rate (with AN interference)
    signal_E = np.abs(h_E @ w_s)[0,0]**2
    # Average AN power at Eve (assuming isotropic AN)
    AN_E_avg = P_n * np.linalg.norm(h_E @ N_B)**2 / N_B.shape[1]
    SINR_E = signal_E / (AN_E_avg + sigma2_E)
    R_E = np.log2(1 + SINR_E)
    
    # Secrecy rate
    R_s = max(R_B - R_E, 0)
    R_s_values.append(R_s)

# Find optimal
optimal_idx = np.argmax(R_s_values)
optimal_alpha = alphas[optimal_idx]

plt.figure(figsize=(10, 6))
plt.plot(alphas, R_s_values, linewidth=2)
plt.axvline(optimal_alpha, color='r', linestyle='--', 
            label=f'Optimal α = {optimal_alpha:.2f}')
plt.xlabel('Power Allocation Parameter α')
plt.ylabel('Secrecy Rate (bits/s/Hz)')
plt.title('Secrecy Rate vs. Power Allocation')
plt.legend()
plt.grid(True)
plt.show()

print(f"Optimal power allocation: α = {optimal_alpha:.3f}")
print(f"Maximum secrecy rate: {max(R_s_values):.3f} bits/s/Hz")
```

### Example 3: Robust AN with CSI Uncertainty

python

```python
import numpy as np
from scipy.linalg import null_space

# System parameters
Nt = 6
P = 1.0
alpha = 0.7
epsilon = 0.2  # CSI error bound

# True Bob's channel
h_B_true = (np.random.randn(1, Nt) + 1j*np.random.randn(1, Nt)) / np.sqrt(2)

# Estimated Bob's channel (with error)
error = epsilon * (np.random.randn(1, Nt) + 1j*np.random.randn(1, Nt)) / np.sqrt(2)
error = error / np.linalg.norm(error) * epsilon  # Normalize to bounded error
h_B_est = h_B_true + error

# Eve's channel
h_E = (np.random.randn(1, Nt) + 1j*np.random.randn(1, Nt)) / np.sqrt(2)

# Design based on estimated channel
w_s = np.sqrt(alpha * P) * h_B_est.conj().T / np.linalg.norm(h_B_est)
N_B_est = null_space(h_B_est)

# Conservative AN design (reduced power to limit leakage)
safety_factor = 0.8  # Reduce AN power for robustness
P_n = (1 - alpha) * P * safety_factor
z = (np.random.randn(N_B_est.shape[1], 1) + 1j*np.random.randn(N_B_est.shape[1], 1)) / np.sqrt(2)
W_n = np.sqrt(P_n / N_B_est.shape[1]) * N_B_est

# Evaluate on TRUE channel
AN_leakage_Bob = np.linalg.norm(h_B_true @ W_n @ z)**2
signal_Bob = np.abs(h_B_true @ w_s)[0,0]**2

print(f"AN leakage to Bob (true channel): {AN_leakage_Bob:.6f}")
print(f"Signal power at Bob: {signal_Bob:.6f}")
print(f"AN-to-Signal ratio at Bob: {AN_leakage_Bob/signal_Bob:.6f}")

# With perfect CSI, this should be ~0
# With imperfect CSI, there will be some leakage
```

### Example 4: Multi-User AN

python

```python
import numpy as np
from scipy.linalg import null_space

# System parameters
Nt = 10   # Transmit antennas
K = 3     # Number of users
P = 1.0

# User channels (each user has 1 antenna)
H_B = (np.random.randn(K, Nt) + 1j*np.random.randn(K, Nt)) / np.sqrt(2)

# Eavesdropper channel
h_E = (np.random.randn(1, Nt) + 1j*np.random.randn(1, Nt)) / np.sqrt(2)

# Check null space dimension
null_dim = Nt - np.linalg.matrix_rank(H_B)
print(f"Null space dimension: {null_dim}")

if null_dim > 0:
    # Compute null space of all user channels
    N_B = null_space(H_B)
    
    # Allocate power
    P_signal = 0.7 * P
    P_AN = 0.3 * P
    
    # ZF precoding for users
    W_zf = H_B.conj().T @ np.linalg.inv(H_B @ H_B.conj().T)
    # Normalize
    power_per_user = np.sum(np.abs(W_zf)**2, axis=0)
    W_zf = W_zf @ np.diag(np.sqrt(P_signal/K / power_per_user))
    
    # AN in null space
    z = (np.random.randn(N_B.shape[1], 1) + 1j*np.random.randn(N_B.shape[1], 1)) / np.sqrt(2)
    W_n = np.sqrt(P_AN / N_B.shape[1]) * N_B
    
    # Verify users don't receive AN
    AN_at_users = H_B @ W_n @ z
    print(f"AN leakage to users: {np.abs(AN_at_users).flatten()}")
    
    # AN power at Eve
    AN_at_Eve = np.linalg.norm(h_E @ W_n @ z)**2
    print(f"AN power at Eve: {AN_at_Eve:.4f}")
else:
    print("Not enough antennas for AN with this many users!")
```

### Example 5: Secrecy Outage Simulation

python

```python
import numpy as np
from scipy.linalg import null_space
import matplotlib.pyplot as plt

# System parameters
Nt = 6
P = 1.0
alpha = 0.6
sigma2 = 0.1
num_trials = 10000

# Target secrecy rate
R_target = 1.0  # bits/s/Hz

outage_count = 0
secrecy_rates = []

for trial in range(num_trials):
    # Generate random channels
    h_B = (np.random.randn(1, Nt) + 1j*np.random.randn(1, Nt)) / np.sqrt(2)
    h_E = (np.random.randn(1, Nt) + 1j*np.random.randn(1, Nt)) / np.sqrt(2)
    
    # Beamforming
    w_s = np.sqrt(alpha * P) * h_B.conj().T / np.linalg.norm(h_B)
    N_B = null_space(h_B)
    
    # AN
    z = (np.random.randn(N_B.shape[1], 1) + 1j*np.random.randn(N_B.shape[1], 1)) / np.sqrt(2)
    W_n = np.sqrt((1-alpha) * P / N_B.shape[1]) * N_B
    
    # Rates
    SNR_B = np.abs(h_B @ w_s)[0,0]**2 / sigma2
    R_B = np.log2(1 + SNR_B)
    
    signal_E = np.abs(h_E @ w_s)[0,0]**2
    AN_E = np.linalg.norm(h_E @ W_n @ z)**2
    SINR_E = signal_E / (AN_E + sigma2)
    R_E = np.log2(1 + SINR_E)
    
    R_s = max(R_B - R_E, 0)
    secrecy_rates.append(R_s)
    
    if R_s < R_target:
        outage_count += 1

outage_prob = outage_count / num_trials
print(f"Secrecy outage probability: {outage_prob:.4f}")
print(f"Average secrecy rate: {np.mean(secrecy_rates):.3f} bits/s/Hz")

# Plot CDF
sorted_rates = np.sort(secrecy_rates)
cdf = np.arange(1, len(sorted_rates)+1) / len(sorted_rates)

plt.figure(figsize=(10, 6))
plt.plot(sorted_rates, cdf, linewidth=2)
plt.axvline(R_target, color='r', linestyle='--', label=f'Target rate = {R_target}')
plt.xlabel('Secrecy Rate (bits/s/Hz)')
plt.ylabel('CDF')
plt.title('Secrecy Rate Distribution with Artificial Noise')
plt.legend()
plt.grid(True)
plt.show()
```

## 12. Advanced Topics

### Directional AN

**Concept:** Instead of isotropic AN, direct more AN power toward likely eavesdropper locations

**Design:**
```
W_n = N_B V
```

where V is optimized to focus AN in certain spatial directions.

**Application:** When partial information about Eve's location is available (e.g., sector-based).

### AN with Energy Harvesting

**Scenario:** Legitimate receivers can harvest energy from AN

**Model:**

-   Bob splits received signal:
    -   Fraction for information decoding
    -   Fraction for energy harvesting

**AN design:** Balance secrecy and energy transfer.

### Reconfigurable Intelligent Surfaces (RIS) with AN

**System:** RIS helps shape propagation environment

**Benefits:**

-   Enhance signal at Bob
-   Suppress signal at Eve
-   Redirect AN toward Eve

**Joint optimization:** RIS phase shifts + AN beamforming.

### AN in Millimeter Wave (mmWave) Systems

**Characteristics:**

-   Highly directional beams
-   Large antenna arrays
-   Sparse scattering

**Challenges:**

-   Narrow beams reduce spatial DoF for AN
-   Beam alignment requirements

**Opportunity:**

-   Massive MIMO provides large null space
-   Inherent directionality provides spatial security

## 13. Comparison with Other PLS Techniques

Technique

Advantages

Disadvantages

When to Use

**Artificial Noise**

Proactive degradation of Eve; works without Eve CSI

Requires extra antennas; consumes power

Multi-antenna systems, moderate SNR

**Beamforming only**

Simple; no power wasted on AN

Requires Eve CSI; limited secrecy

Strong directional channels

**Friendly Jamming**

External helper; more flexibility

Coordination overhead; extra hardware

Cooperative networks

**Spread Spectrum**

Anti-jamming + some security

Processing gain limited; Eve can despread if she knows code

Jamming-prone environments

**Best approach:** Often combine techniques (e.g., AN + beamforming + coding).

## 14. Open Research Problems

1.  **Joint AN and anti-jamming:** How to design AN that helps against both Eve and jammer?
2.  **Machine learning for AN:** Can RL learn optimal AN strategies without explicit channel models?
3.  **AN with reconfigurable antennas:** Dynamic antenna patterns for adaptive AN.
4.  **Quantum-enhanced AN:** Using quantum noise sources for AN generation.
5.  **AN in beyond-5G systems:** Integration with NOMA, OTFS, and other emerging techniques.

## 15. Key Takeaways

-   **Artificial noise exploits spatial DoF** to degrade eavesdropper without affecting legitimate receiver
-   **Null space projection** ensures AN doesn't leak to Bob
-   **Power allocation** between signal and AN is critical for maximizing secrecy rate
-   **Robust design** necessary when Bob's CSI is imperfect
-   **Multi-user scenarios** reduce available DoF for AN
-   **Implementation challenges:** CSI acquisition, computational complexity, hardware impairments
-   **Combined with anti-jamming:** Creates interesting tradeoffs and optimization problems

## 16. Recommended Next Steps

1.  Implement and simulate basic AN scenarios in Python
2.  Study convex optimization for joint beamforming-AN design
3.  Read papers on robust AN under CSI uncertainty
4.  Explore game-theoretic approaches (transmitter vs. jammer vs. eavesdropper)
5.  Investigate deep learning for adaptive AN in time-varying environments
6.  Consider integration with spread spectrum for dual anti-jam/anti-eavesdrop capability


> Written with [StackEdit](https://stackedit.io/).
