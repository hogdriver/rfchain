# TUTORIAL 1: WIRETAP CHANNEL FUNDAMENTALS AND SECRECY CAPACITY

## Introduction

This tutorial provides foundational knowledge in physical layer security, focusing on wiretap channels and secrecy capacity - essential concepts for your PhD research on joint anti-jamming and anti-eavesdropping techniques.

## 1. The Core Problem

Imagine Alice wants to send a confidential message to Bob, but Eve is eavesdropping on the communication. The **wiretap channel** asks: How much information can Alice send to Bob while keeping Eve completely ignorant of the message, using only coding techniques (no cryptography)?

This was first solved by Aaron Wyner in 1975, creating the foundation for physical layer security.

## 2. The Classical Wiretap Channel Model

**Setup:**

-   Alice (transmitter) sends signal X
-   Bob (legitimate receiver) receives Y through a "main channel" with capacity C_B
-   Eve (eavesdropper) receives Z through a "wiretap channel" with capacity C_E
-   Both channels are noisy, but we assume Alice knows their statistics

**Key Insight:** If Bob's channel is better than Eve's (C_B > C_E), Alice can communicate reliably with Bob while ensuring Eve learns _provably nothing_ about the message.

## 3. Secrecy Capacity: The Main Result

**Secrecy Capacity (C_s):**

```
C_s = max[I(X;Y) - I(X;Z)]
```

where the maximum is over all input distributions p(X).

**In simpler terms:**

-   C_s = C_B - C_E (for degraded wiretap channels where Eve's channel is a degraded version of Bob's)

**What this means:**

-   Alice can transmit at rate R_s bits per channel use with perfect secrecy if R_s < C_s
-   "Perfect secrecy" means Eve's mutual information with the message approaches zero as codeword length increases
-   If C_E ≥ C_B, then C_s = 0 (no secure communication possible)

## 4. How It Works: Intuition

**The Coding Scheme:**

1.  **Message splitting:** Alice's k-bit message is encoded into an n-symbol codeword (n >> k)
2.  **Deliberate ambiguity:** The encoding adds controlled randomness that looks like noise to Eve but can be decoded by Bob
3.  **Rate advantage exploitation:** Bob's better channel allows him to distinguish the true message from the randomness, while Eve cannot

**Simple Example:**

-   Suppose Bob's channel is error-free (C_B = log₂|X|)
-   Eve's channel is a binary symmetric channel with error probability 0.5 (completely random, C_E = 0)
-   Then C_s = C_B - 0 = C_B (perfect secrecy at full rate!)

## 5. Gaussian Wiretap Channel

For practical wireless systems, the most important case is the **Gaussian wiretap channel**:

**Model:**

-   Y = X + N_B (Bob receives signal plus Gaussian noise)
-   Z = X + N_E (Eve receives signal plus Gaussian noise)

**Secrecy Capacity:**

```
C_s = (1/2)log₂(1 + SNR_B) - (1/2)log₂(1 + SNR_E)
```

where:

-   SNR_B = P/σ²_B (signal-to-noise ratio at Bob)
-   SNR_E = P/σ²_E (signal-to-noise ratio at Eve)
-   P is transmit power

**Key observations:**

-   Secrecy capacity increases when Bob's SNR improves or Eve's SNR degrades
-   If SNR_E ≥ SNR_B, then C_s = 0
-   Unlike traditional Shannon capacity, you can't always achieve arbitrary rates by increasing power alone—Eve benefits too!

## 6. Important Concepts

**Perfect Secrecy vs. Weak Secrecy vs. Strong Secrecy:**

-   **Perfect secrecy:** I(M;Z^n) = 0 (Shannon's original definition—impractical)
-   **Weak secrecy:** I(M;Z^n)/n → 0 as n → ∞ (information leakage rate vanishes)
-   **Strong secrecy:** I(M;Z^n) → 0 as n → ∞ (total information leakage vanishes)

Most modern work uses strong secrecy as the standard.

**Equivocation:** The conditional entropy H(M|Z^n) measures Eve's uncertainty about message M given her observations. Perfect secrecy requires H(M|Z^n) = H(M).

**Achievability vs. Converse:**

-   **Achievability:** Proving a rate R_s is achievable means constructing a coding scheme that achieves it
-   **Converse:** Proving no rate above R_s is achievable establishes the capacity upper bound

## 7. Extensions Relevant to Your Research

### MIMO Wiretap Channel

Multiple antennas at transmitter/receiver allow:

-   Beamforming to direct signal toward Bob, away from Eve
-   Creating artificial noise in Eve's direction
-   Spatial multiplexing for higher secrecy rates

Secrecy capacity becomes much more complex, depending on channel matrices H_B and H_E.

### Broadcast Channel with Confidential Messages

Multiple legitimate receivers, some messages must be kept secret from other legitimate users. This models cellular networks where users shouldn't decode each other's data.

### Compound Wiretap Channel

Eve's channel state is unknown but belongs to a known set. You must design codes that work against the worst-case eavesdropper—critical for practical scenarios.

### Fading Wiretap Channels

Channel gains vary over time. Key question: Does fading help or hurt secrecy? (Answer: Usually helps due to diversity)

## 8. Connection to Anti-Jamming

**The tension:**

-   Anti-eavesdropping wants to minimize I(X;Z)
-   Anti-jamming wants to maximize I(X;Y) despite interference
-   When you add jamming, the model becomes:
    -   Y = X + J + N_B (Bob's channel with jammer J)
    -   Z = X + N_E (Eve's channel, possibly without jamming)

The secrecy capacity becomes:

```
C_s = max[I(X;Y|J) - I(X;Z)]
```

subject to power constraints and jamming strategies.

**Key challenges:**

-   Power allocation: How much for signal vs. anti-jamming measures?
-   If you use spread spectrum for anti-jamming, does it hurt or help secrecy?
-   Can you exploit jamming to confuse Eve? (Friendly jamming)

## 9. Foundational Papers to Read

**Must-read classics:**

1.  **A.D. Wyner (1975)** - "The Wire-Tap Channel" (Bell System Technical Journal)
    -   The original paper, surprisingly readable
2.  **I. Csiszár and J. Körner (1978)** - "Broadcast Channels with Confidential Messages" (IEEE Trans. Information Theory)
    -   Generalizes to multiple receivers
3.  **S.K. Leung-Yan-Cheong and M.E. Hellman (1978)** - "The Gaussian Wire-Tap Channel" (IEEE Trans. Information Theory)
    -   Derives the Gaussian case formula

**Modern tutorials:**

4.  **Y. Liang, H.V. Poor, S. Shamai (2008)** - "Information Theoretic Security" (Foundations and Trends in Communications and Information Theory)
    -   Comprehensive modern survey, ~150 pages
5.  **M. Bloch and J. Barros (2011)** - "Physical-Layer Security: From Information Theory to Security Engineering" (Cambridge University Press)
    -   Excellent textbook bridging theory and practice

## 10. Python Simulation Exercise

Here's a starter simulation to build intuition:

python

```python
import numpy as np
import matplotlib.pyplot as plt

# Parameters
P = 1.0  # Transmit power
sigma_B = np.linspace(0.01, 2, 100)  # Bob's noise std dev (varying)
sigma_E = 0.5  # Eve's noise std dev (fixed)

# SNRs
SNR_B = P / sigma_B**2
SNR_E = P / sigma_E**2

# Secrecy capacity
C_B = 0.5 * np.log2(1 + SNR_B)
C_E = 0.5 * np.log2(1 + SNR_E)
C_s = np.maximum(C_B - C_E, 0)  # Can't be negative

plt.figure(figsize=(10, 6))
plt.plot(sigma_B, C_B, label="Bob's Capacity", linewidth=2)
plt.axhline(C_E, color='r', linestyle='--', label="Eve's Capacity")
plt.plot(sigma_B, C_s, label="Secrecy Capacity", linewidth=2, color='g')
plt.xlabel("Bob's Noise Level (σ_B)")
plt.ylabel("Capacity (bits/channel use)")
plt.title("Gaussian Wiretap Channel Secrecy Capacity")
plt.legend()
plt.grid(True)
plt.show()
```

**What to observe:**

-   Secrecy capacity is zero when Bob's noise is high (poor channel)
-   There's a threshold beyond which secure communication becomes possible
-   Unlike regular capacity, you can't overcome Eve by just increasing power if she has a comparable channel

## 11. Next Steps

Once you understand wiretap channels, you should:

1.  Study **MIMO wiretap channels** (spatial dimension adds huge complexity and opportunity)
2.  Learn **artificial noise beamforming** (practical technique for MIMO secrecy)
3.  Explore **secrecy outage** (when Eve's channel is random/unknown)
4.  Read about **physical layer key generation** (alternative approach using channel reciprocity)

## 12. Key Takeaways

-   **Secrecy capacity** quantifies the maximum rate of provably secure communication using only physical layer coding
-   **Main principle:** Exploit the difference in channel quality between legitimate receiver and eavesdropper
-   **Gaussian case:** C_s = (1/2)log₂(1 + SNR_B) - (1/2)log₂(1 + SNR_E)
-   **Extension to anti-jamming:** Joint optimization of secrecy and jamming resistance creates complex tradeoffs
-   **Foundation:** Understanding this theory is essential before tackling MIMO, beamforming, and artificial noise techniques# Welcome to StackEdit!

