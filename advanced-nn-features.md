Below is a **specific, production-grade architecture choice** for both the **VAE** and the **(conditional) GAN**, designed to work with:

* Mixed **binary subsets**
* **Permutations / ordered sets**
* **Multiple coupled sets**
* **Continuous variables**
* **Any black-box objective**

No convexity, no gradients of the objective, only gradients of the neural nets themselves.

---

# 1. Unified Decision Vector Representation

All architectures assume the same **flat encoded vector**:

[
v = \texttt{encode_decision}(d) \in \mathbb{R}^D
]

Composed as:

| Component              | Encoding                         |
| ---------------------- | -------------------------------- |
| Unordered subset (z)   | Binary vector                    |
| Ordered set (size (k)) | (n \times k) one-hot (flattened) |
| Multi-sets             | Concatenated blocks              |
| Continuous (x)         | Direct real values               |

After concatenation:

[
D = D_{\text{subsets}} + D_{\text{permutations}} + p
]

All models below operate on this (v \in \mathbb{R}^D).

---

# 2. VAE Architecture (Decision Manifold Model)

This VAE is used for:

* Latent-space optimization
* Smooth exploration
* Interpolation between valid solutions

## 2.1 Latent Model Type

We use a:

✅ **β-VAE with diagonal Gaussian latent**
[
q_\phi(u \mid v) = \mathcal{N}(\mu_\phi(v), \operatorname{diag}(\sigma_\phi^2(v)))
]

Latent dimension:
[
k \in [16, 128]
]
(depends on decision dimension (D))

---

## 2.2 Encoder Network

**Architecture (MLP, fully generic):**

```
Input v ∈ ℝ^D
→ Linear(D, 512) → GELU → BatchNorm
→ Linear(512, 256) → GELU → BatchNorm
→ Linear(256, 128) → GELU
→ Two heads:
   μ(v) ∈ ℝ^k
   log σ²(v) ∈ ℝ^k
```

This works for **any decision encoding dimensionality**.

---

## 2.3 Decoder Network (Mixed-Type Output)

```
Input u ∈ ℝ^k
→ Linear(k, 128) → GELU
→ Linear(128, 256) → GELU
→ Linear(256, 512) → GELU
→ Linear(512, D) → Raw output h ∈ ℝ^D
```

Then apply **typed decoding heads**:

| Variable Type      | Activation           | Post-processing                |
| ------------------ | -------------------- | ------------------------------ |
| Binary bits        | Sigmoid              | Threshold + cardinality repair |
| Permutation blocks | Softmax per position | Argmax + assignment repair     |
| Continuous         | Identity or Tanh     | Clamp to bounds                |

---

## 2.4 VAE Loss

[
\mathcal{L}
===========

\underbrace{|v - \hat v|*2^2}*{\text{reconstruction}}
+
\beta , \underbrace{\mathrm{KL}!\left(q(u|v),|,\mathcal{N}(0,I)\right)}_{\text{regularization}}
]

* (\beta \in [0.1, 5])
* Reconstruction loss is **typed per segment**:

  * BCE for binary
  * Categorical CE for permutations
  * MSE for continuous

---

## 2.5 When to Use the VAE

Use the VAE when:

✅ Decision dimension (D) is large
✅ There are strong dependencies between variables
✅ You want **latent CMA-ES / latent GA**
✅ You have at least a few thousand evaluated solutions

---

# 3. GAN / Conditional GAN Architecture (Elite Generator)

The GAN is used for:

* **Sharp elite proposal generation**
* Targeted sampling of **Pareto regions**
* Diversity repair when GA collapses

---

## 3.1 GAN Type

✅ **Wasserstein GAN with Gradient Penalty (WGAN-GP)**
✅ Optional **Conditional WGAN (cWGAN-GP)**

This gives:

* Stable training
* No mode collapse
* Good for mixed continuous/discrete outputs

---

## 3.2 Conditioning Vector (y)

Depends on problem:

| Conditioning          | Dimension |
| --------------------- | --------- |
| Single-objective bins | 4–10      |
| Pareto front index    | 5–20      |
| Budget tier           | 3–5       |
| Subset size           | 1         |

Final conditioning vector:
[
y \in \mathbb{R}^c
]

---

## 3.3 Generator Network

Input:
[
(\xi, y), \quad \xi \sim \mathcal{N}(0,I), \ \xi \in \mathbb{R}^{64}
]

Architecture:

```
[ξ | y] ∈ ℝ^{64 + c}
→ Linear(64+c, 256) → LeakyReLU
→ Linear(256, 512) → LeakyReLU
→ Linear(512, 512) → LeakyReLU
→ Linear(512, D) → Raw output h ∈ ℝ^D
```

Then same **typed decoding as in the VAE decoder**.

---

## 3.4 Discriminator / Critic Network

Input:
[
(v, y) \in \mathbb{R}^{D + c}
]

Architecture:

```
→ Linear(D+c, 512) → LeakyReLU
→ Linear(512, 256) → LeakyReLU
→ Linear(256, 128) → LeakyReLU
→ Linear(128, 1) → Scalar score
```

Loss: **Wasserstein + Gradient Penalty**

---

## 3.5 GAN Training Data

You train on:

[
\mathcal{S}_{\text{elite}} = {v_i : F(v_i) \in \text{top } 10–20%}
]

With conditioning labels:

* Objective quantiles
* Pareto front IDs
* Budget tiers

---

## 3.6 When to Use GAN vs VAE

| Use Case                       | VAE | GAN |
| ------------------------------ | --- | --- |
| Dimensionality reduction       | ✅   | ❌   |
| Latent continuous optimization | ✅   | ❌   |
| Sharp elite sampling           | ❌   | ✅   |
| Pareto front targeting         | ❌   | ✅   |
| Smooth interpolation           | ✅   | ❌   |

✅ **Best practice: use both.**

---

# 4. Surrogate Model Architectures

Surrogates operate on the same (v \in \mathbb{R}^D).

Recommended defaults:

| Model            | Use                       |
| ---------------- | ------------------------- |
| Random Forest    | First choice, most robust |
| XGBoost          | When (D < 500)            |
| Deep MLP         | When data > 50k           |
| Gaussian Process | Small (D < 50) only       |

Surrogate predicts:

[
\hat F(v), \quad \hat \sigma(v)
]

Used for:

* Pre-screening
* Acquisition functions
* Candidate ranking only

---

# 5. How These Three Work Together (Correct Stack)

```
          ┌────────────┐
          │   GA / MOO │  ← correctness + global search
          └─────┬──────┘
                │
   ┌────────────▼────────────┐
   │                         │
┌──▼───┐               ┌────▼────┐
│  VAE │               │   GAN   │
│latent│               │ elite   │
│search│               │ samples │
└──┬───┘               └────┬────┘
   │                         │
   └────────────┬────────────┘
                ▼
         Candidate Pool
                ▼
          ┌────────────┐
          │  Surrogate │ (rank/filter only)
          └─────┬──────┘
                ▼
        True Black-Box Evaluation
```

✅ GA guarantees correctness
✅ VAE gives *smooth compressed exploration*
✅ GAN gives *sharp elite exploitation*
✅ Surrogate gives *speed only*

---

# 6. Final Recommendation (Default TrainSelPy Stack)

If you want a **single default configuration** that is safe and powerful:

* **VAE:** β-VAE, latent dim 32–64, typed reconstruction losses
* **GAN:** Conditional WGAN-GP with 64-dim noise
* **Surrogate:** Random Forest → XGBoost → MLP fallback
* **Outer loop:** NSGA-II (even for single objective, it works fine)

