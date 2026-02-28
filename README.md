# PQC Security Margin Calculator

A Python tool for computing and analysing **security margins** of lattice-based post-quantum cryptographic (PQC) schemes standardised by NIST (FIPS 203 / FIPS 204).

The tool implements the security margin model developed in the accompanying research work and provides quantitative estimates of how much headroom PQC parameter sets retain under varying assumptions about quantum computational progress.

---

## Mathematical Model

The **security margin** for a parameter set *S* is defined as:

```
M(S) = τ̂(S) − λ(S)
```

where:

| Symbol | Meaning |
|--------|---------|
| `λ(S)` | Classical security level in bits, estimated by the NIST Lattice Estimator |
| `τ̂(S)` | Projected quantum attack cost in bits, computed via SALSA regression: `τ̂(S) = C · log₂(n)` |
| `C` | Regression constant that encodes assumptions about quantum progress |
| `n` | Lattice dimension of the scheme |

Three scenarios capture the range of uncertainty:

| Scenario | Constant `C` | Interpretation |
|----------|:---:|---|
| Conservative (C1) | 47 | Fastest projected quantum progress |
| Moderate (C2) | 52 | Baseline estimate |
| Optimistic (C3) | 57 | Slowest projected quantum progress |

The SALSA regression is calibrated on empirical data from the SALSA (Systematic Analysis of Lattice Security Assumptions) framework at dimensions `n ∈ {128, 350, 512, 1024}`.

---

## Evaluated Schemes

| Scheme | NIST Category | Lattice Dimension `n` | `λ(S)` (bits) |
|--------|:---:|:---:|:---:|
| ML-KEM-512 | 1 | 512 | 118 |
| ML-KEM-768 | 3 | 768 | 182 |
| ML-KEM-1024 | 5 | 1024 | 256 |
| ML-DSA-44 | 2 | 1024 | 128 |
| ML-DSA-65 | 3 | 1280 | 192 |
| ML-DSA-87 | 5 | 1536 | 256 |

---

## Project Structure

```
pqc-security-margin/
├── security_margin.py   # Core: M(S) computation for all parameter sets
├── salsa_regression.py  # OLS regression on SALSA empirical data
├── sensitivity.py       # Sensitivity analysis: C ∈ [47, 57], K-fold improvement
├── visualize.py         # Publication-quality matplotlib plots
├── taxonomy.py          # Quantum threat taxonomy and migration decision matrix
├── main.py              # CLI interface (click-based)
├── requirements.txt     # Python dependencies
└── README.md
```

### Module Descriptions

**`security_margin.py`** -- Core computation engine. Stores NIST parameter sets, computes `τ̂(S)` via the SALSA model, and returns the margin `M(S)` with a qualitative interpretation (high / moderate / low / negative).

**`salsa_regression.py`** -- Fits linear and quadratic OLS models to the SALSA data in the `log₂(n)` domain. Reports coefficients, R², residual standard deviation, and 95% confidence intervals. Supports extrapolation to PQC-relevant dimensions.

**`sensitivity.py`** -- Analyses how `M(S)` responds to: (1) variation of the regression constant `C` over `[47, 57]`, and (2) a hypothetical `K`-fold improvement in lattice attacks (`K = 10⁰, 10¹, …, 10⁶`). Computes the critical `C*` and `K*` at which the margin vanishes.

**`visualize.py`** -- Generates four publication-quality figures using matplotlib:
1. Grouped bar chart of security margins across schemes and scenarios
2. Sensitivity of `M(S)` to the constant `C`
3. Margin degradation under `K`-fold attack improvement
4. SALSA regression fits with extrapolation and confidence bands

**`taxonomy.py`** -- Structured classification of quantum threats (algorithmic, implementation, cryptanalytic, hardware) with a migration decision matrix mapping (threat level, margin category) to recommended actions.

**`main.py`** -- Command-line interface built with `click`. Exposes all analysis capabilities through subcommands.

---

## Installation

```bash
git clone https://github.com/Shtomuch/pqc-security-margin.git
cd pqc-security-margin
pip install -r requirements.txt
```

Requires Python 3.10 or later.

---

## Usage

### Compute security margin for a single scheme

```bash
python main.py margin --scheme ML-KEM-512 --scenario C2
```

Output:
```
Scheme:          ML-KEM-512
Scenario:        C2 (C=52)
lambda(S):       118.0 bits
tau_hat(S):      468.0 bits
Margin M(S):     +350.0 bits
Assessment:      High security margin -- safe for long-term deployment
```

### Evaluate all schemes

```bash
python main.py margin --all-schemes --scenario C1
```

### SALSA regression analysis

```bash
python main.py regression --extrapolate
```

### Sensitivity analysis

```bash
python main.py sensitivity
python main.py sensitivity --scheme ML-KEM-768
```

### Generate plots

```bash
python main.py visualize --output-dir figures/
```

Produces four PNG files in the specified directory at 300 DPI.

### Quantum threat taxonomy

```bash
python main.py taxonomy --matrix
```

---

## Output Interpretation

| Margin `M(S)` | Category | Recommendation |
|:-:|---|---|
| > 40 bits | High | Safe for long-term deployment |
| 20 -- 40 bits | Moderate | Monitor quantum computing developments |
| 0 -- 20 bits | Low | Begin migration planning |
| < 0 bits | Negative | Immediate action required |

---

## References

1. NIST FIPS 203 -- Module-Lattice-Based Key-Encapsulation Mechanism Standard (ML-KEM)
2. NIST FIPS 204 -- Module-Lattice-Based Digital Signature Standard (ML-DSA)
3. Albrecht, M.R. et al. -- Estimate all the {LWE, NTRU} schemes! (Lattice Estimator)
4. Ducas, L. -- SALSA: Systematic Analysis of Lattice Security Assumptions

---

## License

This software is provided for academic and research purposes. See the accompanying research work for citation details.
