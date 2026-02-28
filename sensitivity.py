"""Sensitivity analysis for PQC security margins.

Investigates how the security margin M(S) = tau_hat(S) - lambda(S) responds
to variation in two key parameters:

1. **SALSA regression constant C** -- governs the slope of the estimated
   quantum attack cost tau_hat(S) = C * log2(n).  Sweeping C over a plausible
   range reveals the critical threshold C* at which the margin vanishes for
   each scheme.

2. **ML attack improvement factor K** -- models a hypothetical K-fold
   speedup in lattice attacks (e.g. via machine-learning-assisted sieving).
   The effective margin becomes M_eff(S) = tau_hat(S) - lambda(S) - log2(K).
   The critical improvement K* where M_eff = 0 quantifies each scheme's
   resilience to algorithmic breakthroughs.

Both analyses are parameterised over the three ML-KEM parameter sets
(FIPS 203) with classical security estimates from the NIST Lattice Estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

__all__ = [
    "SensitivityResult",
    "analyze_c_sensitivity",
    "analyze_k_sensitivity",
    "find_critical_c",
    "find_critical_k",
    "run_full_sensitivity_analysis",
]

# ---------------------------------------------------------------------------
# ML-KEM reference data
# ---------------------------------------------------------------------------

_ML_KEM_SCHEMES: list[tuple[str, int, float]] = [
    ("ML-KEM-512", 512, 118.0),
    ("ML-KEM-768", 768, 182.0),
    ("ML-KEM-1024", 1024, 256.0),
]


# ---------------------------------------------------------------------------
# Result container
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class SensitivityResult:
    """Outcome of a single-parameter sensitivity sweep for one scheme.

    Attributes:
        scheme_name: Canonical scheme identifier (e.g. ``ML-KEM-768``).
        parameter_name: Swept parameter (``"C"`` or ``"K"``).
        parameter_values: Evaluated parameter values.
        margin_values: Corresponding security margins M(S) in bits.
        critical_value: Parameter value at which M(S) = 0, or ``None`` if the
            margin remains positive (or negative) over the entire sweep.
    """

    scheme_name: str
    parameter_name: str
    parameter_values: list[float] = field(repr=False)
    margin_values: list[float] = field(repr=False)
    critical_value: float | None

    def __repr__(self) -> str:
        crit = f"{self.critical_value:.4f}" if self.critical_value is not None else "N/A"
        return (
            f"SensitivityResult({self.scheme_name!r}, param={self.parameter_name!r}, "
            f"critical={crit})"
        )


# ---------------------------------------------------------------------------
# Analytical critical-value solvers
# ---------------------------------------------------------------------------


def find_critical_c(lambda_bits: float, n: int) -> float:
    """Compute the critical SALSA constant C* where M(S) = 0.

    From M(S) = C * log2(n) - lambda = 0, we obtain C* = lambda / log2(n).

    Args:
        lambda_bits: Classical security level in bits.
        n: Lattice dimension.

    Returns:
        The critical constant C* below which the security margin is negative.

    Raises:
        ValueError: If *n* is not greater than 1.
    """
    if n <= 1:
        raise ValueError(f"Lattice dimension must be > 1, got {n}")
    return lambda_bits / math.log2(n)


def find_critical_k(lambda_bits: float, n: int, c: float = 52.0) -> float:
    """Compute the critical attack improvement factor K* where M_eff(S) = 0.

    From M_eff = C * log2(n) - lambda - log2(K) = 0, we get
    K* = 2^(C * log2(n) - lambda).

    Args:
        lambda_bits: Classical security level in bits.
        n: Lattice dimension.
        c: SALSA regression constant.

    Returns:
        The critical improvement factor K*.  Values greater than 1 indicate
        the scheme has a positive baseline margin; the attack must improve
        by at least K* to break even.

    Raises:
        ValueError: If *n* is not greater than 1.
    """
    if n <= 1:
        raise ValueError(f"Lattice dimension must be > 1, got {n}")
    tau = c * math.log2(n)
    margin = tau - lambda_bits
    return 2.0 ** margin


# ---------------------------------------------------------------------------
# Sweep analyses
# ---------------------------------------------------------------------------


def analyze_c_sensitivity(
    scheme_name: str,
    lambda_bits: float,
    n: int,
    c_range: tuple[float, float] = (47.0, 57.0),
    step: float = 0.5,
) -> SensitivityResult:
    """Sweep the SALSA constant C and record the resulting margins.

    Args:
        scheme_name: Scheme identifier for labelling.
        lambda_bits: Classical security level in bits.
        n: Lattice dimension.
        c_range: Inclusive (min, max) bounds for C.
        step: Increment between successive C values.

    Returns:
        A ``SensitivityResult`` with ``parameter_name="C"`` and the
        analytically determined critical C*.
    """
    log2_n = math.log2(n)

    c_values: list[float] = []
    margin_values: list[float] = []

    c = c_range[0]
    while c <= c_range[1] + step / 2:
        c_values.append(round(c, 4))
        margin_values.append(c * log2_n - lambda_bits)
        c += step

    critical = find_critical_c(lambda_bits, n)
    # Only report the critical value if it falls within the swept range
    if c_range[0] <= critical <= c_range[1]:
        critical_value: float | None = critical
    else:
        critical_value = None

    return SensitivityResult(
        scheme_name=scheme_name,
        parameter_name="C",
        parameter_values=c_values,
        margin_values=margin_values,
        critical_value=critical_value,
    )


def analyze_k_sensitivity(
    scheme_name: str,
    lambda_bits: float,
    n: int,
    c: float = 52.0,
    k_powers: list[int] | None = None,
) -> SensitivityResult:
    """Sweep the attack improvement factor K and record effective margins.

    K is expressed as powers of 10: K = 10^p for each p in *k_powers*.
    The effective margin is M_eff(S) = tau_hat(S) - lambda(S) - log2(K).

    Args:
        scheme_name: Scheme identifier for labelling.
        lambda_bits: Classical security level in bits.
        n: Lattice dimension.
        c: SALSA regression constant (default 52).
        k_powers: Exponents p such that K = 10^p.  Defaults to
            ``[0, 1, 2, 3, 4, 5, 6]``.

    Returns:
        A ``SensitivityResult`` with ``parameter_name="K"`` and the
        analytically determined critical K*.
    """
    if k_powers is None:
        k_powers = [0, 1, 2, 3, 4, 5, 6]

    tau = c * math.log2(n)
    baseline_margin = tau - lambda_bits

    k_values: list[float] = []
    margin_values: list[float] = []

    for p in k_powers:
        k = 10.0 ** p
        k_values.append(k)
        margin_values.append(baseline_margin - math.log2(k))

    critical_k = find_critical_k(lambda_bits, n, c)

    return SensitivityResult(
        scheme_name=scheme_name,
        parameter_name="K",
        parameter_values=k_values,
        margin_values=margin_values,
        critical_value=critical_k,
    )


# ---------------------------------------------------------------------------
# Full analysis driver
# ---------------------------------------------------------------------------


def run_full_sensitivity_analysis(c: float = 52.0) -> list[SensitivityResult]:
    """Run C-sensitivity and K-sensitivity analyses for all ML-KEM schemes.

    Args:
        c: Baseline SALSA regression constant used for the K-sensitivity
           sweep.  The C-sensitivity sweep always covers [47, 57].

    Returns:
        A list of ``SensitivityResult`` objects -- two per scheme (one for
        C, one for K), ordered by scheme then by parameter.
    """
    results: list[SensitivityResult] = []

    for name, n, lambda_bits in _ML_KEM_SCHEMES:
        results.append(analyze_c_sensitivity(name, lambda_bits, n))
        results.append(analyze_k_sensitivity(name, lambda_bits, n, c=c))

    return results
