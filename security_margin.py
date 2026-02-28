"""Security margin estimation for post-quantum cryptographic schemes.

Implements the security margin model M(S) = tau_hat(S) - lambda(S), where:

- lambda(S) is the classical security level in bits, derived from the NIST
  Lattice Estimator for a given parameter set S.
- tau_hat(S) is the estimated quantum attack cost in bits, computed via the
  SALSA (Systematic Analysis of Lattice Security Assumptions) regression model:
      tau_hat(S) = C * log2(n)
  where n is the lattice dimension and C is a scenario-dependent constant.

Three scenarios capture uncertainty in quantum computational progress:

- Conservative (C1): C = 47 -- pessimistic, assumes fastest quantum progress
- Moderate (C2):     C = 52 -- baseline estimate
- Optimistic (C3):   C = 57 -- assumes slowest quantum progress

The margin M(S) quantifies how many bits of security headroom remain beyond
the classical baseline, accounting for projected quantum attack capabilities.

Reference parameter sets are drawn from NIST PQC standardization (FIPS 203/204)
with classical security estimates from the NIST Lattice Estimator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ParameterSet",
    "SecurityMarginResult",
    "MarginCategory",
    "CONSERVATIVE_C",
    "MODERATE_C",
    "OPTIMISTIC_C",
    "get_scheme",
    "get_all_schemes",
    "compute_tau",
    "compute_margin",
    "evaluate_all_schemes",
]

CONSERVATIVE_C: float = 47.0
MODERATE_C: float = 52.0
OPTIMISTIC_C: float = 57.0


class MarginCategory(Enum):
    """Qualitative interpretation of a computed security margin."""

    HIGH = "High security margin -- safe for long-term deployment"
    MODERATE = "Moderate margin -- monitor developments"
    LOW = "Low margin -- consider migration planning"
    NEGATIVE = "Negative margin -- immediate action required"


@dataclass(frozen=True, slots=True)
class ParameterSet:
    """NIST PQC parameter set with associated classical security estimate.

    Attributes:
        name: Canonical scheme identifier (e.g. ``ML-KEM-768``).
        security_category: NIST security category (1--5).
        lattice_dimension: Effective lattice dimension n used in security
            estimation.
        classical_security_bits: Classical security level lambda(S) in bits,
            as estimated by the NIST Lattice Estimator.
    """

    name: str
    security_category: int
    lattice_dimension: int
    classical_security_bits: float

    def __repr__(self) -> str:
        return (
            f"ParameterSet(name={self.name!r}, cat={self.security_category}, "
            f"n={self.lattice_dimension}, lambda={self.classical_security_bits})"
        )


@dataclass(frozen=True, slots=True)
class SecurityMarginResult:
    """Result of a security margin computation for a single scheme.

    Attributes:
        scheme_name: Identifier of the evaluated scheme.
        lambda_bits: Classical security level in bits.
        tau_bits: Estimated quantum attack cost in bits (tau_hat).
        margin_bits: Security margin M(S) = tau_hat(S) - lambda(S).
        interpretation: Qualitative assessment of the margin.
    """

    scheme_name: str
    lambda_bits: float
    tau_bits: float
    margin_bits: float
    interpretation: str

    def __repr__(self) -> str:
        return (
            f"SecurityMarginResult({self.scheme_name!r}, "
            f"M={self.margin_bits:+.1f} bits)"
        )


# ---------------------------------------------------------------------------
# Reference parameter sets
# ---------------------------------------------------------------------------

_PARAMETER_SETS: dict[str, ParameterSet] = {}


def _register(*sets: ParameterSet) -> None:
    for ps in sets:
        _PARAMETER_SETS[ps.name] = ps


_register(
    # ML-KEM (FIPS 203) -- CRYSTALS-Kyber successor
    ParameterSet(
        name="ML-KEM-512",
        security_category=1,
        lattice_dimension=512,
        classical_security_bits=118.0,
    ),
    ParameterSet(
        name="ML-KEM-768",
        security_category=3,
        lattice_dimension=768,
        classical_security_bits=182.0,
    ),
    ParameterSet(
        name="ML-KEM-1024",
        security_category=5,
        lattice_dimension=1024,
        classical_security_bits=256.0,
    ),
    # ML-DSA (FIPS 204) -- CRYSTALS-Dilithium successor
    ParameterSet(
        name="ML-DSA-44",
        security_category=2,
        lattice_dimension=1024,
        classical_security_bits=128.0,
    ),
    ParameterSet(
        name="ML-DSA-65",
        security_category=3,
        lattice_dimension=1280,
        classical_security_bits=192.0,
    ),
    ParameterSet(
        name="ML-DSA-87",
        security_category=5,
        lattice_dimension=1536,
        classical_security_bits=256.0,
    ),
)


# ---------------------------------------------------------------------------
# Core computation
# ---------------------------------------------------------------------------


def _interpret_margin(margin: float) -> str:
    """Return a qualitative interpretation of the security margin."""
    if margin > 40:
        return MarginCategory.HIGH.value
    if margin > 20:
        return MarginCategory.MODERATE.value
    if margin > 0:
        return MarginCategory.LOW.value
    return MarginCategory.NEGATIVE.value


def get_scheme(name: str) -> ParameterSet:
    """Look up a parameter set by its canonical name.

    Args:
        name: Scheme identifier, e.g. ``"ML-KEM-768"``.

    Returns:
        The corresponding ``ParameterSet``.

    Raises:
        KeyError: If *name* does not match any registered scheme.
    """
    try:
        return _PARAMETER_SETS[name]
    except KeyError:
        available = ", ".join(sorted(_PARAMETER_SETS))
        raise KeyError(
            f"Unknown scheme {name!r}. Available: {available}"
        ) from None


def get_all_schemes() -> list[ParameterSet]:
    """Return all registered parameter sets, ordered by lattice dimension."""
    return sorted(_PARAMETER_SETS.values(), key=lambda ps: ps.lattice_dimension)


def compute_tau(n: int, c: float = MODERATE_C) -> float:
    """Estimate the quantum attack cost via the SALSA regression model.

    Computes tau_hat = C * log2(n).

    Args:
        n: Lattice dimension.
        c: Regression constant. Defaults to the moderate scenario (52).

    Returns:
        Estimated quantum attack cost in bits.

    Raises:
        ValueError: If *n* is not a positive integer.
    """
    if n <= 0:
        raise ValueError(f"Lattice dimension must be positive, got {n}")
    return c * math.log2(n)


def compute_margin(
    scheme: str | ParameterSet,
    c: float = MODERATE_C,
) -> SecurityMarginResult:
    """Compute the security margin for a single scheme.

    M(S) = tau_hat(S) - lambda(S)

    Args:
        scheme: Either a scheme name (str) or a ``ParameterSet`` instance.
        c: SALSA regression constant. Defaults to the moderate scenario.

    Returns:
        A ``SecurityMarginResult`` containing the computed margin and its
        qualitative interpretation.
    """
    ps = get_scheme(scheme) if isinstance(scheme, str) else scheme
    tau = compute_tau(ps.lattice_dimension, c)
    margin = tau - ps.classical_security_bits
    return SecurityMarginResult(
        scheme_name=ps.name,
        lambda_bits=ps.classical_security_bits,
        tau_bits=tau,
        margin_bits=margin,
        interpretation=_interpret_margin(margin),
    )


def evaluate_all_schemes(c: float = MODERATE_C) -> list[SecurityMarginResult]:
    """Compute security margins for every registered parameter set.

    Args:
        c: SALSA regression constant. Defaults to the moderate scenario.

    Returns:
        List of ``SecurityMarginResult`` objects, ordered by lattice dimension.
    """
    return [compute_margin(ps, c) for ps in get_all_schemes()]
