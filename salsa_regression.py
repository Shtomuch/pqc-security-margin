"""SALSA regression module.

Performs OLS regression on SALSA (Status of Lattice-based Signature and
Authentication) framework data to model the relationship between lattice
dimension and the cost of best known quantum attacks.

The empirical data points represent log2(gate count) for the best known
attacks at lattice dimensions n in {128, 350, 512, 1024}. We fit both
linear and quadratic models in the log2(n) domain and provide extrapolation
to PQC-relevant dimensions.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np
from numpy.typing import NDArray
from scipy import stats

__all__ = [
    "RegressionResult",
    "fit_linear_model",
    "fit_quadratic_model",
    "predict",
    "compare_models",
    "get_salsa_data",
    "extrapolate_to_pqc_dimensions",
]

# Raw SALSA data: (lattice dimension n, log2 gate count of best known attack)
_SALSA_DIMENSIONS = np.array([128, 350, 512, 1024], dtype=np.float64)
_SALSA_COSTS = np.array([6.6, 10.0, 11.5, 13.3], dtype=np.float64)

# Standard PQC dimensions for extrapolation
_PQC_DIMENSIONS = (512, 768, 1024, 1280, 1536)


@dataclass(frozen=True)
class RegressionResult:
    """Stores the outcome of an OLS regression fit.

    Attributes:
        model_type: Either ``"linear"`` or ``"quadratic"``, indicating the
            polynomial degree in the log2(n) domain.
        coefficients: Fitted coefficients in descending power order.
            Linear: ``(a, b)`` for ``a * log2(n) + b``.
            Quadratic: ``(a, b, c)`` for ``a * log2(n)^2 + b * log2(n) + c``.
        r_squared: Coefficient of determination.
        residual_std: Standard deviation of residuals.
        confidence_intervals_95: 95 % confidence interval for each coefficient,
            ordered to match *coefficients*.
    """

    model_type: str
    coefficients: tuple[float, ...]
    r_squared: float
    residual_std: float
    confidence_intervals_95: list[tuple[float, float]]


def get_salsa_data() -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return the raw SALSA data points.

    Returns:
        A pair ``(n_values, cost_values)`` where *n_values* are lattice
        dimensions and *cost_values* are log2 of the gate count for the
        best known quantum attack at each dimension.
    """
    return _SALSA_DIMENSIONS.copy(), _SALSA_COSTS.copy()


def fit_linear_model(
    n_values: Sequence[float] | NDArray[np.float64],
    cost_values: Sequence[float] | NDArray[np.float64],
) -> RegressionResult:
    """Fit a linear model ``cost = a * log2(n) + b``.

    The independent variable is transformed to log2(n) before fitting.
    Confidence intervals are derived from the standard errors reported by
    ``scipy.stats.linregress``.
    """
    x = np.log2(np.asarray(n_values, dtype=np.float64))
    y = np.asarray(cost_values, dtype=np.float64)

    result = stats.linregress(x, y)
    slope: float = float(result.slope)
    intercept: float = float(result.intercept)
    r_squared: float = float(result.rvalue) ** 2

    residuals = y - (slope * x + intercept)
    residual_std = float(np.std(residuals, ddof=2))

    n = len(x)
    t_crit = float(stats.t.ppf(0.975, df=n - 2))
    slope_ci = (
        slope - t_crit * float(result.stderr),
        slope + t_crit * float(result.stderr),
    )
    intercept_ci = (
        intercept - t_crit * float(result.intercept_stderr),
        intercept + t_crit * float(result.intercept_stderr),
    )

    return RegressionResult(
        model_type="linear",
        coefficients=(slope, intercept),
        r_squared=r_squared,
        residual_std=residual_std,
        confidence_intervals_95=[slope_ci, intercept_ci],
    )


def fit_quadratic_model(
    n_values: Sequence[float] | NDArray[np.float64],
    cost_values: Sequence[float] | NDArray[np.float64],
) -> RegressionResult:
    """Fit a quadratic model ``cost = a * log2(n)^2 + b * log2(n) + c``.

    Uses ordinary least squares via the normal equations (numpy polyfit).
    Confidence intervals are computed from the covariance matrix of the fit.
    """
    x = np.log2(np.asarray(n_values, dtype=np.float64))
    y = np.asarray(cost_values, dtype=np.float64)

    coeffs, cov = np.polyfit(x, y, deg=2, cov=True)
    a, b, c = float(coeffs[0]), float(coeffs[1]), float(coeffs[2])

    y_hat = np.polyval(coeffs, x)
    ss_res = float(np.sum((y - y_hat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r_squared = 1.0 - ss_res / ss_tot

    n = len(x)
    deg = 2
    residual_std = float(np.sqrt(ss_res / max(n - (deg + 1), 1)))

    t_crit = float(stats.t.ppf(0.975, df=max(n - (deg + 1), 1)))
    std_errors = np.sqrt(np.diag(cov))
    confidence_intervals = [
        (float(coeffs[i]) - t_crit * float(std_errors[i]),
         float(coeffs[i]) + t_crit * float(std_errors[i]))
        for i in range(len(coeffs))
    ]

    return RegressionResult(
        model_type="quadratic",
        coefficients=(a, b, c),
        r_squared=r_squared,
        residual_std=residual_std,
        confidence_intervals_95=confidence_intervals,
    )


def predict(model: RegressionResult, n: int) -> float:
    """Predict the log2 attack cost for a given lattice dimension *n*.

    Evaluates the fitted polynomial (linear or quadratic) at ``log2(n)``.
    """
    log2_n = math.log2(n)
    return float(np.polyval(np.array(model.coefficients), log2_n))


def compare_models() -> dict[str, object]:
    """Fit both linear and quadratic models on the SALSA data and compare.

    Returns a dictionary with keys ``"linear"``, ``"quadratic"``,
    ``"recommendation"``, and per-model R-squared values.
    """
    n_vals, cost_vals = get_salsa_data()
    linear = fit_linear_model(n_vals, cost_vals)
    quadratic = fit_quadratic_model(n_vals, cost_vals)

    # Prefer the simpler model unless the quadratic R^2 is meaningfully better
    r2_improvement = quadratic.r_squared - linear.r_squared
    if r2_improvement > 0.01:
        recommendation = "quadratic"
    else:
        recommendation = "linear"

    return {
        "linear": linear,
        "quadratic": quadratic,
        "linear_r_squared": linear.r_squared,
        "quadratic_r_squared": quadratic.r_squared,
        "recommendation": recommendation,
    }


def extrapolate_to_pqc_dimensions(
    model: RegressionResult,
) -> dict[str, float]:
    """Extrapolate the fitted model to standard PQC lattice dimensions.

    Evaluates the model at n in {512, 768, 1024, 1280, 1536} and returns
    a mapping from ``"n=<dim>"`` to predicted log2 attack cost.
    """
    return {f"n={dim}": predict(model, dim) for dim in _PQC_DIMENSIONS}
