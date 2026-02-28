"""Publication-quality visualization for PQC security margin analysis.

Generates four matplotlib figures suitable for inclusion in academic papers
and dissertations:

1. **Security margins** -- grouped bar chart comparing M(S) across ML-KEM
   parameter sets and SALSA regression scenarios.
2. **C-sensitivity** -- line plot showing how M(S) varies with the SALSA
   regression constant C, including critical thresholds.
3. **K-degradation** -- line plot of effective margin under hypothetical
   K-fold attack improvements.
4. **SALSA regression** -- scatter plot of empirical data with linear and
   quadratic fits extrapolated to PQC dimensions.
"""

from __future__ import annotations

import math
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from numpy.typing import NDArray

from salsa_regression import (
    RegressionResult,
    extrapolate_to_pqc_dimensions,
    fit_linear_model,
    fit_quadratic_model,
    get_salsa_data,
    predict,
)
from security_margin import (
    CONSERVATIVE_C,
    MODERATE_C,
    OPTIMISTIC_C,
    SecurityMarginResult,
    compute_margin,
    get_all_schemes,
)
from sensitivity import (
    SensitivityResult,
    analyze_c_sensitivity,
    analyze_k_sensitivity,
)

__all__ = [
    "plot_security_margins",
    "plot_c_sensitivity",
    "plot_k_degradation",
    "plot_regression",
    "generate_all_plots",
]

# ---------------------------------------------------------------------------
# Publication style configuration
# ---------------------------------------------------------------------------

_PALETTE = {
    "conservative": "#2c3e50",
    "moderate": "#2980b9",
    "optimistic": "#7fb3d8",
    "kem_512": "#c0392b",
    "kem_768": "#2980b9",
    "kem_1024": "#27ae60",
    "linear_fit": "#e74c3c",
    "quadratic_fit": "#2c3e50",
    "data_points": "#2c3e50",
    "confidence": "#bdc3c7",
    "threshold": "#e74c3c",
    "zero_line": "#7f8c8d",
}

_SCHEME_COLORS = {
    "ML-KEM-512": _PALETTE["kem_512"],
    "ML-KEM-768": _PALETTE["kem_768"],
    "ML-KEM-1024": _PALETTE["kem_1024"],
}

_SCENARIO_COLORS = [
    _PALETTE["conservative"],
    _PALETTE["moderate"],
    _PALETTE["optimistic"],
]

_SCENARIO_LABELS = [
    f"Conservative (C={CONSERVATIVE_C:.0f})",
    f"Moderate (C={MODERATE_C:.0f})",
    f"Optimistic (C={OPTIMISTIC_C:.0f})",
]

_SCENARIO_C_VALUES = [CONSERVATIVE_C, MODERATE_C, OPTIMISTIC_C]


def _apply_publication_style() -> None:
    """Configure matplotlib rcParams for publication-quality output."""
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["Computer Modern Roman", "Times New Roman", "DejaVu Serif"],
        "font.size": 11,
        "axes.labelsize": 12,
        "axes.titlesize": 13,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "figure.figsize": (8, 5),
        "figure.dpi": 300,
        "savefig.dpi": 300,
        "savefig.bbox": "tight",
        "axes.grid": True,
        "grid.alpha": 0.3,
        "grid.linestyle": "--",
        "axes.spines.top": False,
        "axes.spines.right": False,
        "lines.linewidth": 1.5,
        "lines.markersize": 6,
    })


# ---------------------------------------------------------------------------
# ML-KEM scheme filter
# ---------------------------------------------------------------------------

_ML_KEM_NAMES = ("ML-KEM-512", "ML-KEM-768", "ML-KEM-1024")


def _get_ml_kem_schemes() -> list[tuple[str, int, float]]:
    """Return ML-KEM parameter sets as (name, n, lambda) tuples."""
    schemes = []
    for ps in get_all_schemes():
        if ps.name in _ML_KEM_NAMES:
            schemes.append((ps.name, ps.lattice_dimension, ps.classical_security_bits))
    return schemes


# ---------------------------------------------------------------------------
# Plot 1: Security margins (grouped bar chart)
# ---------------------------------------------------------------------------


def plot_security_margins(
    output_path: str | Path = "output/security_margins.png",
) -> None:
    """Grouped bar chart of security margins across schemes and scenarios.

    Three groups (ML-KEM-512, ML-KEM-768, ML-KEM-1024) each contain three
    bars corresponding to the conservative, moderate, and optimistic SALSA
    regression scenarios.  Horizontal reference lines mark M = 0 (critical
    threshold) and M = 20 (caution zone).

    Args:
        output_path: Destination file path for the saved figure.
    """
    _apply_publication_style()

    schemes = _get_ml_kem_schemes()
    scheme_names = [s[0] for s in schemes]
    n_schemes = len(scheme_names)
    n_scenarios = len(_SCENARIO_C_VALUES)

    # Compute margins: rows = schemes, columns = scenarios
    margins = np.zeros((n_schemes, n_scenarios))
    for i, name in enumerate(scheme_names):
        for j, c in enumerate(_SCENARIO_C_VALUES):
            result = compute_margin(name, c)
            margins[i, j] = result.margin_bits

    fig, ax = plt.subplots(figsize=(9, 5.5))

    bar_width = 0.22
    x = np.arange(n_schemes)

    for j in range(n_scenarios):
        offset = (j - (n_scenarios - 1) / 2) * bar_width
        bars = ax.bar(
            x + offset,
            margins[:, j],
            width=bar_width,
            label=_SCENARIO_LABELS[j],
            color=_SCENARIO_COLORS[j],
            edgecolor="white",
            linewidth=0.5,
        )
        # Value annotations
        for bar in bars:
            height = bar.get_height()
            va = "bottom" if height >= 0 else "top"
            ax.annotate(
                f"{height:.1f}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3 if height >= 0 else -3),
                textcoords="offset points",
                ha="center",
                va=va,
                fontsize=8,
            )

    # Reference lines
    ax.axhline(y=0, color=_PALETTE["threshold"], linewidth=1.2, label="$M = 0$ (critical)")
    ax.axhline(
        y=20,
        color=_PALETTE["zero_line"],
        linewidth=1.0,
        linestyle="--",
        label="$M = 20$ (caution)",
    )

    ax.set_xlabel("Scheme")
    ax.set_ylabel("Security Margin $M(S)$ (bits)")
    ax.set_title("Post-Quantum Security Margins Under SALSA Regression Scenarios")
    ax.set_xticks(x)
    ax.set_xticklabels(scheme_names)
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 2: C-sensitivity
# ---------------------------------------------------------------------------


def plot_c_sensitivity(
    output_path: str | Path = "output/c_sensitivity.png",
) -> None:
    """Line plot of security margin as a function of the SALSA constant C.

    For each ML-KEM scheme, the margin M(S) = C * log2(n) - lambda(S) is
    plotted over a range of C values.  Vertical dashed lines mark the
    critical C* where M(S) = 0.

    Args:
        output_path: Destination file path for the saved figure.
    """
    _apply_publication_style()

    schemes = _get_ml_kem_schemes()
    c_range = (10.0, 57.0)
    step = 0.25

    fig, ax = plt.subplots(figsize=(8, 5))

    for name, n, lambda_bits in schemes:
        result = analyze_c_sensitivity(
            scheme_name=name,
            lambda_bits=lambda_bits,
            n=n,
            c_range=c_range,
            step=step,
        )
        color = _SCHEME_COLORS[name]
        ax.plot(
            result.parameter_values,
            result.margin_values,
            color=color,
            label=name,
        )
        # Mark critical C*
        if result.critical_value is not None:
            ax.axvline(
                x=result.critical_value,
                color=color,
                linewidth=0.8,
                linestyle=":",
            )
            ax.annotate(
                f"$C^* = {result.critical_value:.2f}$",
                xy=(result.critical_value, 0),
                xytext=(8, 12),
                textcoords="offset points",
                fontsize=9,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )

    ax.axhline(y=0, color=_PALETTE["threshold"], linewidth=1.0)

    # Shade the three scenario bands
    ax.axvspan(CONSERVATIVE_C, OPTIMISTIC_C, alpha=0.08, color=_PALETTE["moderate"])
    ax.annotate(
        "Plausible range",
        xy=((CONSERVATIVE_C + OPTIMISTIC_C) / 2, ax.get_ylim()[1] * 0.9),
        ha="center",
        fontsize=9,
        fontstyle="italic",
        color=_PALETTE["moderate"],
    )

    ax.set_xlabel("SALSA Regression Constant $C$")
    ax.set_ylabel("Security Margin $M(S)$ (bits)")
    ax.set_title("Sensitivity of Security Margin to SALSA Constant $C$")
    ax.legend(loc="upper left", framealpha=0.9)

    fig.tight_layout()
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 3: K-degradation
# ---------------------------------------------------------------------------


def plot_k_degradation(
    output_path: str | Path = "output/k_degradation.png",
) -> None:
    """Line plot of effective margin under K-fold attack improvement.

    The x-axis shows log10(K) and the y-axis the effective margin
    M_eff(S) = tau_hat(S) - lambda(S) - log2(K).  Vertical markers
    indicate the critical K* for each scheme.

    Args:
        output_path: Destination file path for the saved figure.
    """
    _apply_publication_style()

    schemes = _get_ml_kem_schemes()
    k_powers = list(range(0, 151))

    fig, ax = plt.subplots(figsize=(8, 5))

    # Stagger annotation offsets to avoid overlap when critical values are close
    annotation_offsets = [(-40, 18), (8, -20), (-40, -20)]

    for idx, (name, n, lambda_bits) in enumerate(schemes):
        result = analyze_k_sensitivity(
            scheme_name=name,
            lambda_bits=lambda_bits,
            n=n,
            c=MODERATE_C,
            k_powers=k_powers,
        )
        log10_k = [math.log10(k) if k > 0 else 0 for k in result.parameter_values]
        color = _SCHEME_COLORS[name]
        ax.plot(
            log10_k,
            result.margin_values,
            color=color,
            label=name,
        )
        # Mark critical K*
        if result.critical_value is not None and result.critical_value > 1:
            log10_k_crit = math.log10(result.critical_value)
            ax.plot(
                log10_k_crit,
                0,
                marker="o",
                color=color,
                markersize=7,
                zorder=5,
            )
            offset = annotation_offsets[idx % len(annotation_offsets)]
            ax.annotate(
                f"$\\log_{{10}} K^* = {log10_k_crit:.1f}$",
                xy=(log10_k_crit, 0),
                xytext=offset,
                textcoords="offset points",
                fontsize=9,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.8),
            )

    ax.axhline(y=0, color=_PALETTE["threshold"], linewidth=1.0)

    ax.set_xlabel("Attack Improvement Factor $\\log_{10}(K)$")
    ax.set_ylabel("Effective Security Margin $M_{\\mathrm{eff}}(S)$ (bits)")
    ax.set_title(
        f"Margin Degradation Under Attack Improvement ($C = {MODERATE_C:.0f}$)"
    )
    ax.legend(loc="upper right", framealpha=0.9)

    fig.tight_layout()
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Plot 4: SALSA regression
# ---------------------------------------------------------------------------


def plot_regression(
    output_path: str | Path = "output/salsa_regression.png",
) -> None:
    """Scatter plot of SALSA data with linear and quadratic regression fits.

    Empirical data points are plotted in the log2(n) domain alongside
    both fitted models.  Extrapolation to PQC-relevant dimensions
    (n = 512 .. 1536) is shown as dashed extensions with a 95 %
    confidence band for the linear model.

    Args:
        output_path: Destination file path for the saved figure.
    """
    _apply_publication_style()

    n_values, cost_values = get_salsa_data()
    log2_n = np.log2(n_values)

    linear = fit_linear_model(n_values, cost_values)
    quadratic = fit_quadratic_model(n_values, cost_values)

    # Dense x grid for smooth curves
    x_fit = np.linspace(log2_n.min() - 0.5, log2_n.max() + 0.5, 200)
    x_extrap = np.linspace(log2_n.max(), math.log2(1536) + 0.3, 200)
    x_full = np.concatenate([x_fit, x_extrap])

    y_linear = np.polyval(np.array(linear.coefficients), x_full)
    y_quadratic = np.polyval(np.array(quadratic.coefficients), x_full)

    # Confidence band for linear model (simplified prediction interval)
    n_obs = len(log2_n)
    x_mean = float(np.mean(log2_n))
    ss_x = float(np.sum((log2_n - x_mean) ** 2))
    t_crit = 4.303  # t(0.975, df=2) for n=4 observations
    se_pred = linear.residual_std * np.sqrt(
        1.0 + 1.0 / n_obs + (x_full - x_mean) ** 2 / ss_x
    )
    y_lin_fit = np.polyval(np.array(linear.coefficients), x_full)
    y_upper = y_lin_fit + t_crit * se_pred
    y_lower = y_lin_fit - t_crit * se_pred

    fig, ax = plt.subplots(figsize=(9, 5.5))

    # Confidence band
    ax.fill_between(
        x_full,
        y_lower,
        y_upper,
        alpha=0.15,
        color=_PALETTE["confidence"],
        label="95% prediction interval (linear)",
    )

    # Regression lines (fit region)
    ax.plot(
        x_fit,
        np.polyval(np.array(linear.coefficients), x_fit),
        color=_PALETTE["linear_fit"],
        label=(
            f"Linear: $y = {linear.coefficients[0]:.3f} \\, \\log_2 n "
            f"+ ({linear.coefficients[1]:.3f})$ "
            f"($R^2 = {linear.r_squared:.4f}$)"
        ),
    )
    ax.plot(
        x_fit,
        np.polyval(np.array(quadratic.coefficients), x_fit),
        color=_PALETTE["quadratic_fit"],
        label=(
            f"Quadratic ($R^2 = {quadratic.r_squared:.4f}$)"
        ),
    )

    # Extrapolation region (dashed)
    ax.plot(
        x_extrap,
        np.polyval(np.array(linear.coefficients), x_extrap),
        color=_PALETTE["linear_fit"],
        linestyle="--",
        linewidth=1.2,
    )
    ax.plot(
        x_extrap,
        np.polyval(np.array(quadratic.coefficients), x_extrap),
        color=_PALETTE["quadratic_fit"],
        linestyle="--",
        linewidth=1.2,
    )

    # Data points (on top)
    ax.scatter(
        log2_n,
        cost_values,
        color=_PALETTE["data_points"],
        s=60,
        zorder=5,
        edgecolors="white",
        linewidths=0.8,
        label="SALSA empirical data",
    )
    # Label each data point
    for xi, yi, ni in zip(log2_n, cost_values, n_values):
        ax.annotate(
            f"$n={int(ni)}$",
            xy=(xi, yi),
            xytext=(6, 6),
            textcoords="offset points",
            fontsize=8,
        )

    # Mark PQC dimensions on x-axis
    pqc_dims = [512, 768, 1024, 1280, 1536]
    for dim in pqc_dims:
        log2_dim = math.log2(dim)
        ax.axvline(x=log2_dim, color="#bdc3c7", linewidth=0.6, linestyle=":")
        ax.text(
            log2_dim,
            ax.get_ylim()[0] + 0.5,
            f"$n={dim}$",
            rotation=90,
            fontsize=7,
            va="bottom",
            ha="right",
            color="#7f8c8d",
        )

    # Vertical separator between fit and extrapolation regions
    ax.axvline(
        x=float(log2_n.max()),
        color=_PALETTE["zero_line"],
        linewidth=0.8,
        linestyle="-.",
        alpha=0.5,
    )
    ax.text(
        float(log2_n.max()) + 0.1,
        ax.get_ylim()[1] * 0.95 if ax.get_ylim()[1] > 0 else 15,
        "extrapolation",
        fontsize=8,
        fontstyle="italic",
        color=_PALETTE["zero_line"],
        va="top",
    )

    ax.set_xlabel("$\\log_2(n)$ (lattice dimension)")
    ax.set_ylabel("$\\log_2$(gate count)")
    ax.set_title("SALSA Regression: Attack Cost vs. Lattice Dimension")
    ax.legend(loc="upper left", fontsize=9, framealpha=0.9)

    fig.tight_layout()
    _save_figure(fig, output_path)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _save_figure(fig: matplotlib.figure.Figure, path: str | Path) -> None:
    """Ensure the output directory exists and save the figure."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def generate_all_plots(output_dir: str | Path = "output") -> None:
    """Generate all four publication-quality plots.

    Creates the output directory if it does not exist, then writes:
    - ``security_margins.png``
    - ``c_sensitivity.png``
    - ``k_degradation.png``
    - ``salsa_regression.png``

    Args:
        output_dir: Directory in which to write the figure files.
    """
    base = Path(output_dir)
    base.mkdir(parents=True, exist_ok=True)

    plot_security_margins(base / "security_margins.png")
    plot_c_sensitivity(base / "c_sensitivity.png")
    plot_k_degradation(base / "k_degradation.png")
    plot_regression(base / "salsa_regression.png")


if __name__ == "__main__":
    generate_all_plots()
