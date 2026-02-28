"""Command-line interface for the PQC Security Margin analysis tool.

Provides subcommands for computing security margins, running regression
analysis, performing sensitivity sweeps, generating publication-quality
plots, and displaying the quantum threat taxonomy.

Usage examples::

    python main.py margin --scheme ML-KEM-512 --scenario C2
    python main.py margin --all
    python main.py regression
    python main.py sensitivity --scheme ML-KEM-768
    python main.py visualize --output-dir figures/
    python main.py taxonomy
"""

from __future__ import annotations

import sys
from pathlib import Path

import click

from security_margin import (
    CONSERVATIVE_C,
    MODERATE_C,
    OPTIMISTIC_C,
    compute_margin,
    evaluate_all_schemes,
    get_all_schemes,
    get_scheme,
)
from salsa_regression import (
    compare_models,
    extrapolate_to_pqc_dimensions,
    fit_linear_model,
    fit_quadratic_model,
    get_salsa_data,
)
from sensitivity import (
    analyze_c_sensitivity,
    analyze_k_sensitivity,
    find_critical_c,
    find_critical_k,
    run_full_sensitivity_analysis,
)
from taxonomy import (
    format_decision_matrix,
    format_taxonomy_report,
)

_SCENARIO_MAP: dict[str, float] = {
    "C1": CONSERVATIVE_C,
    "C2": MODERATE_C,
    "C3": OPTIMISTIC_C,
}


@click.group()
@click.version_option(version="1.0.0", prog_name="pqc-security-margin")
def cli() -> None:
    """PQC Security Margin Calculator.

    Computes and analyses security margins for NIST post-quantum
    cryptographic schemes under varying assumptions about quantum
    computational progress.
    """


# ---------------------------------------------------------------------------
# margin subcommand
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--scheme", "-s",
    type=click.Choice(
        ["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024",
         "ML-DSA-44", "ML-DSA-65", "ML-DSA-87"],
        case_sensitive=True,
    ),
    default=None,
    help="Parameter set to evaluate.",
)
@click.option(
    "--scenario", "-c",
    type=click.Choice(["C1", "C2", "C3"], case_sensitive=True),
    default="C2",
    show_default=True,
    help="Scenario: C1 (conservative), C2 (moderate), C3 (optimistic).",
)
@click.option(
    "--all-schemes", "-a",
    is_flag=True,
    default=False,
    help="Evaluate all registered parameter sets.",
)
def margin(scheme: str | None, scenario: str, all_schemes: bool) -> None:
    """Compute the security margin M(S) for a PQC parameter set."""
    c = _SCENARIO_MAP[scenario]
    scenario_label = f"{scenario} (C={c:.0f})"

    if all_schemes:
        results = evaluate_all_schemes(c)
        click.echo(f"\nSecurity margins under scenario {scenario_label}:")
        click.echo("=" * 72)
        click.echo(f"{'Scheme':<16} {'lambda(S)':>10} {'tau_hat(S)':>12} {'M(S)':>8}   Interpretation")
        click.echo("-" * 72)
        for r in results:
            click.echo(
                f"{r.scheme_name:<16} {r.lambda_bits:>10.1f} {r.tau_bits:>12.1f} "
                f"{r.margin_bits:>+8.1f}   {r.interpretation}"
            )
        click.echo("=" * 72)
        return

    if scheme is None:
        click.echo("Error: specify --scheme or use --all-schemes.", err=True)
        sys.exit(1)

    result = compute_margin(scheme, c)
    click.echo(f"\nScheme:          {result.scheme_name}")
    click.echo(f"Scenario:        {scenario_label}")
    click.echo(f"lambda(S):       {result.lambda_bits:.1f} bits")
    click.echo(f"tau_hat(S):      {result.tau_bits:.1f} bits")
    click.echo(f"Margin M(S):     {result.margin_bits:+.1f} bits")
    click.echo(f"Assessment:      {result.interpretation}")


# ---------------------------------------------------------------------------
# regression subcommand
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--extrapolate", "-e",
    is_flag=True,
    default=False,
    help="Extrapolate to PQC-relevant lattice dimensions.",
)
def regression(extrapolate: bool) -> None:
    """Fit OLS regression models to SALSA data."""
    n_vals, cost_vals = get_salsa_data()
    linear = fit_linear_model(n_vals, cost_vals)
    quadratic = fit_quadratic_model(n_vals, cost_vals)

    click.echo("\nSALSA Regression Analysis")
    click.echo("=" * 60)

    click.echo("\nData points (n, log2 gate count):")
    for n, c in zip(n_vals, cost_vals):
        click.echo(f"  n = {n:>6.0f}  ->  {c:.1f}")

    click.echo(f"\nLinear model: cost = a * log2(n) + b")
    click.echo(f"  a = {linear.coefficients[0]:.4f}  (95% CI: [{linear.confidence_intervals_95[0][0]:.4f}, {linear.confidence_intervals_95[0][1]:.4f}])")
    click.echo(f"  b = {linear.coefficients[1]:.4f}  (95% CI: [{linear.confidence_intervals_95[1][0]:.4f}, {linear.confidence_intervals_95[1][1]:.4f}])")
    click.echo(f"  R-squared = {linear.r_squared:.6f}")
    click.echo(f"  Residual std = {linear.residual_std:.4f}")

    click.echo(f"\nQuadratic model: cost = a * log2(n)^2 + b * log2(n) + c")
    click.echo(f"  a = {quadratic.coefficients[0]:.6f}  (95% CI: [{quadratic.confidence_intervals_95[0][0]:.6f}, {quadratic.confidence_intervals_95[0][1]:.6f}])")
    click.echo(f"  b = {quadratic.coefficients[1]:.4f}  (95% CI: [{quadratic.confidence_intervals_95[1][0]:.4f}, {quadratic.confidence_intervals_95[1][1]:.4f}])")
    click.echo(f"  c = {quadratic.coefficients[2]:.4f}  (95% CI: [{quadratic.confidence_intervals_95[2][0]:.4f}, {quadratic.confidence_intervals_95[2][1]:.4f}])")
    click.echo(f"  R-squared = {quadratic.r_squared:.6f}")
    click.echo(f"  Residual std = {quadratic.residual_std:.4f}")

    comparison = compare_models()
    click.echo(f"\nRecommended model: {comparison['recommendation']}")

    if extrapolate:
        model = linear if comparison["recommendation"] == "linear" else quadratic
        ext = extrapolate_to_pqc_dimensions(model)
        click.echo(f"\nExtrapolation ({comparison['recommendation']} model):")
        for dim_label, cost in sorted(ext.items()):
            click.echo(f"  {dim_label}: {cost:.2f}")

    click.echo("=" * 60)


# ---------------------------------------------------------------------------
# sensitivity subcommand
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--scheme", "-s",
    type=click.Choice(["ML-KEM-512", "ML-KEM-768", "ML-KEM-1024"], case_sensitive=True),
    default=None,
    help="Scheme to analyse (default: all ML-KEM).",
)
def sensitivity(scheme: str | None) -> None:
    """Run sensitivity analysis on C and K parameters."""
    if scheme is not None:
        ps = get_scheme(scheme)
        results = [
            analyze_c_sensitivity(ps.name, ps.classical_security_bits, ps.lattice_dimension),
            analyze_k_sensitivity(ps.name, ps.classical_security_bits, ps.lattice_dimension),
        ]
    else:
        results = run_full_sensitivity_analysis()

    click.echo("\nSensitivity Analysis")
    click.echo("=" * 72)

    for r in results:
        click.echo(f"\n  {r.scheme_name} -- parameter {r.parameter_name}")
        click.echo(f"  Critical {r.parameter_name}*: ", nl=False)
        if r.critical_value is not None:
            if r.parameter_name == "K":
                click.echo(f"{r.critical_value:.2e}")
            else:
                click.echo(f"{r.critical_value:.4f}")
        else:
            click.echo("outside analysed range")

        click.echo(f"  Margin range: [{min(r.margin_values):.1f}, {max(r.margin_values):.1f}] bits")

    click.echo("\n" + "=" * 72)


# ---------------------------------------------------------------------------
# visualize subcommand
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--output-dir", "-o",
    type=click.Path(),
    default="output",
    show_default=True,
    help="Directory for generated plots.",
)
def visualize(output_dir: str) -> None:
    """Generate publication-quality plots."""
    from visualize import generate_all_plots

    out = Path(output_dir)
    generate_all_plots(out)
    click.echo(f"\nPlots saved to {out.resolve()}/")


# ---------------------------------------------------------------------------
# taxonomy subcommand
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--matrix", "-m",
    is_flag=True,
    default=False,
    help="Also display the migration decision matrix.",
)
def taxonomy(matrix: bool) -> None:
    """Display the quantum threat taxonomy."""
    click.echo()
    click.echo(format_taxonomy_report())

    if matrix:
        click.echo()
        click.echo(format_decision_matrix())


if __name__ == "__main__":
    cli()
