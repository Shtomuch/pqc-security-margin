"""Quantum threat taxonomy and migration decision matrix.

Provides a structured classification of quantum threats to lattice-based
post-quantum cryptography, together with a decision matrix that maps
(threat severity, migration urgency) pairs to recommended actions.

The taxonomy follows the framework developed in Section 3 of the
accompanying research, distinguishing between:

- Algorithmic threats (improved classical/quantum lattice attacks)
- Implementation threats (side-channel, fault injection)
- Cryptanalytic threats (structural weaknesses in specific schemes)
- Quantum hardware threats (progress toward fault-tolerant quantum computers)

The decision matrix (Table 3.3) assigns each combination of threat level
and security margin category to a concrete migration recommendation.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

__all__ = [
    "ThreatCategory",
    "ThreatLevel",
    "MigrationUrgency",
    "ThreatEntry",
    "DecisionCell",
    "get_threat_taxonomy",
    "get_decision_matrix",
    "lookup_recommendation",
    "format_taxonomy_report",
    "format_decision_matrix",
]


class ThreatCategory(Enum):
    """Top-level categories of quantum threats to PQC schemes."""

    ALGORITHMIC = "Algorithmic"
    IMPLEMENTATION = "Implementation"
    CRYPTANALYTIC = "Cryptanalytic"
    QUANTUM_HARDWARE = "Quantum hardware"


class ThreatLevel(Enum):
    """Qualitative severity of a threat vector."""

    LOW = "Low"
    MEDIUM = "Medium"
    HIGH = "High"
    CRITICAL = "Critical"


class MigrationUrgency(Enum):
    """Urgency of cryptographic migration action."""

    MONITOR = "Monitor"
    PLAN = "Plan migration"
    EXECUTE = "Execute migration"
    EMERGENCY = "Emergency response"


@dataclass(frozen=True, slots=True)
class ThreatEntry:
    """Single entry in the threat taxonomy.

    Attributes:
        category: High-level threat category.
        name: Short identifier for the specific threat.
        description: One-sentence summary.
        level: Current estimated severity.
        time_horizon: Approximate time frame (e.g. "5--10 years").
        affected_schemes: Schemes primarily affected.
    """

    category: ThreatCategory
    name: str
    description: str
    level: ThreatLevel
    time_horizon: str
    affected_schemes: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class DecisionCell:
    """Single cell in the migration decision matrix.

    Attributes:
        margin_category: Security margin interpretation (row label).
        threat_level: Current threat severity (column label).
        urgency: Recommended migration urgency.
        action: Concrete recommended action.
    """

    margin_category: str
    threat_level: ThreatLevel
    urgency: MigrationUrgency
    action: str


# ---------------------------------------------------------------------------
# Threat taxonomy
# ---------------------------------------------------------------------------

_TAXONOMY: list[ThreatEntry] = [
    # Algorithmic
    ThreatEntry(
        category=ThreatCategory.ALGORITHMIC,
        name="Improved lattice sieving",
        description=(
            "Advances in sieving algorithms (e.g. lattice sieving with "
            "nearest-neighbour techniques) reduce the cost exponent for SVP."
        ),
        level=ThreatLevel.MEDIUM,
        time_horizon="5--15 years",
        affected_schemes=("ML-KEM", "ML-DSA"),
    ),
    ThreatEntry(
        category=ThreatCategory.ALGORITHMIC,
        name="ML-assisted cryptanalysis",
        description=(
            "Machine learning models accelerate lattice reduction or "
            "identify structural weaknesses in specific parameter sets."
        ),
        level=ThreatLevel.MEDIUM,
        time_horizon="3--10 years",
        affected_schemes=("ML-KEM", "ML-DSA"),
    ),
    ThreatEntry(
        category=ThreatCategory.ALGORITHMIC,
        name="Quantum lattice sieving",
        description=(
            "Grover-accelerated sieving on a fault-tolerant quantum computer "
            "achieves sqrt speedup over classical sieving."
        ),
        level=ThreatLevel.HIGH,
        time_horizon="10--20 years",
        affected_schemes=("ML-KEM", "ML-DSA"),
    ),
    # Implementation
    ThreatEntry(
        category=ThreatCategory.IMPLEMENTATION,
        name="Side-channel leakage",
        description=(
            "Timing, power, or electromagnetic side channels leak secret key "
            "material from unprotected implementations."
        ),
        level=ThreatLevel.HIGH,
        time_horizon="Immediate",
        affected_schemes=("ML-KEM", "ML-DSA"),
    ),
    ThreatEntry(
        category=ThreatCategory.IMPLEMENTATION,
        name="Fault injection",
        description=(
            "Induced faults during decapsulation or signing reveal secret "
            "key bits through differential fault analysis."
        ),
        level=ThreatLevel.MEDIUM,
        time_horizon="Immediate",
        affected_schemes=("ML-KEM", "ML-DSA"),
    ),
    # Cryptanalytic
    ThreatEntry(
        category=ThreatCategory.CRYPTANALYTIC,
        name="Algebraic structure exploitation",
        description=(
            "Discovery of exploitable algebraic structure in module lattices "
            "that reduces effective security below claimed levels."
        ),
        level=ThreatLevel.LOW,
        time_horizon="Unknown",
        affected_schemes=("ML-KEM", "ML-DSA"),
    ),
    ThreatEntry(
        category=ThreatCategory.CRYPTANALYTIC,
        name="Hybrid attack improvements",
        description=(
            "Combining lattice reduction with meet-in-the-middle or "
            "combinatorial techniques narrows the security margin."
        ),
        level=ThreatLevel.MEDIUM,
        time_horizon="5--10 years",
        affected_schemes=("ML-KEM-512", "ML-DSA-44"),
    ),
    # Quantum hardware
    ThreatEntry(
        category=ThreatCategory.QUANTUM_HARDWARE,
        name="Fault-tolerant quantum computer",
        description=(
            "A sufficiently large fault-tolerant quantum computer enables "
            "execution of quantum lattice sieving at cryptographic scale."
        ),
        level=ThreatLevel.HIGH,
        time_horizon="15--30 years",
        affected_schemes=("ML-KEM", "ML-DSA"),
    ),
    ThreatEntry(
        category=ThreatCategory.QUANTUM_HARDWARE,
        name="Rapid qubit scaling",
        description=(
            "Faster-than-expected growth in logical qubit counts shortens "
            "the timeline to cryptographically relevant quantum computers."
        ),
        level=ThreatLevel.MEDIUM,
        time_horizon="10--20 years",
        affected_schemes=("ML-KEM", "ML-DSA"),
    ),
]


# ---------------------------------------------------------------------------
# Decision matrix (Table 3.3)
# ---------------------------------------------------------------------------

_MARGIN_CATEGORIES = [
    "High (M > 40)",
    "Moderate (20 < M <= 40)",
    "Low (0 < M <= 20)",
    "Negative (M <= 0)",
]

_DECISION_MATRIX: list[DecisionCell] = [
    # High margin
    DecisionCell("High (M > 40)", ThreatLevel.LOW, MigrationUrgency.MONITOR,
                 "Continue monitoring; no immediate action required."),
    DecisionCell("High (M > 40)", ThreatLevel.MEDIUM, MigrationUrgency.MONITOR,
                 "Monitor threat evolution; review annually."),
    DecisionCell("High (M > 40)", ThreatLevel.HIGH, MigrationUrgency.PLAN,
                 "Develop migration roadmap; identify fallback schemes."),
    DecisionCell("High (M > 40)", ThreatLevel.CRITICAL, MigrationUrgency.PLAN,
                 "Accelerate migration planning; prepare hybrid deployment."),
    # Moderate margin
    DecisionCell("Moderate (20 < M <= 40)", ThreatLevel.LOW, MigrationUrgency.MONITOR,
                 "Routine monitoring sufficient."),
    DecisionCell("Moderate (20 < M <= 40)", ThreatLevel.MEDIUM, MigrationUrgency.PLAN,
                 "Begin migration planning; evaluate parameter upgrades."),
    DecisionCell("Moderate (20 < M <= 40)", ThreatLevel.HIGH, MigrationUrgency.EXECUTE,
                 "Execute migration to higher parameter sets."),
    DecisionCell("Moderate (20 < M <= 40)", ThreatLevel.CRITICAL, MigrationUrgency.EXECUTE,
                 "Immediate migration to higher security category."),
    # Low margin
    DecisionCell("Low (0 < M <= 20)", ThreatLevel.LOW, MigrationUrgency.PLAN,
                 "Plan parameter upgrade or scheme migration."),
    DecisionCell("Low (0 < M <= 20)", ThreatLevel.MEDIUM, MigrationUrgency.EXECUTE,
                 "Execute migration; deploy hybrid schemes."),
    DecisionCell("Low (0 < M <= 20)", ThreatLevel.HIGH, MigrationUrgency.EXECUTE,
                 "Urgent migration required; activate contingency plan."),
    DecisionCell("Low (0 < M <= 20)", ThreatLevel.CRITICAL, MigrationUrgency.EMERGENCY,
                 "Emergency response; switch to alternative PQC family."),
    # Negative margin
    DecisionCell("Negative (M <= 0)", ThreatLevel.LOW, MigrationUrgency.EXECUTE,
                 "Migrate to higher parameter set immediately."),
    DecisionCell("Negative (M <= 0)", ThreatLevel.MEDIUM, MigrationUrgency.EMERGENCY,
                 "Emergency migration; consider scheme replacement."),
    DecisionCell("Negative (M <= 0)", ThreatLevel.HIGH, MigrationUrgency.EMERGENCY,
                 "Critical: replace scheme; deploy symmetric fallback."),
    DecisionCell("Negative (M <= 0)", ThreatLevel.CRITICAL, MigrationUrgency.EMERGENCY,
                 "System compromised; halt sensitive operations; full remediation."),
]


def get_threat_taxonomy() -> list[ThreatEntry]:
    """Return the full quantum threat taxonomy."""
    return list(_TAXONOMY)


def get_decision_matrix() -> list[DecisionCell]:
    """Return all cells of the migration decision matrix."""
    return list(_DECISION_MATRIX)


def lookup_recommendation(
    margin_bits: float,
    threat_level: ThreatLevel,
) -> DecisionCell:
    """Look up the recommended action for a given margin and threat level.

    Args:
        margin_bits: Security margin M(S) in bits.
        threat_level: Current assessed threat severity.

    Returns:
        The matching ``DecisionCell`` from the decision matrix.
    """
    if margin_bits > 40:
        cat = "High (M > 40)"
    elif margin_bits > 20:
        cat = "Moderate (20 < M <= 40)"
    elif margin_bits > 0:
        cat = "Low (0 < M <= 20)"
    else:
        cat = "Negative (M <= 0)"

    for cell in _DECISION_MATRIX:
        if cell.margin_category == cat and cell.threat_level == threat_level:
            return cell

    raise ValueError(f"No matrix entry for margin={cat!r}, threat={threat_level!r}")


def format_taxonomy_report() -> str:
    """Format the threat taxonomy as a plain-text table for terminal output."""
    lines: list[str] = []
    lines.append("=" * 90)
    lines.append("QUANTUM THREAT TAXONOMY FOR LATTICE-BASED PQC")
    lines.append("=" * 90)

    header = f"{'Category':<20} {'Threat':<28} {'Level':<10} {'Horizon':<15} {'Schemes'}"
    lines.append(header)
    lines.append("-" * 90)

    for entry in _TAXONOMY:
        schemes = ", ".join(entry.affected_schemes)
        lines.append(
            f"{entry.category.value:<20} "
            f"{entry.name:<28} "
            f"{entry.level.value:<10} "
            f"{entry.time_horizon:<15} "
            f"{schemes}"
        )

    lines.append("=" * 90)
    return "\n".join(lines)


def format_decision_matrix() -> str:
    """Format the migration decision matrix as a plain-text table.

    Rows: security margin categories.  Columns: threat levels.
    Each cell shows the recommended urgency.
    """
    levels = [ThreatLevel.LOW, ThreatLevel.MEDIUM, ThreatLevel.HIGH, ThreatLevel.CRITICAL]
    col_width = 18

    lines: list[str] = []
    lines.append("=" * 100)
    lines.append("MIGRATION DECISION MATRIX (Table 3.3)")
    lines.append("=" * 100)

    # Header row
    header = f"{'Margin \\ Threat':<25}"
    for lvl in levels:
        header += f"{lvl.value:^{col_width}}"
    lines.append(header)
    lines.append("-" * 100)

    # Data rows
    for cat in _MARGIN_CATEGORIES:
        row = f"{cat:<25}"
        for lvl in levels:
            cell = lookup_recommendation(
                margin_bits=_margin_category_to_value(cat),
                threat_level=lvl,
            )
            row += f"{cell.urgency.value:^{col_width}}"
        lines.append(row)

    lines.append("=" * 100)
    return "\n".join(lines)


def _margin_category_to_value(cat: str) -> float:
    """Map a margin category label to a representative numeric value."""
    if "High" in cat:
        return 50.0
    if "Moderate" in cat:
        return 30.0
    if "Low" in cat:
        return 10.0
    return -10.0
