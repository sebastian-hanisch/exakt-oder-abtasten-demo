"""Vorgerechnete Messungen laden: `xab_dimension.json` (Experiment 4) und `xab_messungen.json` (Experimente 1 bis 3, je 400 Ketten).

Beide Dateien kommen aus `tools/xab_make_dimension.py` bzw. `tools/xab_make_messungen.py` (mit `--check`); die App rechnet sie nicht live (viele Ketten über
64 000 Schritte). In App und README stehen sie als „vorgerechnet“."""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

ROOT = Path(__file__).resolve().parent


@lru_cache(maxsize=None)
def dimension() -> dict:
    return json.loads((ROOT / "xab_dimension.json").read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def messungen() -> dict:
    return json.loads((ROOT / "xab_messungen.json").read_text(encoding="utf-8"))


def dimension_row(d: int, run: str = "iso_a") -> dict:
    """Eine Zeile der Tabelle für d Raten: Gitter und Kette, ESS je Schritt und Kettenschritte für 0,01 sd (10 000 / ESS)."""
    e = dimension()["d"][str(d)]
    ch = e["ketten"][run]
    return {
        "d": d,
        "grid_error": {int(g): tuple(v) for g, v in e["gitter_fehler"].items()},
        "ess_per_step": ch["ess_per_step"],
        "accept": ch["accept"],
        "rmse": {int(t): v for t, v in ch["rmse_in_sd"].items()},
        "chain_steps_1e2": 1e4 / ch["ess_per_step"],
    }


def grid_points_needed(d: int, g_needed: int = 10) -> int:
    return g_needed ** d


def crossing_dimension(run: str = "iso_a", g: int = 10, accuracy_sd: float = 0.01, ess_extrapolated_d1: float | None = None, d_max: int = 12) -> int | None:
    """Erste Dimension d, in der das Gitter (g^d Auswertungen) mehr kostet als die Kette (N = 1/(accuracy² · ESS je Schritt)).

    Bis d = 6 aus den gemessenen ESS je Schritt; darüber Hochrechnung mit ESS je Schritt ≈ Mittel(ESS · d)/d über d = 1..6 (nicht gemessen)."""
    meas = {d: dimension()["d"][str(d)]["ketten"][run]["ess_per_step"] for d in range(1, 7)}
    mean_ess_d = sum(meas[d] * d for d in meas) / len(meas)
    for d in range(1, d_max + 1):
        ess = meas[d] if d in meas else mean_ess_d / d
        if g ** d > 1.0 / (accuracy_sd ** 2 * ess):
            return d
    return None
