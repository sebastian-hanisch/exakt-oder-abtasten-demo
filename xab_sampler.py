"""Abtasten: Gate-Kette, direkte Gamma-Ziehung und Zufallsschritt-Metropolis mit festem Schritt (Zähleinheiten Schritte und Ziehungen).

Zufall nur aus dem ganzzahligen SplitMix64 (`xab_rng.py`, Kopie der Vorlage der Website). Die Zufallsfolge wird hier mit numpy-Ganzzahlen (uint64, Überlauf
rechnet modulo 2^64) vektorisiert: `uniforms(seed, n, offset)` liefert dieselben Zahlen wie `n` Aufrufe von `SplitMix64(seed).random()` nach `offset` Aufrufen
(Test `tests/test_sampler.py`). Die Ganzzahl- und Gleichverteilungsfolge ist damit auf jeder Plattform bitgleich; `np.log`, `np.cos` und `np.exp` in den
Ketten sind es in der letzten Gleitkommastelle nicht, Zufallsanteile werden darum nur als Band mit Standardfehler behauptet.

Der Zufallsschritt-Metropolis ist hier ein Messobjekt (fester, runder Schritt), kein Lehrgegenstand; Verfahren und Vorschlagsfallen: Nachbar-Demo
`posterior-sampling-demo` (stetig, 48-dimensional).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

import xab_chain as CH
import xab_rng

_GOLD = np.uint64(0x9E3779B97F4A7C15)
_M1 = np.uint64(0xBF58476D1CE4E5B9)
_M2 = np.uint64(0x94D049BB133111EB)
_S30, _S27, _S31, _S11 = np.uint64(30), np.uint64(27), np.uint64(31), np.uint64(11)
_TWO53 = float(1 << 53)
_CHUNK_VALUES = 600_000     # so viele Zufallszahlen höchstens auf einmal im Speicher


def uniforms(seed: int, n: int, offset: int = 0) -> np.ndarray:
    """Die Zufallszahlen offset+1 … offset+n von SplitMix64(seed), gleichverteilt in [0, 1) mit 53 Bit (wie `SplitMix64.random`)."""
    k = np.arange(offset + 1, offset + n + 1, dtype=np.uint64)
    z = np.uint64(seed & xab_rng.MASK) + _GOLD * k
    z = (z ^ (z >> _S30)) * _M1
    z = (z ^ (z >> _S27)) * _M2
    z = z ^ (z >> _S31)
    return (z >> _S11).astype(np.float64) / _TWO53


def box_muller(u1: np.ndarray, u2: np.ndarray) -> np.ndarray:
    """Standardnormalzahlen aus je zwei Gleichverteilten (wie `SplitMix64.normal`, aber mit numpy-Funktionen)."""
    return np.sqrt(-2.0 * np.log(np.maximum(u1, 1e-12))) * np.cos(2.0 * math.pi * u2)


# ------------------------------------------------------------------------------------------------------------------------------------------
# Gate-Kette
# ------------------------------------------------------------------------------------------------------------------------------------------
def gate_thresholds(c: int, k: int, a: float) -> tuple[list[float], list[float]]:
    """Schwellen der uniformisierten Kette: u < up[s] → s + 1; u ≥ dn[s] → s − 1; sonst bleibt der Zustand (Aufstieg a/Λ, Abstieg min(s, c)/Λ, Λ = größte Austrittsrate)."""
    lam = CH.uniformization_rate(c, k, a)
    up = [a / lam if s < k else 0.0 for s in range(k + 1)]
    dn = [1.0 - min(s, c) / lam for s in range(k + 1)]
    return up, dn


def simulate_gate(c: int, k: int, a: float, t_steps: int, r: int, seed: int, start: int | np.ndarray = 0) -> np.ndarray:
    """`r` Ketten zu je `t_steps` Schritten (Kette i nimmt die Zufallszahlen i·T+1 … (i+1)·T); gibt je Kette den Zeitanteil im Zustand K zurück.

    Jeder Schritt zieht eine Zahl u: u < a/Λ steigt, u ≥ 1 − min(s, c)/Λ fällt, sonst bleibt die Kette (Zeile von P = I + Q/Λ). Gezählt werden die Schritte,
    nach denen die Kette im Zustand K steht; der Start zählt nicht. Reine Python-Schleife (ein Schritt etwa 0,3 µs)."""
    up, dn = gate_thresholds(c, k, a)
    starts = np.broadcast_to(np.asarray(start, dtype=int), (r,))
    out = np.empty(r)
    for i in range(r):
        s = int(starts[i])
        cnt = 0
        for u in uniforms(seed, t_steps, i * t_steps).tolist():
            if u < up[s]:
                s += 1
            elif u >= dn[s]:
                s -= 1
            if s == k:
                cnt += 1
        out[i] = cnt / t_steps
    return out


def simulate_gate_states(c: int, k: int, a: float, t_steps: int, r: int, seed: int, start: int = 0) -> np.ndarray:
    """Wie `simulate_gate`, gibt aber den Zustand nach `t_steps` Schritten je Kette zurück (für Vergleiche mit P^t)."""
    up, dn = gate_thresholds(c, k, a)
    out = np.empty(r, dtype=int)
    for i in range(r):
        s = start
        for u in uniforms(seed, t_steps, i * t_steps).tolist():
            if u < up[s]:
                s += 1
            elif u >= dn[s]:
                s -= 1
        out[i] = s
    return out


# ------------------------------------------------------------------------------------------------------------------------------------------
# Direkte Ziehung aus Gamma(Form ganz, Rate)
# ------------------------------------------------------------------------------------------------------------------------------------------
def gamma_direct(shape: int, rate: float, n_draws: int, seed: int) -> np.ndarray:
    """Unabhängige Ziehungen aus Gamma(shape, rate) für ganze Form: Summe von `shape` Exponentialzahlen = −ln(Produkt von `shape` Zufallszahlen)/rate.

    Ein `exact_log` (decimal, plattformfest) je Ziehung; die Vorlage hat keinen Gamma-Sampler, nichtganze Form wird nicht angeboten."""
    if int(shape) != shape or shape < 1:
        raise ValueError("Form muss eine ganze Zahl ≥ 1 sein")
    shape = int(shape)
    u = 1.0 - uniforms(seed, n_draws * shape).reshape(n_draws, shape)    # (0, 1]
    prod = np.prod(u, axis=1)
    return np.array([-xab_rng.exact_log(float(x)) / rate for x in prod])


# ------------------------------------------------------------------------------------------------------------------------------------------
# Zufallsschritt-Metropolis mit festem Schritt
# ------------------------------------------------------------------------------------------------------------------------------------------
@dataclass
class ChainRun:
    x_final: np.ndarray
    accept_rate: float
    means_at: dict[int, np.ndarray] = field(default_factory=dict)     # laufendes Mittel je Kette nach N Schritten
    trace: np.ndarray | None = None                                    # Verlauf der ersten Kette


def rwm_1d(logp, x0: np.ndarray, step: float, n_steps: int, seed: int, lower: float | None = None, checkpoints=(), keep_trace: bool = False) -> ChainRun:
    """Zufallsschritt-Metropolis in einer Dimension, vektorisiert über die Ketten `x0`; Vorschlag x + step·N(0, 1), Annahme mit u3 (drei Zahlen je Schritt und Kette).

    `logp(x)` ist die logarithmierte unnormierte Dichte (Feldargument); `lower` verwirft Vorschläge ≤ lower (Träger λ > 0). Rückgabe: Endzustand, Annahmerate,
    laufendes Mittel je Kette an den Kontrollpunkten und (optional) der Verlauf der ersten Kette."""
    x = np.array(x0, dtype=float)
    r = x.size
    lp = logp(x)
    run = np.zeros(r)
    acc = 0.0
    cps = set(checkpoints)
    out: dict[int, np.ndarray] = {}
    trace = np.empty(n_steps) if keep_trace else None
    chunk = max(1, _CHUNK_VALUES // (3 * r))
    for t0 in range(0, n_steps, chunk):
        m = min(chunk, n_steps - t0)
        u = uniforms(seed, 3 * r * m, 3 * r * t0).reshape(m, 3, r)
        z = box_muller(u[:, 0, :], u[:, 1, :])
        lu = np.log(np.maximum(u[:, 2, :], 1e-300))
        for i in range(m):
            prop = x + step * z[i]
            lpp = logp(prop)
            ok = lu[i] < lpp - lp
            if lower is not None:
                ok &= prop > lower
            x = np.where(ok, prop, x)
            lp = np.where(ok, lpp, lp)
            acc += float(ok.mean())
            run += x
            if keep_trace:
                trace[t0 + i] = x[0]
            if (t0 + i + 1) in cps:
                out[t0 + i + 1] = run / (t0 + i + 1)
    return ChainRun(x, acc / n_steps, out, trace)


def rwm_multi(logp, x0: np.ndarray, chol: np.ndarray, n_steps: int, seed: int, checkpoints=()) -> ChainRun:
    """Zufallsschritt-Metropolis in d Dimensionen: Vorschlag x + L z mit z ~ N(0, I_d) und fester Matrix `chol` (= Schritt · Cholesky-Faktor der Vorschlagskovarianz).

    `logp(x)` bekommt ein Feld (Ketten, d) und gibt (Ketten,) zurück. Je Schritt und Kette werden 2d + 1 Zufallszahlen gezogen."""
    x = np.array(x0, dtype=float)
    r, d = x.shape
    lp = logp(x)
    run = np.zeros_like(x)
    acc = 0.0
    cps = set(checkpoints)
    out: dict[int, np.ndarray] = {}
    per = 2 * d + 1
    chunk = max(1, _CHUNK_VALUES // (per * r))
    for t0 in range(0, n_steps, chunk):
        m = min(chunk, n_steps - t0)
        u = uniforms(seed, per * r * m, per * r * t0).reshape(m, r, per)
        z = box_muller(u[:, :, :d], u[:, :, d:2 * d])                  # (m, r, d)
        lu = np.log(np.maximum(u[:, :, 2 * d], 1e-300))
        steps = z @ chol.T
        for i in range(m):
            prop = x + steps[i]
            lpp = logp(prop)
            ok = lu[i] < lpp - lp
            x = np.where(ok[:, None], prop, x)
            lp = np.where(ok, lpp, lp)
            acc += float(ok.mean())
            run += x
            if (t0 + i + 1) in cps:
                out[t0 + i + 1] = run / (t0 + i + 1)
    return ChainRun(x, acc / n_steps, out, None)


def ess_per_step_from_rmse(rmse_in_sd: float, n_steps: int) -> float:
    """ESS je Schritt aus dem gemessenen RMSE des Kettenmittels in Einheiten der Standardabweichung des Ziels: (1/RMSE)² / N (unabhängige Ziehungen: 1)."""
    return (1.0 / rmse_in_sd) ** 2 / n_steps


def rmse_in_sd(means: np.ndarray, truth: float | np.ndarray, sd: float | np.ndarray) -> float:
    """RMSE der Kettenmittel `means` gegen den exakten Wert, in Einheiten von sd (bei mehreren Koordinaten über Ketten und Koordinaten gemittelt)."""
    return float(np.sqrt(np.mean(((means - truth) / sd) ** 2)))
