"""Gate-Kette: ausrechnen statt abtasten (Zähleinheit Gleichungen).

Herkunft: neu geschrieben nach den Formeln in der README von markov-queue-demo (Zustand = Zahl der Lkw im System, K = Plätze INSGESAMT, Spuren
eingeschlossen; Abgangsrate min(n, c); uniformisierte Kette P = I + Q/Λ mit Λ = größte Austrittsrate). Es wird nichts aus der Nachbar-Demo importiert.

Alle Größen sind für die stationäre Kette exakt (Matrixrechnung), nicht geschätzt:
    π            stationäre Verteilung (vier Wege: geschlossene Form, lineares System, Eigenvektor, Matrixpotenz)
    d(t)         max_x TV(P^t(x, ·), π), t_mix = kleinstes t mit d(t) ≤ 1/4 (Levin/Peres/Wilmer, Kap. 4 und 12)
    σ²           asymptotische Varianz des Zeitmittels einer Anzeige f: σ² = 2 ⟨f~, Z f~⟩_π − ⟨f~, f~⟩_π mit Z = (I − P + 1π)⁻¹ (eigene Rechnung)
    ESS je Schritt = Var_π(f) / σ²   (1 = so gut wie unabhängige Ziehungen)
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

Z_95 = 1.96
T_MIX_MAX = 5000


def gate_rates(c: int, k: int, a: float) -> tuple[np.ndarray, np.ndarray]:
    """Aufstiegsrate up[n] (n < K, sonst 0) und Abstiegsrate dn[n] = min(n, c) (dn[0] = 0)."""
    up = np.array([a if n < k else 0.0 for n in range(k + 1)], dtype=float)
    dn = np.array([float(min(n, c)) for n in range(k + 1)], dtype=float)
    return up, dn


def uniformization_rate(c: int, k: int, a: float) -> float:
    """Λ = größte Austrittsrate max_n (up_n + dn_n) (wie in der Nachbar-Demo): gleich a + c, solange K > c; für K = c ist es a + c − 1 (Zustand K hat keine Aufstiegsrate)."""
    up, dn = gate_rates(c, k, a)
    return float((up + dn).max())


def generator(c: int, k: int, a: float) -> np.ndarray:
    up, dn = gate_rates(c, k, a)
    q = np.zeros((k + 1, k + 1))
    for n in range(k + 1):
        if n < k:
            q[n, n + 1] = up[n]
        if n > 0:
            q[n, n - 1] = dn[n]
        q[n, n] = -q[n].sum()
    return q


def uniformized(c: int, k: int, a: float) -> tuple[np.ndarray, float]:
    """P = I + Q/Λ mit Λ = größte Austrittsrate (= a + c für K > c); ein Schritt entspricht 1/Λ Abfertigungsdauern."""
    lam = uniformization_rate(c, k, a)
    return np.eye(k + 1) + generator(c, k, a) / lam, lam


def pi_closed(c: int, k: int, a: float) -> np.ndarray:
    """Geschlossene Form der Geburts-Sterbe-Kette: π_n ∝ Π_{j ≤ n} a / min(j, c)."""
    w = [1.0]
    for n in range(1, k + 1):
        w.append(w[-1] * a / min(n, c))
    w = np.array(w)
    return w / w.sum()


def pi_four_ways(c: int, k: int, a: float) -> dict[str, np.ndarray]:
    """π auf vier Wegen: geschlossene Form, lineares System (eine Gleichung durch die Normierung ersetzt), Eigenvektor zum Eigenwert 1, Matrixpotenz P^5000."""
    p, _ = uniformized(c, k, a)
    n = k + 1
    a_mat = p.T - np.eye(n)
    a_mat[0, :] = 1.0
    b = np.zeros(n)
    b[0] = 1.0
    lin = np.linalg.solve(a_mat, b)
    w, v = np.linalg.eig(p.T)
    j = int(np.argmax(w.real))
    eig = np.real(v[:, j])
    eig = eig / eig.sum()
    pot = np.linalg.matrix_power(p, 5000)[0]
    return {"geschlossen": pi_closed(c, k, a), "linear": lin, "eigenvektor": eig, "matrixpotenz": pot}


def max_spread(pis: dict[str, np.ndarray]) -> float:
    """Größte Abweichung eines Weges vom ersten (geschlossene Form)."""
    ref = next(iter(pis.values()))
    return float(max(np.abs(ref - x).max() for x in pis.values()))


def tv_curve(p: np.ndarray, pi: np.ndarray, t_max: int) -> list[float]:
    """d(t) für t = 0..t_max (schlechtester Start)."""
    out = []
    pt = np.eye(p.shape[0])
    for _ in range(t_max + 1):
        out.append(float(0.5 * np.abs(pt - pi[None, :]).sum(axis=1).max()))
        pt = pt @ p
    return out


def t_mix(p: np.ndarray, pi: np.ndarray, eps: float = 0.25, t_max: int = T_MIX_MAX) -> int | None:
    """Kleinstes t mit d(t) ≤ eps (None, wenn bis t_max nicht erreicht)."""
    pt = np.eye(p.shape[0])
    for t in range(t_max + 1):
        if 0.5 * np.abs(pt - pi[None, :]).sum(axis=1).max() <= eps:
            return t
        pt = pt @ p
    return None


def t_rel(p: np.ndarray) -> float:
    """Relaxationszeit 1/(1 − λ*) mit λ* = größter Betrag der Eigenwerte außer dem Eigenwert 1."""
    ev = np.sort(np.linalg.eigvals(p).real)
    lam_star = max(abs(ev[0]), abs(ev[-2]))
    return float(1.0 / (1.0 - lam_star))


def asymptotic_variance(p: np.ndarray, pi: np.ndarray, f: np.ndarray) -> tuple[float, float]:
    """(σ², Var_π(f)) für das Zeitmittel von f(X_t) der stationären Kette."""
    n = len(pi)
    z = np.linalg.inv(np.eye(n) - p + np.ones((n, 1)) @ pi[None, :])
    ft = f - float(pi @ f)
    var = float((pi * ft) @ ft)
    return float(2 * (pi * ft) @ (z @ ft) - var), var


def autocov_sum_variance(p: np.ndarray, pi: np.ndarray, f: np.ndarray, k_max: int = 3000) -> float:
    """Gegenprobe zu `asymptotic_variance`: σ² = Var + 2 Σ_k Cov(f(X_0), f(X_k)), bis k_max summiert."""
    ft = f - float(pi @ f)
    var = float((pi * ft) @ ft)
    s = 0.0
    v = ft.copy()
    for _ in range(k_max):
        v = p @ v
        s += float((pi * ft) @ v)
    return var + 2 * s


def steps_for_band(sigma2: float, eps: float, z: float = Z_95) -> float:
    """Schritte T, damit das 95-%-Band des Zeitmittels ±eps breit ist: T = z² σ² / eps² (asymptotisch)."""
    return z * z * sigma2 / (eps * eps)


def erlang_b(c: int, a: float) -> float:
    """Erlang-B-Formel (Rekursion B(0) = 1, B(m) = a B(m−1) / (m + a B(m−1))), unabhängig von der Kette."""
    b = 1.0
    for m in range(1, c + 1):
        b = a * b / (m + a * b)
    return b


@dataclass(frozen=True)
class GateMetrics:
    blocking: float
    mean_n: float
    mean_queue: float
    mean_wait: float          # Wartezeit der Angenommenen in Abfertigungsdauern (Little: Lq / (a (1 − π_K)))
    utilization: float        # E[min(n, c)] / c


def gate_metrics(c: int, k: int, a: float, pi: np.ndarray) -> GateMetrics:
    n = np.arange(k + 1)
    lq = float(np.sum(np.maximum(n - c, 0) * pi))
    accepted = a * (1.0 - float(pi[k]))
    return GateMetrics(float(pi[k]), float(n @ pi), lq, lq / accepted, float(np.sum(np.minimum(n, c) * pi) / c))


@dataclass(frozen=True)
class GateExact:
    """Alles, was sich für eine Gate-Kette ausrechnen lässt (eine Rechnung, Zähleinheit: K + 1 Gleichungen)."""

    c: int
    k: int
    a: float
    lam: float
    pi: np.ndarray
    pi_deviation: float
    equations: int
    t_mix: int | None
    t_rel: float
    sigma2_loss: float
    var_loss: float
    sigma2_n: float
    var_n: float

    @property
    def loss(self) -> float:
        return float(self.pi[-1])

    @property
    def ess_loss(self) -> float:
        return self.var_loss / self.sigma2_loss

    @property
    def ess_n(self) -> float:
        return self.var_n / self.sigma2_n

    @property
    def p(self) -> np.ndarray:
        return uniformized(self.c, self.k, self.a)[0]

    def steps_for(self, eps: float) -> float:
        return steps_for_band(self.sigma2_loss, eps)

    def band(self, t_steps: int) -> float:
        """Halbbreite des 95-%-Bandes für den Zeitanteil im Zustand K nach T Schritten: 1,96 √(σ²/T) (exakt, stationär)."""
        return Z_95 * math.sqrt(self.sigma2_loss / t_steps)


def exact_gate(c: int, k: int, a: float) -> GateExact:
    p, lam = uniformized(c, k, a)
    pis = pi_four_ways(c, k, a)
    pi = pis["geschlossen"]
    f_loss = np.zeros(k + 1)
    f_loss[k] = 1.0
    f_n = np.arange(k + 1, dtype=float)
    s2l, vl = asymptotic_variance(p, pi, f_loss)
    s2n, vn = asymptotic_variance(p, pi, f_n)
    return GateExact(c, k, a, lam, pi, max_spread(pis), k + 1, t_mix(p, pi), t_rel(p), s2l, vl, s2n, vn)
