"""Posterior einer Ankunftsrate: Gamma–Poisson (geschlossen), Lognormal-Prior (Gitter), mehrere gekoppelte Raten (Tensorgitter).

Modell (ausgedacht, keine reale Datenquelle): Ankünfte in der Beobachtungsdauer T sind Poisson(λ·T); Zeit in Abfertigungsdauern.
    Prior Gamma(α, β) (Form α, Rate β)      → Posterior Gamma(α + n, β + T), exakt
    Prior ln λ ~ N(ln Median, s0²)          → kein Gamma, aber ein Gitter in u = ln λ genügt
    d Schichten mit je einer Rate λ_j       → Posterior in u = ln λ, Prior Gauß (Median 2, sd 0,5, Korrelation 0,7^|i−j|), Zählungen n_j in T = 5

Die Quadratur ist ein Gitter in u = ln λ auf Gipfel ± 6 Laplace-Standardabweichungen mit Trapezregel; die Referenz in den Tests ist `scipy.integrate.quad`.
Zähleinheit: Dichteauswertungen (ein Gitterpunkt = eine Auswertung)."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from scipy import stats

BOX_WIDTH = 6.0            # Kasten Gipfel ± 6 Laplace-sd (Gitter)
REF_WIDTH = 8.0            # Kasten der feinen Referenz im Ein-Parameter-Fall
REF_POINTS = 4001
DIM_DATA = (8, 6, 11, 9, 7, 10)      # Zählungen n_j der Schichten (ausgedacht)
DIM_T = 5.0
DIM_MEDIAN, DIM_SD, DIM_RHO = 2.0, 0.5, 0.7


# ------------------------------------------------------------------------------------------------------------------------------------------
# Gamma–Poisson (konjugat)
# ------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class GammaPosterior:
    shape: float
    rate: float

    @property
    def dist(self):
        return stats.gamma(self.shape, scale=1.0 / self.rate)

    @property
    def mean(self) -> float:
        return self.shape / self.rate

    @property
    def var(self) -> float:
        return self.shape / self.rate ** 2

    @property
    def sd(self) -> float:
        return math.sqrt(self.var)

    @property
    def mode(self) -> float:
        return max(self.shape - 1.0, 0.0) / self.rate

    def interval(self, level: float = 0.95) -> tuple[float, float]:
        lo = (1.0 - level) / 2.0
        return float(self.dist.ppf(lo)), float(self.dist.ppf(1.0 - lo))


def gamma_poisson_posterior(alpha: float, beta: float, n: int, t: float) -> GammaPosterior:
    """Prior Gamma(α, β), n Ankünfte in der Dauer T: Posterior Gamma(α + n, β + T)."""
    return GammaPosterior(alpha + n, beta + t)


def gamma_log_density(post: GammaPosterior):
    """Logarithmierte unnormierte Dichte von Gamma(Form, Rate) auf λ > 0 (−∞ sonst), als Feldfunktion für den Zufallsschritt-Metropolis."""
    def logp(x):
        return np.where(x > 0, (post.shape - 1.0) * np.log(np.maximum(x, 1e-300)) - post.rate * x, -np.inf)
    return logp


def sequential_posterior(alpha: float, beta: float, parts: list[tuple[int, float]]) -> GammaPosterior:
    """Teildaten (n_i, T_i) nacheinander verrechnen: der Posterior des einen Schritts ist der Prior des nächsten."""
    a, b = alpha, beta
    for n_i, t_i in parts:
        a, b = a + n_i, b + t_i
    return GammaPosterior(a, b)


def evidence(alpha: float, beta: float, n: int, t: float) -> float:
    """P(n Ankünfte in T) unter dem Gamma-Prior: negativ-binomial mit Erfolgswahrscheinlichkeit β/(β + T)."""
    return float(stats.nbinom(alpha, beta / (beta + t)).pmf(n))


def split_data(n: int, t: float) -> list[tuple[int, float]]:
    """Zwei Teile der Daten für den Umschalter „Reihenfolge der Daten egal“: erstes Drittel der Ankünfte in 7/20 der Dauer, der Rest in der übrigen Zeit."""
    n1 = n // 3
    t1 = 0.35 * t
    return [(n1, t1), (n - n1, t - t1)]


# ------------------------------------------------------------------------------------------------------------------------------------------
# Ein Parameter, beliebiger Prior (Gitter in u = ln λ)
# ------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class Prior1D:
    """Prior auf λ: ("gamma", α, β) oder ("lognormal", Median, s0)."""

    kind: str
    p1: float
    p2: float

    def log_density_u(self, u: np.ndarray) -> np.ndarray:
        """log-Prior in u = ln λ (inklusive Jacobi-Faktor), bis auf eine Konstante."""
        if self.kind == "gamma":
            return self.p1 * u - self.p2 * np.exp(u)
        if self.kind == "lognormal":
            return -((u - math.log(self.p1)) ** 2) / (2.0 * self.p2 ** 2)
        raise ValueError(self.kind)

    def grad_hess(self, u: float) -> tuple[float, float]:
        if self.kind == "gamma":
            return self.p1 - self.p2 * math.exp(u), -self.p2 * math.exp(u)
        return -(u - math.log(self.p1)) / self.p2 ** 2, -1.0 / self.p2 ** 2


def log_post_u(u: np.ndarray, n: int, t: float, prior: Prior1D) -> np.ndarray:
    """Logarithmierte unnormierte Posteriordichte in u = ln λ: n u − T e^u + log-Prior."""
    return n * u - t * np.exp(u) + prior.log_density_u(u)


def laplace_1d(n: int, t: float, prior: Prior1D) -> tuple[float, float]:
    """Gipfel u* und Laplace-Standardabweichung (Newton-Verfahren auf dem Gradienten)."""
    u = math.log(max(n, 1) / t)
    for _ in range(100):
        gp, hp = prior.grad_hess(u)
        g = n - t * math.exp(u) + gp
        h = -t * math.exp(u) + hp
        step = g / h
        u -= step
        if abs(step) < 1e-14:
            break
    gp, hp = prior.grad_hess(u)
    return u, math.sqrt(1.0 / (t * math.exp(u) - hp))


@dataclass(frozen=True)
class Moments1D:
    mean_u: float
    sd_u: float
    mean_lambda: float


def grid_moments_1d(n: int, t: float, prior: Prior1D, g: int, width: float = BOX_WIDTH) -> Moments1D:
    """Momente des Posteriors aus einem Gitter von `g` Punkten in u auf Gipfel ± width·Laplace-sd (Trapezregel; Zähleinheit: g Dichteauswertungen)."""
    u_star, sd_l = laplace_1d(n, t, prior)
    us = np.linspace(u_star - width * sd_l, u_star + width * sd_l, g)
    lp = log_post_u(us, n, t, prior)
    w = np.exp(lp - lp.max())
    w[0] *= 0.5
    w[-1] *= 0.5
    z = w.sum()
    mu = float((w * us).sum() / z)
    sd = math.sqrt(float((w * (us - mu) ** 2).sum() / z))
    ml = float((w * np.exp(us)).sum() / z)
    return Moments1D(mu, sd, ml)


def reference_moments_1d(n: int, t: float, prior: Prior1D) -> Moments1D:
    """Fein aufgelöste Referenz (4001 Punkte, Kasten ± 8 Laplace-sd); die Tests prüfen sie gegen `scipy.integrate.quad`."""
    return grid_moments_1d(n, t, prior, REF_POINTS, REF_WIDTH)


def grid_errors_in_sd(n: int, t: float, prior: Prior1D, g: int) -> tuple[float, float, float]:
    """Fehler des Gitters mit g Punkten gegen die Referenz: E[u] in sd, sd[u] relativ, E[λ] in sd[u]."""
    ref = reference_moments_1d(n, t, prior)
    m = grid_moments_1d(n, t, prior, g)
    return (abs(m.mean_u - ref.mean_u) / ref.sd_u, abs(m.sd_u - ref.sd_u) / ref.sd_u, abs(m.mean_lambda - ref.mean_lambda) / ref.sd_u)


def grid_sampler_start(n: int, t: float, prior: Prior1D, uniforms_r: np.ndarray) -> np.ndarray:
    """Startwerte im Gleichgewicht per Inversion auf dem feinen Gitter (Zufallszahlen von außen)."""
    u_star, sd_l = laplace_1d(n, t, prior)
    us = np.linspace(u_star - REF_WIDTH * sd_l, u_star + REF_WIDTH * sd_l, REF_POINTS)
    lp = log_post_u(us, n, t, prior)
    cdf = np.cumsum(np.exp(lp - lp.max()))
    cdf /= cdf[-1]
    return np.interp(uniforms_r, cdf, us)


# ------------------------------------------------------------------------------------------------------------------------------------------
# Mehrere gekoppelte Raten (Modell D)
# ------------------------------------------------------------------------------------------------------------------------------------------
@dataclass(frozen=True)
class DimModel:
    d: int
    counts: np.ndarray
    t: float
    m0: np.ndarray
    prec: np.ndarray          # Präzisionsmatrix des Gauß-Priors in u

    def log_post(self, u: np.ndarray) -> np.ndarray:
        """u: (..., d). log p(u) = Σ_j (n_j u_j − T e^{u_j}) − ½ (u − m0)ᵀ Σ0⁻¹ (u − m0), bis auf eine Konstante."""
        x = u - self.m0
        quad = np.einsum("...i,ij,...j->...", x, self.prec, x)
        return (self.counts * u - self.t * np.exp(u)).sum(axis=-1) - 0.5 * quad


def make_dim_model(d: int) -> DimModel:
    if not 1 <= d <= len(DIM_DATA):
        raise ValueError("d muss zwischen 1 und 6 liegen")
    idx = np.arange(d)
    cov0 = DIM_SD ** 2 * DIM_RHO ** np.abs(idx[:, None] - idx[None, :])
    return DimModel(d, np.array(DIM_DATA[:d], dtype=float), DIM_T, np.full(d, math.log(DIM_MEDIAN)), np.linalg.inv(cov0))


def laplace_dim(model: DimModel) -> tuple[np.ndarray, np.ndarray]:
    """Gipfel und Kovarianz der Laplace-Näherung (Newton)."""
    u = np.log(np.maximum(model.counts, 0.5) / model.t)          # Startpunkt auch für eine Schicht ohne Ankünfte (n_j = 0)
    for _ in range(100):
        g = model.counts - model.t * np.exp(u) - model.prec @ (u - model.m0)
        h = -np.diag(model.t * np.exp(u)) - model.prec
        step = np.linalg.solve(h, g)
        u = u - step
        if np.abs(step).max() < 1e-14:
            break
    h = -np.diag(model.t * np.exp(u)) - model.prec
    return u, np.linalg.inv(-h)


def tensor_grid_moments(model: DimModel, g: int, width: float = BOX_WIDTH) -> tuple[np.ndarray, np.ndarray]:
    """Randmomente E[u_j], sd[u_j] aus einem Tensorgitter mit g Punkten je Achse (g^d Dichteauswertungen) auf dem Kasten Gipfel ± width·Laplace-sd je Achse.

    Der Logarithmus der Dichte wird je Achse (lineare Terme) und je Achsenpaar (Prior-Kopplung) per Broadcasting aufgebaut."""
    d = model.d
    umode, cov = laplace_dim(model)
    sdl = np.sqrt(np.diag(cov))
    axes = [np.linspace(umode[j] - width * sdl[j], umode[j] + width * sdl[j], g) for j in range(d)]
    lp = np.zeros([g] * d)
    for j in range(d):
        sh = [1] * d
        sh[j] = g
        lp = lp + (model.counts[j] * axes[j] - model.t * np.exp(axes[j])).reshape(sh)
    xs = [axes[j] - model.m0[j] for j in range(d)]
    for i in range(d):
        for j in range(i, d):
            fac = -0.5 * model.prec[i, j] * (1.0 if i == j else 2.0)
            sh = [1] * d
            sh[i] = g
            if i == j:
                lp = lp + fac * (xs[i] ** 2).reshape(sh)
            else:
                sh[j] = g
                lp = lp + fac * np.outer(xs[i], xs[j]).reshape(sh)
    w = np.exp(lp - lp.max())
    z = w.sum()
    means = np.zeros(d)
    sds = np.zeros(d)
    for j in range(d):
        other = tuple(k for k in range(d) if k != j)
        marg = w.sum(axis=other) / z
        means[j] = float((marg * axes[j]).sum())
        sds[j] = math.sqrt(float((marg * (axes[j] - means[j]) ** 2).sum()))
    return means, sds
