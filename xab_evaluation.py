"""Auswertung: die vier Experimente als Rechnungen mit Zähleinheit (Gleichungen, Dichteauswertungen, Schritte, Ziehungen) und die Verdikte als reine Funktionen.

Ein Verdikt ist eine reine Funktion `verdict_*(...) -> (Art, Text)`; die App zeigt es mit `getattr(st, Art)(Text)`, `tests/test_verdicts.py` prüft Schwellen und jede
Zahl im Text an handgesetzten Werten. Alle Aussagen gelten nur für die hier gerechneten ausgedachten Modelle (Poisson-Zählungen, genannte Priors, Gitter um den Gipfel,
Zufallsschritt-Metropolis mit festem Schritt); Zähleinheit steht jeweils dabei."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache

import numpy as np
from scipy import stats

import xab_chain as CH
import xab_constants as C
import xab_format as F
import xab_posterior as PO
import xab_reference as REF
import xab_sampler as S


@dataclass(frozen=True)
class Settings:
    seed: int = C.DEFAULTS["seed"]
    c: int = C.DEFAULTS["c"]
    k: int = C.DEFAULTS["k"]
    last: int = C.DEFAULTS["last"]
    t_steps: int = C.DEFAULTS["t_steps"]
    r_chains: int = C.DEFAULTS["r_chains"]
    eps_pp: float = C.DEFAULTS["eps_pp"]
    alpha: int = C.DEFAULTS["alpha"]
    beta: float = C.DEFAULTS["beta"]
    n_obs: int = C.DEFAULTS["n_obs"]
    t_obs: float = C.DEFAULTS["t_obs"]
    gp_step: float = C.DEFAULTS["gp_step"]
    gp_n_steps: int = C.DEFAULTS["gp_n_steps"]
    gp_reps: int = C.DEFAULTS["gp_reps"]
    gp_split: bool = C.DEFAULTS["gp_split"]
    prior: str = C.DEFAULTS["prior"]
    median: float = C.DEFAULTS["median"]
    s0: float = C.DEFAULTS["s0"]
    grid_g: int = C.DEFAULTS["grid_g"]
    ln_n_steps: int = C.DEFAULTS["ln_n_steps"]
    ln_reps: int = C.DEFAULTS["ln_reps"]
    dim_d: int = C.DEFAULTS["dim_d"]
    dim_g: int = C.DEFAULTS["dim_g"]
    dim_acc: str = C.DEFAULTS["dim_acc"]


def sub_seed(seed: int, k: int) -> int:
    """Unabhängige Zufallsfolgen aus einem Zufalls-Seed: je Verwendung ein eigener Block."""
    return seed + k * 1_000_003


# ==========================================================================================================================================
# Experiment 1: ein Gate, ausrechnen oder laufen lassen
# ==========================================================================================================================================
def gate_offer(c: int, last: int) -> float:
    """Angebot a in Erlang: Last in Prozent mal Zahl der Spuren (Abfertigungsdauer 1)."""
    return last / 100.0 * c


@lru_cache(maxsize=64)
def gate_exact(c: int, k: int, last: int) -> CH.GateExact:
    return CH.exact_gate(c, k, gate_offer(c, last))


def sim_allowed(r: int, t: int) -> bool:
    return r * t <= C.SIM_CAP


@lru_cache(maxsize=16)
def _gate_sim(c: int, k: int, last: int, t_steps: int, r: int, seed: int) -> tuple[float, ...]:
    return tuple(S.simulate_gate(c, k, gate_offer(c, last), t_steps, r, seed, 0).tolist())


@dataclass(frozen=True, eq=False)
class GateRun:
    exact: CH.GateExact
    metrics: CH.GateMetrics
    t_need: float                      # Schritte für ±eps (95 %), exakt aus σ²
    eps: float                         # als Anteil (0,005 für 0,5 Prozentpunkte)
    t_steps: int
    r_chains: int
    shares: np.ndarray | None          # simulierter Zeitanteil im Zustand K je Kette (None, wenn R·T über der Obergrenze)
    band: float                        # 1,96 √(σ²/T)
    coverage: float | None             # Anteil der Ketten innerhalb von ±eps um den exakten Verlust
    erlang_b: float | None             # nur wenn K = c (kein Warteplatz)


def run_gate(s: Settings) -> GateRun:
    ex = gate_exact(s.c, s.k, s.last)
    eps = s.eps_pp / 100.0
    shares = None
    cov = None
    if sim_allowed(s.r_chains, s.t_steps):
        shares = np.array(_gate_sim(s.c, s.k, s.last, s.t_steps, s.r_chains, s.seed))
        cov = float(np.mean(np.abs(shares - ex.loss) <= eps))
    return GateRun(ex, CH.gate_metrics(s.c, s.k, ex.a, ex.pi), ex.steps_for(eps), eps, s.t_steps, s.r_chains, shares, ex.band(s.t_steps), cov,
                   CH.erlang_b(s.c, ex.a) if s.k == s.c else None)


def frozen_coverage(c: int, k: int, last: int, eps_pp: float) -> list[dict] | None:
    """Vorgerechnete Überdeckung (400 Ketten) für die Voreinstellung 2 Spuren, 6 Plätze, Last 80 % und ε = 1,0 bzw. 0,5 Prozentpunkte, sonst None."""
    if (c, k, last) != (2, 6, 80):
        return None
    rows = [r for r in REF.messungen()["gate_ueberdeckung"] if abs(r["eps"] * 100 - eps_pp) < 1e-9]
    return rows or None


# ==========================================================================================================================================
# Experiment 2: Gamma–Poisson
# ==========================================================================================================================================
@dataclass(frozen=True, eq=False)
class GammaRun:
    post: PO.GammaPosterior
    seq_post: PO.GammaPosterior | None
    evidence: float
    draws: np.ndarray
    draws_mean: float
    draws_sd: float
    ks_d: float
    ks_p: float
    trace: np.ndarray                  # Verlauf der ersten Kette
    chain_step_sd: float
    accept: float
    ess: float                         # ESS je Schritt aus R Ketten (Start aus der exakten Verteilung)
    rmse_sd: float
    chain_steps_001: float             # Kettenschritte für RMSE 0,01 sd = 10 000 / ESS
    table: tuple                       # ((Schritt in sd, Annahmerate, ESS je Schritt), …)


@lru_cache(maxsize=16)
def _gamma_core(alpha: int, beta: float, n_obs: int, t_obs: float, step_sd: float, n_steps: int, reps: int, seed: int):
    post = PO.gamma_poisson_posterior(alpha, beta, n_obs, t_obs)
    shape = int(post.shape)
    draws = S.gamma_direct(shape, post.rate, C.DIRECT_DRAWS, sub_seed(seed, 1))
    ks = stats.kstest(draws, post.dist.cdf)
    x0 = S.gamma_direct(shape, post.rate, reps, sub_seed(seed, 2))
    logp = PO.gamma_log_density(post)
    run = S.rwm_1d(logp, x0, step_sd * post.sd, n_steps, sub_seed(seed, 3), lower=0.0, checkpoints=(n_steps,), keep_trace=True)
    rmse = S.rmse_in_sd(run.means_at[n_steps], post.mean, post.sd)
    rows = []
    r_tab = min(reps, C.STEP_GRID_R)
    for i, st_sd in enumerate(C.STEP_GRID_SD):
        rt = S.rwm_1d(logp, x0[:r_tab], st_sd * post.sd, C.STEP_GRID_N, sub_seed(seed, 10 + i), lower=0.0, checkpoints=(C.STEP_GRID_N,))
        rows.append((st_sd, rt.accept_rate, S.ess_per_step_from_rmse(S.rmse_in_sd(rt.means_at[C.STEP_GRID_N], post.mean, post.sd), C.STEP_GRID_N)))
    return post, draws, ks, run, rmse, tuple(rows)


def run_gamma(s: Settings) -> GammaRun:
    post, draws, ks, run, rmse, table = _gamma_core(s.alpha, s.beta, s.n_obs, s.t_obs, s.gp_step, s.gp_n_steps, s.gp_reps, s.seed)
    ess = S.ess_per_step_from_rmse(rmse, s.gp_n_steps)
    seq = PO.sequential_posterior(s.alpha, s.beta, PO.split_data(s.n_obs, s.t_obs)) if s.gp_split else None
    return GammaRun(post, seq, PO.evidence(s.alpha, s.beta, s.n_obs, s.t_obs), draws, float(draws.mean()), float(draws.std(ddof=1)), float(ks.statistic),
                    float(ks.pvalue), run.trace, s.gp_step, run.accept_rate, ess, rmse, 1e4 / ess, table)


def frozen_gamma_table(alpha: int, beta: float, n: int, t: float) -> list[dict] | None:
    """Vorgerechnete ESS je Schritt (400 Ketten) für den Posterior Gamma(34, 21) (Prior Gamma(2, 1), 32 Ankünfte in 20), sonst None."""
    if (alpha, beta, n, t) != (2, 1.0, 32, 20.0):
        return None
    return REF.messungen()["gamma_ketten"]


# ==========================================================================================================================================
# Experiment 3: nichtkonjugater Prior, ein Gitter genügt
# ==========================================================================================================================================
def make_prior(s: Settings) -> PO.Prior1D:
    if s.prior == C.PRIOR_LOGNORMAL:
        return PO.Prior1D("lognormal", s.median, s.s0)
    return PO.Prior1D("gamma", C.GAMMA_PRIOR[0], C.GAMMA_PRIOR[1])


@dataclass(frozen=True, eq=False)
class GridRun:
    prior: PO.Prior1D
    ref: PO.Moments1D
    conj_mean: float                   # Mittel von λ im konjugaten Fall Gamma(2, 1) (Gamma(34, 21))
    grid: PO.Moments1D
    err: tuple[float, float, float]    # E[u] in sd, sd[u] relativ, E[λ] in sd
    errs_by_g: tuple                   # ((G, (e1, e2, e3)), …) für die Fehlerkurve
    g_need: int | None                 # kleinstes G der Liste mit allen Fehlern < 0,01 sd
    ess: float
    rmse_sd: float
    accept: float
    chain_steps_001: float
    lam: np.ndarray                    # Dichtekurve in λ (Referenz)
    dens: np.ndarray
    grid_lam: np.ndarray               # Auswertungspunkte des Gitters
    grid_dens: np.ndarray
    conj_dens: np.ndarray


@lru_cache(maxsize=16)
def _grid_chain(kind: str, p1: float, p2: float, n_steps: int, reps: int, seed: int):
    prior = PO.Prior1D(kind, p1, p2)
    n, t = C.LN_N_OBS, C.LN_T_OBS
    ref = PO.reference_moments_1d(n, t, prior)
    x0 = PO.grid_sampler_start(n, t, prior, S.uniforms(sub_seed(seed, 4), reps))
    run = S.rwm_1d(lambda x: PO.log_post_u(x, n, t, prior), x0, C.LN_CHAIN_STEP_SD * ref.sd_u, n_steps, sub_seed(seed, 5), checkpoints=(n_steps,))
    rmse = S.rmse_in_sd(run.means_at[n_steps], ref.mean_u, ref.sd_u)
    return ref, run.accept_rate, rmse


def run_grid(s: Settings) -> GridRun:
    prior = make_prior(s)
    n, t = C.LN_N_OBS, C.LN_T_OBS
    ref, accept, rmse = _grid_chain(prior.kind, prior.p1, prior.p2, s.ln_n_steps, s.ln_reps, s.seed)
    grid = PO.grid_moments_1d(n, t, prior, s.grid_g)
    err = (abs(grid.mean_u - ref.mean_u) / ref.sd_u, abs(grid.sd_u - ref.sd_u) / ref.sd_u, abs(grid.mean_lambda - ref.mean_lambda) / ref.sd_u)
    errs = tuple((g, PO.grid_errors_in_sd(n, t, prior, g)) for g in C.LN_G_LIST)
    ok = [g for g, e in errs if max(e) < 0.01]
    ess = S.ess_per_step_from_rmse(rmse, s.ln_n_steps)
    u_star, sd_l = PO.laplace_1d(n, t, prior)
    us = np.linspace(u_star - 6 * sd_l, u_star + 6 * sd_l, 400)
    lp = PO.log_post_u(us, n, t, prior)
    lam = np.exp(us)
    dens = np.exp(lp - lp.max()) / lam
    norm = np.trapezoid(dens, lam)
    g_us = np.linspace(u_star - PO.BOX_WIDTH * sd_l, u_star + PO.BOX_WIDTH * sd_l, s.grid_g)
    g_lp = PO.log_post_u(g_us, n, t, prior)
    g_lam = np.exp(g_us)
    conj = PO.gamma_poisson_posterior(C.GAMMA_PRIOR[0], C.GAMMA_PRIOR[1], n, t)
    return GridRun(prior, ref, conj.mean, grid, err, errs, min(ok) if ok else None, ess, rmse, accept, 1e4 / ess, lam, dens / norm, g_lam,
                   np.exp(g_lp - lp.max()) / g_lam / norm, conj.dist.pdf(lam))


# ==========================================================================================================================================
# Experiment 4: mehrere gekoppelte Raten
# ==========================================================================================================================================
@lru_cache(maxsize=24)
def _dim_grid(d: int, g: int) -> tuple[float, float]:
    ref = REF.dimension()["d"][str(d)]
    m, sd = PO.tensor_grid_moments(PO.make_dim_model(d), g)
    return (float(np.max(np.abs(m - np.array(ref["e_u"])) / np.array(ref["sd_u"]))), float(np.max(np.abs(sd - np.array(ref["sd_u"])) / np.array(ref["sd_u"]))))


@dataclass(frozen=True, eq=False)
class DimRun:
    d: int
    g: int
    points: int
    live_err: tuple[float, float]      # größter Fehler E[u_j] in sd und sd[u_j] relativ des Gitters G^d gegen die vorgerechnete Referenz
    rows: tuple                        # je d = 1..6: (d, 10^d, Kettenschritte iso_a, iso_b, pre_a, ESS je Schritt iso_a)
    crossing: dict                     # Lauf → erste Dimension, in der das Gitter mehr Auswertungen braucht als die Kette Schritte
    accuracy: float
    grid_needed: int


def run_dim(s: Settings) -> DimRun:
    acc = C.DIM_ACC_VALUE[s.dim_acc]
    g_acc = C.DIM_GRID_FOR_ACC[s.dim_acc]
    rows = []
    for d in range(1, 7):
        ch = REF.dimension()["d"][str(d)]["ketten"]
        rows.append((d, g_acc ** d, *(1.0 / (acc ** 2 * ch[k]["ess_per_step"]) for k in ("iso_a", "iso_b", "pre_a")), ch["iso_a"]["ess_per_step"]))
    cross = {k: REF.crossing_dimension(k, g_acc, acc) for k in ("iso_a", "iso_b", "pre_a")}
    return DimRun(s.dim_d, s.dim_g, s.dim_g ** s.dim_d, _dim_grid(s.dim_d, s.dim_g), tuple(rows), cross, acc, g_acc)


# ==========================================================================================================================================
# Ring (Tabelle im Expander): Z(λ) = Spur(Tⁿ) gegen Aufzählung
# ==========================================================================================================================================
def ring_z_transfer(n: int, lam: float) -> float:
    t = np.array([[1.0, lam], [1.0, 0.0]])
    return float(np.trace(np.linalg.matrix_power(t, n)))


def ring_z_enumeration(n: int, lam: float) -> tuple[float, int]:
    """Z(λ) und Zahl der Belegungen (keine zwei benachbarten Plätze im Ring belegt) durch Aufzählung aller 2ⁿ Bitmuster."""
    x = np.arange(1 << n, dtype=np.uint32)
    rot = ((x << np.uint32(1)) | (x >> np.uint32(n - 1))) & np.uint32((1 << n) - 1)
    ok = (x & rot) == 0
    pop = np.zeros(x.shape, dtype=np.int64)
    for b in range(n):
        pop += ((x >> np.uint32(b)) & np.uint32(1)).astype(np.int64)
    pop = pop[ok]
    return float(np.sum(lam ** pop.astype(float))), int(ok.sum())


@lru_cache(maxsize=4)
def ring_rows(lam: float = 3.0, sizes: tuple = (6, 12, 20)) -> tuple:
    rows = []
    for n in sizes:
        zt = ring_z_transfer(n, lam)
        ze, states = ring_z_enumeration(n, lam)
        rows.append((n, states, zt, ze))
    return tuple(rows)


# ==========================================================================================================================================
# Verdikte (reine Funktionen)
# ==========================================================================================================================================
def verdict_gate(loss: float, equations: int, spread: float, t_mix: int | None, ess_loss: float, eps_pp: float, t_need: float, coverage_rows: list[dict] | None = None) -> tuple[str, str]:
    """Kernbefund des Experiments 1; der Bandzusatz erscheint nur für die vorgerechnete Voreinstellung (400 Ketten)."""
    if t_mix is None:
        mix = f"erreicht das Gleichgewicht (TV ≤ 1/4) in höchstens {CH.T_MIX_MAX} Schritten nicht"
    else:
        mix = f"ist nach t_mix(1/4) = {t_mix} Schritten im Gleichgewicht"
    text = (
        f"Verlust exakt **{F.prozent(loss, 2)}** aus {equations} Gleichungen (vier Rechenwege stimmen auf {F.potenz(spread)} überein). Die Kette {mix}, liefert aber "
        f"**{F.dez(ess_loss, 3)}** unabhängige Ziehungen je Schritt für die Verlustanzeige: für ±{F.dez(eps_pp, 1)} Prozentpunkte (95 %) braucht sie rund "
        f"**{F.groesse(t_need)}** Schritte."
    )
    if coverage_rows:
        parts = []
        for r in coverage_rows:
            parts.append(f"{F.prozent(r['ueberdeckung'], 1)} bei Start {'leer' if r['start'] == 'leer' else 'aus π'}")
        text += f" Vorgerechnet deckte das Band in {coverage_rows[0]['ketten']} Ketten (T = {F.ganz(coverage_rows[0]['t_schritte'])} Schritte): " + ", ".join(parts) + "."
    return ("success" if t_mix is not None else "warning"), text


def verdict_gamma(post: PO.GammaPosterior, alpha: float, beta: float, n: int, t: float, ks_p: float, draws: int, step_sd: float, accept: float, ess: float,
                  table: tuple, steps_001: float, frozen: list[dict] | None = None) -> tuple[str, str]:
    lo, hi = post.interval()
    best = max(e for _, _, e in table)
    grid_text = "; ".join(f"{F.dez(s_, 1)} sd: {F.dez(e, 2)}" for s_, _, e in table)
    text = (
        f"Prior Gamma({F.zahl(alpha)}, {F.zahl(beta)}), {n} Ankünfte in {F.zahl(t)} Abfertigungsdauern: Posterior exakt **Gamma({F.zahl(post.shape)}, {F.zahl(post.rate)})**, "
        f"Mittel {F.dez(post.mean, 3)}, sd {F.dez(post.sd, 3)}, Intervall ({F.dez(lo, 3)}; {F.dez(hi, 3)}), ohne Kette. "
        f"Unabhängige Ziehungen passen (Kolmogorow-Smirnow-Test p = {F.dez(ks_p, 2)}, {F.ganz(draws)} Ziehungen). Die Kette hat bei Schritt {F.dez(step_sd, 1)} sd "
        f"(Annahmerate {F.dez(accept, 2)}) etwa **{F.dez(ess, 3)}** unabhängige Ziehungen je Schritt; über die Schrittweiten der Tabelle: {grid_text}. "
        f"Für ein Mittel auf 0,01 sd: 10 000 unabhängige Ziehungen gegen rund **{F.groesse(steps_001)}** Kettenschritte gegen 0 (exakt)."
    )
    if frozen:
        text += (f" Vorgerechnet mit {frozen[0]['ketten']} Ketten zu {F.ganz(frozen[0]['schritte'])} Schritten (Start aus der exakten Verteilung): ESS je Schritt "
                 + ", ".join(f"{F.dez(r['ess_je_schritt'], 3)} bei {F.zahl(r['schritt_sd'], 1)} sd" for r in frozen) + ".")
    kind = "success" if ks_p > 0.01 else "warning"
    if ks_p <= 0.01:
        text += " Achtung: der Test lehnt die direkten Ziehungen ab (p ≤ 0,01); bei fester Zufallsfolge kommt das in einem von hundert Fällen vor."
    if best >= 1.0:
        text += " Mindestens eine Kette liegt bei oder über 1 unabhängiger Ziehung je Schritt (Rauschen der Schätzung, nicht erwartet)."
    return kind, text


def verdict_grid(prior_label: str, is_lognormal: bool, g: int, err: tuple[float, float, float], g_need: int | None, mean_lambda: float, conj_mean: float,
                 chain_steps_001: float) -> tuple[str, str]:
    worst = max(err)
    if g < C.COARSE_GRID:
        return "warning", (f"Gitter zu grob: mit G = {g} Punkten liegt der Fehler bei {F.dez(worst, 2)} sd (größter der drei Fehler); in der Messung braucht es mindestens G = "
                           f"{g_need if g_need is not None else C.LN_G_LIST[0]} für 0,01 sd.")
    if is_lognormal:
        lead = f"Mit {prior_label} ist der Posterior kein Gamma (E[λ] = {F.dez(mean_lambda, 3)} statt {F.dez(conj_mean, 3)} im konjugaten Fall). "
    else:
        lead = f"Mit {prior_label} ist der Posterior ein Gamma, E[λ] = {F.dez(mean_lambda, 3)} (hier braucht man kein Gitter, aber es trifft ihn auch). "
    if worst < 0.01:
        body = f"Ein Gitter mit G = {g} Punkten trifft Mittel und sd auf 0,01 sd (größter Fehler {F.potenz(worst)} sd), die Kette braucht dafür rund **{F.groesse(chain_steps_001)}** Schritte."
        return "success", lead + body
    body = (f"Ein Gitter mit G = {g} Punkten liegt noch bei {F.dez(worst, 3)} sd Fehler (zu grob für 0,01 sd"
            + (f"; in der Liste 6, 8, 10, 12 reicht erst G = {g_need}" if g_need else "") + f"); die Kette braucht für 0,01 sd rund {F.groesse(chain_steps_001)} Schritte.")
    return "info", lead + body


def verdict_dimension(d: int, g: int, live_err: tuple[float, float], rows: tuple, crossing: dict, accuracy_label: str, g_acc: int) -> tuple[str, str]:
    """Kernbefund des Experiments 4. `rows`: je d (d, G^d, Schritte Lauf A, Lauf B, vorkonditioniert, ESS); `crossing`: Lauf → erste Dimension, in der die Kette weniger Schritte
    braucht als das Gitter Auswertungen (über d = 6 hinaus aus der Hochrechnung ESS je Schritt ≈ Mittel(ESS · d)/d)."""
    cross = crossing["iso_a"]
    same = len(set(crossing.values())) == 1
    n = len(rows)

    def span(row):
        return f"{F.groesse(min(row[2:5]))} bis {F.groesse(max(row[2:5]))}"

    if cross is None:
        head = f"Für {accuracy_label} kostet das Gitter ({g_acc}^d Auswertungen) bis d = 12 weniger als die Kette (Hochrechnung über d = {n} hinaus)."
    elif cross == 1:
        head = f"Für {accuracy_label} braucht die Kette schon bei d = 1 weniger Schritte, als das Gitter Auswertungen hat."
    elif cross > n:
        head = (f"Für {accuracy_label} hat das Gitter ({g_acc}^d Auswertungen) in allen gemessenen Zeilen (d ≤ {n}) weniger Auswertungen als die Kette Schritte ({F.groesse(rows[-1][1])} gegen "
                f"{span(rows[-1])} bei d = {n}); die Hochrechnung kreuzt erst bei d = {cross} (hochgerechnet, nicht gemessen).")
    else:
        before, at = rows[cross - 2], rows[cross - 1]
        head = (f"Bis d = {cross - 1} hat das Gitter weniger Dichteauswertungen ({F.groesse(before[1])} gegen {span(before)} Kettenschritte je nach Lauf, für {accuracy_label}), "
                f"ab d = {cross} die Kette ({F.groesse(at[1])} gegen {span(at)}).")
        head += (" Das gilt in allen drei vorgerechneten Läufen (zwei Zufallsfolgen mit isotropem Schritt, ein vorkonditionierter Schritt)." if same else
                 " Die drei vorgerechneten Läufe kreuzen nicht bei derselben Dimension: " + ", ".join(f"{name} bei d = {crossing[name]}" for name in ("iso_a", "iso_b", "pre_a")) + ".")
    tail = (f" Hier: d = {d}, G = {g}: {F.groesse(g ** d)} Gitterpunkte, größter Fehler der Randmittel {F.potenz(live_err[0])} sd, der Randstandardabweichungen {F.potenz(live_err[1])} (relativ).")
    return "info", head + tail


def core_questions(gate: GateRun, gp: GammaRun, grid: GridRun, dim: DimRun, s: Settings) -> list[tuple[str, str, str]]:
    """Die drei Kernfragen mit Antworten aus den laufenden Rechnungen: (Frage, Art, Text)."""
    ex = gate.exact
    q1 = (
        "Lässt sich das, was ich wissen will, ausrechnen?",
        "success",
        f"Gate: ja, π aus {ex.equations} Gleichungen (Verlust {F.prozent(ex.loss, 2)}). Rate mit Gamma-Prior: ja, Posterior Gamma({F.zahl(gp.post.shape)}, {F.zahl(gp.post.rate)}) "
        f"aus zwei Zahlen. Prior aus Experiment 3 ({s.prior}): ein Gitter mit G = {s.grid_g} Punkten (größter Fehler {F.potenz(max(grid.err))} sd). {s.dim_d} gekoppelte Raten: ein Tensorgitter mit "
        f"{F.groesse(dim.points)} Punkten (größter Fehler {F.potenz(dim.live_err[0])} sd).",
    )
    q2 = (
        "Wie viele Gleichungen, Auswertungen oder Schritte kostet es ausgerechnet, wie viele abgetastet?",
        "info",
        f"Gate: {ex.equations} Gleichungen gegen rund {F.groesse(gate.t_need)} Schritte für ±{F.dez(s.eps_pp, 1)} Prozentpunkte. Rate: 2 Zahlen gegen rund "
        f"{F.groesse(gp.chain_steps_001)} Schritte für 0,01 sd. Prior aus Experiment 3: {s.grid_g} Auswertungen gegen rund {F.groesse(grid.chain_steps_001)} Schritte.",
    )
    cross = dim.crossing["iso_a"]
    q3 = (
        "Ab wann lohnt die Kette?",
        "warning",
        (f"In dieser Messung erst ab d = {cross} gekoppelten Raten (Genauigkeit {s.dim_acc}); bis d = {cross - 1} verliert die Kette, ebenso bei Gate und Gamma–Poisson."
         if cross is not None and cross > 1 else "In dieser Messung schon ab d = 1."),
    )
    return [q1, q2, q3]
