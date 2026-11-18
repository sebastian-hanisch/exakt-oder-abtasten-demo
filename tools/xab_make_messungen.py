"""Vorgerechnete Messungen mit 400 Ketten (Experimente 1 bis 3), die zu teuer für die Live-Rechnung sind: xab_messungen.json.

    python tools/xab_make_messungen.py            # rechnet alles neu und schreibt xab_messungen.json (einige Minuten)
    python tools/xab_make_messungen.py --check    # rechnet mit kleinem Aufwand nach und vergleicht (Größenordnung)

A  Überdeckung des Bandes ±ε beim Zeitanteil im Zustand K (Gate Voreinstellung: 2 Spuren, 6 Plätze, Last 80 %): T = ⌈1,96² σ²/ε²⌉ Schritte (σ² exakt),
   400 Ketten, ε = 0,01 und 0,005, Start leer (Zustand 0, ohne Einlauf) und Start aus π; Zufalls-Seeds 7201 bis 7204.
B  Gamma(34, 21) (Prior Gamma(2, 1), 32 Ankünfte in 20): Zufallsschritt-Metropolis mit Schritt 0,5 / 1 / 2,4 / 5 / 12 sd, 400 Ketten × 5 000 Schritte, Start aus der
   exakten Verteilung (direkte Ziehung), RMSE des Mittels bei N = 250, 1 000, 5 000; dazu 20 000 direkte Ziehungen (Zufalls-Seed 7301) mit Kolmogorow-Smirnow-Test
   und die Kette mit überstreutem Start (0,3 / 1 / 2,5 / 4, kein Einlauf verworfen).
C  Prior Lognormal (Median 2, s0 = 0,5), 32 in 20: Schritt 2,4 sd in u = ln λ, 400 Ketten × 16 000 Schritte, Start aus der exakten Verteilung (Inversion), RMSE von E[u]
   bei N = 250, 1 000, 4 000, 16 000.
ESS je Schritt = (1/RMSE)² / N (RMSE in Einheiten der Standardabweichung des Ziels); Kettenschritte für 0,01 sd = 10 000 / (ESS je Schritt)."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import xab_chain as C  # noqa: E402
import xab_posterior as P  # noqa: E402
import xab_sampler as S  # noqa: E402

OUT = ROOT / "xab_messungen.json"
GATE = (2, 6, 1.6)
STEPS_SD = (0.5, 1.0, 2.4, 5.0, 12.0)
SEEDS = {"cover": 7201, "gamma_direct": 7301, "gamma_start": 7302, "gamma_chain": 7303, "gamma_over": 7304, "ln_start": 7401, "ln_chain": 7402}


def gate_coverage(eps: float, start: str, r: int, seed: int) -> dict:
    c, k, a = GATE
    g = C.exact_gate(c, k, a)
    t_steps = math.ceil(C.steps_for_band(g.sigma2_loss, eps))
    if start == "leer":
        starts = 0
    else:
        u = S.uniforms(seed + 5000, r)
        starts = np.minimum(np.searchsorted(np.cumsum(g.pi), u, side="right"), k)
    est = S.simulate_gate(c, k, a, t_steps, r, seed, starts)
    cover = float(np.mean(np.abs(est - g.loss) <= eps))
    return {"eps": eps, "start": start, "t_schritte": t_steps, "ketten": r, "zufalls_seed": seed, "ueberdeckung": cover,
            "streuung_gemessen": float(est.std(ddof=1)), "streuung_exakt": math.sqrt(g.sigma2_loss / t_steps), "mittel": float(est.mean())}


def gamma_logp(post: P.GammaPosterior):
    def logp(x):
        return np.where(x > 0, (post.shape - 1) * np.log(np.maximum(x, 1e-300)) - post.rate * x, -np.inf)
    return logp


def gamma_chain(step_sd: float, r: int, n_steps: int, checkpoints, seed: int, x0=None) -> dict:
    post = P.gamma_poisson_posterior(2, 1, 32, 20.0)
    if x0 is None:
        x0 = S.gamma_direct(int(post.shape), post.rate, r, SEEDS["gamma_start"])
    run = S.rwm_1d(gamma_logp(post), x0, step_sd * post.sd, n_steps, seed, lower=0.0, checkpoints=checkpoints)
    rmse = {str(t): S.rmse_in_sd(run.means_at[t], post.mean, post.sd) for t in checkpoints}
    last = checkpoints[-1]
    return {"schritt_sd": step_sd, "ketten": r, "schritte": n_steps, "annahmerate": run.accept_rate, "rmse_in_sd": rmse,
            "ess_je_schritt": S.ess_per_step_from_rmse(rmse[str(last)], last)}


def lognormal_chain(r: int, n_steps: int, checkpoints, seed: int) -> dict:
    prior = P.Prior1D("lognormal", 2.0, 0.5)
    ref = P.reference_moments_1d(32, 20.0, prior)
    x0 = P.grid_sampler_start(32, 20.0, prior, S.uniforms(SEEDS["ln_start"], r))
    run = S.rwm_1d(lambda x: P.log_post_u(x, 32, 20.0, prior), x0, 2.4 * ref.sd_u, n_steps, seed, checkpoints=checkpoints)
    rmse = {str(t): S.rmse_in_sd(run.means_at[t], ref.mean_u, ref.sd_u) for t in checkpoints}
    last = checkpoints[-1]
    return {"schritt_sd": 2.4, "ketten": r, "schritte": n_steps, "annahmerate": run.accept_rate, "rmse_in_sd": rmse,
            "ess_je_schritt": S.ess_per_step_from_rmse(rmse[str(last)], last)}


def build(r: int = 400, verbose: bool = True) -> dict:
    t0 = time.time()
    data: dict = {"meta": {"beschreibung": "Messungen mit 400 Ketten für Experimente 1 bis 3 (ausgedachte Zählungen, Gate aus markov-queue-demo)",
                           "zufalls_seeds": SEEDS, "python": sys.version.split()[0], "numpy": np.__version__}}
    data["gate_ueberdeckung"] = [gate_coverage(eps, start, r, SEEDS["cover"] + i)
                                 for i, (eps, start) in enumerate([(0.01, "leer"), (0.01, "pi"), (0.005, "leer"), (0.005, "pi")])]
    if verbose:
        print(f"A: {time.time() - t0:.0f} s", [round(x["ueberdeckung"], 3) for x in data["gate_ueberdeckung"]], flush=True)
    post = P.gamma_poisson_posterior(2, 1, 32, 20.0)
    draws = S.gamma_direct(34, 21.0, 20000, SEEDS["gamma_direct"])
    ks = stats.kstest(draws, post.dist.cdf)
    data["gamma_direkt"] = {"ziehungen": 20000, "zufalls_seed": SEEDS["gamma_direct"], "mittel": float(draws.mean()), "sd": float(draws.std(ddof=1)),
                            "ks_d": float(ks.statistic), "ks_p": float(ks.pvalue)}
    data["gamma_ketten"] = [gamma_chain(s, r, 5000, (250, 1000, 5000), SEEDS["gamma_chain"] + i) for i, s in enumerate(STEPS_SD)]
    over = np.array([0.3, 1.0, 2.5, 4.0] * (r // 4))
    run = S.rwm_1d(gamma_logp(post), over, 2.4 * post.sd, 5000, SEEDS["gamma_over"], lower=0.0, checkpoints=(5000,))
    data["gamma_ueberstreut"] = {"ketten": len(over), "schritte": 5000, "rmse_in_sd": S.rmse_in_sd(run.means_at[5000], post.mean, post.sd)}
    if verbose:
        print(f"B: {time.time() - t0:.0f} s", [round(x["ess_je_schritt"], 3) for x in data["gamma_ketten"]], flush=True)
    data["lognormal_kette"] = lognormal_chain(r, 16000, (250, 1000, 4000, 16000), SEEDS["ln_chain"])
    if verbose:
        print(f"C: {time.time() - t0:.0f} s", round(data["lognormal_kette"]["ess_je_schritt"], 3), flush=True)
    return data


def check(path: Path = OUT) -> list[str]:
    """Rechnet mit 60 Ketten nach: Überdeckung im Band, ESS je Schritt im Faktor 2 der eingefrorenen Werte."""
    frozen = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    row = frozen["gate_ueberdeckung"][0]
    cov = gate_coverage(row["eps"], "pi", 60, 99001)
    if not 0.80 <= cov["ueberdeckung"] <= 1.0:
        problems.append(f"Überdeckung {cov['ueberdeckung']:.2f} (60 Ketten) außerhalb [0,80; 1]")
    for row in frozen["gamma_ketten"]:
        fresh = gamma_chain(row["schritt_sd"], 60, 1000, (1000,), 99002)
        ratio = fresh["ess_je_schritt"] / row["ess_je_schritt"]
        if not 0.5 <= ratio <= 2.0:
            problems.append(f"Gamma, Schritt {row['schritt_sd']} sd: ESS je Schritt {fresh['ess_je_schritt']:.3f} gegen {row['ess_je_schritt']:.3f}")
    ln = lognormal_chain(60, 2000, (2000,), 99003)
    if not 0.5 <= ln["ess_je_schritt"] / frozen["lognormal_kette"]["ess_je_schritt"] <= 2.0:
        problems.append("Lognormal: ESS je Schritt weicht um mehr als den Faktor 2 ab")
    return problems


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--check" in argv:
        problems = check()
        print("\n".join(problems) if problems else "xab_messungen.json stimmt (Größenordnung)")
        return 1 if problems else 0
    data = build()
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
