"""Vorgerechnete Tabelle „mehrere gekoppelte Raten“ (Experiment 4): Referenz-Momente, Gitterfehler und Ketten-Messungen für d = 1 bis 6.

    python tools/xab_make_dimension.py            # rechnet alles neu und schreibt xab_dimension.json (einige Minuten)
    python tools/xab_make_dimension.py --check    # rechnet mit kleinem Aufwand nach und vergleicht mit xab_dimension.json (Sekunden bis wenige Minuten)

Modell (ausgedacht, keine reale Datenquelle): d Schichten, je eine Ankunftsrate λ_j, Zählungen n = (8, 6, 11, 9, 7, 10)[:d] in T = 5, Prior auf u = ln λ Gauß mit
Median 2, sd 0,5, Korrelation 0,7^|i−j| (xab_posterior.make_dim_model). Referenz: Tensorgitter mit G = 400 / 100 / 60 / 30 / 18 / 14 je Achse (bis 7,5·10⁶ Punkte).
Gitter: G ∈ {4, 5, 6, 8, 10, 12} je Achse, Kasten Gipfel ± 6 Laplace-sd. Ketten: Zufallsschritt-Metropolis mit festem Schritt,
  iso_a, iso_b: isotroper Schritt 2,38/√d · mittlere Laplace-sd, zwei Zufallsfolgen (Zufalls-Seeds 7601 und 7602), je 240 Ketten × 64 000 Schritte;
  pre_a: Schritt 2,38/√d mal Cholesky-Faktor der Laplace-Kovarianz (vorkonditioniert), Zufalls-Seed 7603, 240 Ketten.
Start: Laplace-Gipfel plus Laplace-sd mal Normalzahl (kein Einlauf verworfen). ESS je Schritt = (1/RMSE)² / N aus N = 64 000 (RMSE in sd, über Ketten und
Koordinaten gemittelt). Zähleinheit: Dichteauswertungen (Gitterpunkte bzw. Kettenschritte)."""

from __future__ import annotations

import json
import math
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

import xab_posterior as P  # noqa: E402
import xab_sampler as S  # noqa: E402

OUT = ROOT / "xab_dimension.json"
REF_G = {1: 400, 2: 100, 3: 60, 4: 30, 5: 18, 6: 14}
GRID_G = (4, 5, 6, 8, 10, 12)
CHECKPOINTS = (1000, 4000, 16000, 64000)
RUNS = {"iso_a": ("iso", 7601), "iso_b": ("iso", 7602), "pre_a": ("pre", 7603)}
R_CHAINS = 240
D_MAX = 6


def chain_run(d: int, kind: str, seed: int, r: int, n_steps: int, checkpoints, ref_mean, ref_sd) -> dict:
    model = P.make_dim_model(d)
    umode, cov = P.laplace_dim(model)
    sdl = np.sqrt(np.diag(cov))
    base = 2.38 / math.sqrt(d)
    chol = base * np.linalg.cholesky(cov) if kind == "pre" else base * float(np.mean(sdl)) * np.eye(d)
    u0 = S.uniforms(seed + 100000, 2 * r * d).reshape(r, 2, d)
    x0 = umode[None, :] + sdl[None, :] * S.box_muller(u0[:, 0, :], u0[:, 1, :])
    run = S.rwm_multi(model.log_post, x0, chol, n_steps, seed, checkpoints)
    rmse = {str(t): S.rmse_in_sd(run.means_at[t], ref_mean[None, :], ref_sd[None, :]) for t in checkpoints}
    last = checkpoints[-1]
    return {"seed": seed, "r": r, "steps": n_steps, "kind": kind, "step": float(np.mean(np.diag(chol))) if kind == "iso" else float(base),
            "accept": run.accept_rate, "rmse_in_sd": rmse, "ess_per_step": S.ess_per_step_from_rmse(rmse[str(last)], last)}


def grid_errors(d: int, ref_mean, ref_sd, grids=GRID_G, max_points=3_500_000) -> dict:
    model = P.make_dim_model(d)
    out = {}
    for g in grids:
        if g ** d > max_points:
            continue
        m, s = P.tensor_grid_moments(model, g)
        out[str(g)] = [float(np.max(np.abs(m - ref_mean) / ref_sd)), float(np.max(np.abs(s - ref_sd) / ref_sd))]
    return out


def build(dims=range(1, D_MAX + 1), r=R_CHAINS, n_steps=64000, checkpoints=CHECKPOINTS, ref_g=REF_G, verbose=True) -> dict:
    data = {"meta": {
        "beschreibung": "Experiment 4: Tensorgitter gegen Zufallsschritt-Metropolis, d = 1..6 gekoppelte Raten (ausgedachte Zählungen, kein reales Datenset)",
        "zaehlungen": list(P.DIM_DATA), "dauer": P.DIM_T, "prior": {"median": P.DIM_MEDIAN, "sd": P.DIM_SD, "rho": P.DIM_RHO},
        "referenz_g": {str(k): v for k, v in ref_g.items()}, "gitter_g": list(GRID_G), "kasten": P.BOX_WIDTH, "ketten": r, "schritte": n_steps,
        "kontrollpunkte": list(checkpoints), "laeufe": {k: {"art": v[0], "zufalls_seed": v[1]} for k, v in RUNS.items()},
        "python": sys.version.split()[0], "numpy": np.__version__}, "d": {}}
    for d in dims:
        t0 = time.time()
        model = P.make_dim_model(d)
        ref_mean, ref_sd = P.tensor_grid_moments(model, ref_g[d])
        entry = {"referenz_g": ref_g[d], "e_u": ref_mean.tolist(), "sd_u": ref_sd.tolist(), "gitter_fehler": grid_errors(d, ref_mean, ref_sd), "ketten": {}}
        for name, (kind, seed) in RUNS.items():
            entry["ketten"][name] = chain_run(d, kind, seed + d, r, n_steps, checkpoints, ref_mean, ref_sd)
        data["d"][str(d)] = entry
        if verbose:
            print(f"d = {d}: {time.time() - t0:.0f} s; ESS je Schritt " + ", ".join(f"{k} {v['ess_per_step']:.3f}" for k, v in entry["ketten"].items()), flush=True)
    return data


def check(path: Path = OUT, dims=(1, 2, 3, 4), r=40, n_steps=4000) -> list[str]:
    """Rechnet mit kleinem Aufwand nach: Referenz und Gitterfehler exakt (relativ 1e-6), Ketten nur als Größenordnung (ESS je Schritt im Faktor 2)."""
    frozen = json.loads(Path(path).read_text(encoding="utf-8"))
    problems = []
    for d in dims:
        e = frozen["d"][str(d)]
        model = P.make_dim_model(d)
        ref_mean, ref_sd = P.tensor_grid_moments(model, e["referenz_g"])
        if not (np.allclose(ref_mean, e["e_u"], rtol=1e-6, atol=1e-9) and np.allclose(ref_sd, e["sd_u"], rtol=1e-6)):
            problems.append(f"d = {d}: Referenz-Momente weichen ab")
        for g, (e1, e2) in grid_errors(d, np.array(e["e_u"]), np.array(e["sd_u"])).items():
            f1, f2 = e["gitter_fehler"][g]
            if not (math.isclose(e1, f1, rel_tol=1e-3, abs_tol=1e-9) and math.isclose(e2, f2, rel_tol=1e-3, abs_tol=1e-9)):
                problems.append(f"d = {d}, G = {g}: Gitterfehler {e1:.3e}/{e2:.3e} statt {f1:.3e}/{f2:.3e}")
        fresh = chain_run(d, "iso", 99000 + d, r, n_steps, (n_steps,), np.array(e["e_u"]), np.array(e["sd_u"]))
        ratio = fresh["ess_per_step"] / e["ketten"]["iso_a"]["ess_per_step"]
        if not 0.5 <= ratio <= 2.0:
            problems.append(f"d = {d}: ESS je Schritt {fresh['ess_per_step']:.3f} gegen eingefroren {e['ketten']['iso_a']['ess_per_step']:.3f}")
    return problems


def main(argv=None) -> int:
    argv = sys.argv[1:] if argv is None else argv
    if "--check" in argv:
        problems = check()
        print("\n".join(problems) if problems else "xab_dimension.json stimmt (Referenz und Gitterfehler exakt, Ketten als Größenordnung)")
        return 1 if problems else 0
    data = build()
    OUT.write_text(json.dumps(data, ensure_ascii=False, indent=1) + "\n", encoding="utf-8")
    print(f"geschrieben: {OUT}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
