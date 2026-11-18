"""Claims: jede Zahl der README wird hier nachgerechnet oder gegen die eingefrorene Messung gehalten (DEMO-PLAYBOOK §4, §11a).

Je Test: die Zahl aus dem Kern (exakt: Brüche, Ganzzahlen; Gleitkomma relativ 1e-9; Zufallsanteile als Band) und die Prüfung, dass **dieselbe Zahl in der README steht** (`readme(...)`).
Tabelle „Zahl → Test (Band oder exakt)“:

| README-Zahl | Test | Art |
|---|---|---|
| Gate-Tabelle (Verlust, t_mix, t_rel, ESS, Schritte) | test_gate_table | Ganzzahl exakt, Gleitkomma gerundet auf die genannten Stellen |
| 7 Gleichungen, vier Wege < 1e-12 | test_gate_four_ways | Schranke |
| 10-mal genauer = 100-mal mehr Schritte | test_square_root_law | exakt (Formel) und gemessen (Verhältnis im Band) |
| Überdeckung 96,5 / 95,75 / 95,25 / 95,25 %, Streuung 0,00248 | test_gate_coverage_frozen | eingefroren (400 Ketten) + Band um 95 % |
| Gamma(34, 21), Evidenz, Reihenfolge | test_gamma_poisson_numbers | exakt / 1e-9 |
| KS D = 0,0071, p = 0,27 (20 000 Ziehungen) | test_direct_draws_frozen | neu gerechnet, Band |
| ESS je Schritt 0,041 … 0,068, 43 500 Schritte | test_gamma_chain_frozen | eingefroren + Band um den Neu-Lauf |
| E[λ] 1,64215; Gitterfehler G = 4 … 12 | test_lognormal_grid | 1e-9 relativ / Rundung |
| ESS 0,224; 44 600; 4,5·10⁸ | test_lognormal_chain_frozen | eingefroren |
| Tabelle d = 1 … 6, Kreuzung bei d = 6 | test_dimension_table_frozen | eingefroren, Zählung über alle Zeilen |
| Kreuzung d = 9 (hochgerechnet) | test_dimension_extrapolation | Rechnung aus den eingefrorenen ESS |
| Ring 154 / 22 258 / 17 580 751 | test_ring_table | ganzzahlig exakt |
"""

import json
import math
import re
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import xab_chain as CH
import xab_evaluation as EV
import xab_format as F
import xab_posterior as PO
import xab_reference as REF
import xab_sampler as S

ROOT = Path(__file__).resolve().parent.parent
README = (ROOT / "README.md").read_text(encoding="utf-8")


def readme(*strings):
    for s in strings:
        assert s in README, f"in der README fehlt: {s!r}"


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 1
# ----------------------------------------------------------------------------------------------------------------------------------------
GATE_CASES = {
    "Voreinstellung (2, 6, 80 %)": (2, 6, 80, ("7,60 %", 10, "8,58", "0,151", "0,063", "71 585")),
    "eine Spur (1, 6, 80 %)": (1, 6, 80, ("6,63 %", 12, "9,56", "0,140", "0,056", "67 886")),
    "Überlast (2, 6, 150 %)": (2, 6, 150, ("35,96 %", 12, "7,57", "0,174", "0,073", "203 884")),
    "vier Spuren (4, 4, 80 %)": (4, 4, 80, ("22,81 %", 5, "3,73", "0,358", "0,161", "75 684")),
    "Kette mit 6 Zuständen (2, 5, 80 %)": (2, 5, 80, ("10,27 %", 7, "6,39", "0,182", "0,086", "77 908")),
}


@pytest.mark.parametrize("name", list(GATE_CASES))
def test_gate_table(name):
    c, k, last, (loss, t_mix, t_rel, ess_loss, ess_n, steps) = GATE_CASES[name]
    ex = EV.gate_exact(c, k, last)
    assert F.prozent(ex.loss, 2) == loss
    assert ex.t_mix == t_mix
    assert F.dez(ex.t_rel, 2) == t_rel
    assert F.dez(ex.ess_loss, 3) == ess_loss and F.dez(ex.ess_n, 3) == ess_n
    assert F.ganz(ex.steps_for(0.005)) == steps
    row = f"| {name.split(' (')[0]}"
    assert any(line.startswith(row) and loss in line and f"| {t_mix} |" in line and t_rel in line and ess_loss in line and ess_n in line and steps in line
               for line in README.splitlines()), name
    assert ex.equations == k + 1


def test_gate_table_four_lanes_equals_erlang_b_and_k5_has_six_states():
    assert EV.gate_exact(4, 4, 80).loss == pytest.approx(CH.erlang_b(4, 3.2), rel=1e-12)
    assert EV.gate_exact(2, 5, 80).equations == 6 and EV.gate_exact(2, 6, 80).equations == 7
    readme("Erlang B", "7 Gleichungen", "die Zahl 7 gilt für K = 5 (6 Zustände)")


def test_gate_four_ways():
    for c, k, last, _ in GATE_CASES.values():
        ex = EV.gate_exact(c, k, last)
        assert ex.pi_deviation < 1e-12
        p = ex.p
        assert np.abs(ex.pi @ p - ex.pi).max() < 1e-15
        flow = ex.pi[:, None] * p
        assert np.abs(flow - flow.T).max() < 1e-15
    assert EV.gate_exact(2, 6, 80).equations == 7
    readme("**7 Gleichungen**", "unter 10⁻¹² überein", "10⁻¹⁵")


def test_square_root_law():
    ex = EV.gate_exact(2, 6, 80)
    assert ex.steps_for(0.0005) / ex.steps_for(0.005) == pytest.approx(100.0, rel=1e-12)       # 10-mal genauer, 100-mal mehr Schritte (Formel)
    # gemessen: RMSE der Gamma-Kette fällt von N = 250 auf N = 5000 (Faktor 20) um √20 = 4,47 (je Schrittweite im Band 3,5 bis 6)
    for row in REF.messungen()["gamma_ketten"]:
        ratio = row["rmse_in_sd"]["250"] / row["rmse_in_sd"]["5000"]
        assert 3.5 <= ratio <= 6.0, row["schritt_sd"]
    readme("100-mal mehr Schritte", "T ∝ 1/ε²")


def test_gate_coverage_frozen():
    rows = REF.messungen()["gate_ueberdeckung"]
    assert [(r["eps"], r["start"], r["t_schritte"], r["ketten"]) for r in rows] == [(0.01, "leer", 17897, 400), (0.01, "pi", 17897, 400), (0.005, "leer", 71586, 400), (0.005, "pi", 71586, 400)]
    ex = EV.gate_exact(2, 6, 80)
    assert math.ceil(ex.steps_for(0.01)) == 17897 and math.ceil(ex.steps_for(0.005)) == 71586
    covers = [r["ueberdeckung"] for r in rows]
    assert covers == [0.965, 0.9575, 0.9525, 0.9525]
    se = math.sqrt(0.95 * 0.05 / 400)
    assert all(abs(c - 0.95) <= 2 * se for c in covers)                   # innerhalb von zwei Standardfehlern (alle vier)
    assert round(100 * se, 1) == 1.1
    for r in rows:
        assert 0.8 <= r["streuung_gemessen"] / r["streuung_exakt"] <= 1.2                # Verhältnis der Streuung zum Orakel √(σ²/T)
    assert F.dez(rows[2]["streuung_gemessen"], 5) == "0,00248" and F.dez(rows[2]["streuung_exakt"], 5) == "0,00255"
    readme("96,5 %", "95,75 %", "95,25 %", "0,00248", "0,00255", "(T = 17 897)", "(T = 71 586)")


def test_gate_coverage_recomputed_small():
    """Neu gerechnet mit 60 Ketten (Band, nicht die eingefrorene Zahl): Überdeckung nach dem Orakel-Band, Mittel nahe π_K."""
    import importlib.util

    spec = importlib.util.spec_from_file_location("xab_make_messungen", ROOT / "tools" / "xab_make_messungen.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    row = mod.gate_coverage(0.01, "pi", 60, 4242)
    assert 0.85 <= row["ueberdeckung"] <= 1.0
    ex = EV.gate_exact(2, 6, 80)
    assert abs(row["mittel"] - ex.loss) < 4 * math.sqrt(ex.sigma2_loss / 17897 / 60)


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 2
# ----------------------------------------------------------------------------------------------------------------------------------------
def test_gamma_poisson_numbers():
    post = PO.gamma_poisson_posterior(2, 1.0, 32, 20.0)
    assert (post.shape, post.rate) == (34, 21.0)
    lo, hi = post.interval()
    assert (F.dez(post.mean, 3), F.dez(post.sd, 3), F.dez(lo, 3), F.dez(hi, 3)) == ("1,619", "0,278", "1,121", "2,207")
    assert PO.evidence(2, 1.0, 32, 20.0) == pytest.approx(1.570427e-2, rel=1e-6)
    seq = PO.sequential_posterior(2, 1.0, [(10, 7.0), (22, 13.0)])
    assert (seq.shape, seq.rate) == (post.shape, post.rate)
    small = PO.gamma_poisson_posterior(2, 1.0, 3, 2.0)
    assert (small.shape, small.rate, small.mean, small.var) == (5, 3.0, pytest.approx(5 / 3), pytest.approx(5 / 9))
    readme("**Gamma(34, 21)**", "Mittel 1,619, sd 0,278", "(1,121; 2,207)", "1,570427·10⁻²", "10 Ankünfte in 7, dann 22 in 13", "Gamma(5, 3), Mittel 5/3, Varianz 5/9, Evidenz 32/243")


def test_direct_draws_frozen():
    frozen = REF.messungen()["gamma_direkt"]
    assert (frozen["ziehungen"], frozen["zufalls_seed"]) == (20000, 7301)
    # neu gerechnet (SplitMix64 und exact_log sind plattformfest; Toleranz für das letzte Bit der Produkte)
    draws = S.gamma_direct(34, 21.0, 20000, 7301)
    post = PO.gamma_poisson_posterior(2, 1.0, 32, 20.0)
    ks = stats.kstest(draws, post.dist.cdf)
    assert ks.statistic == pytest.approx(frozen["ks_d"], rel=1e-6) and ks.pvalue == pytest.approx(frozen["ks_p"], rel=1e-4)
    assert (F.dez(frozen["mittel"], 4), F.dez(frozen["sd"], 4), F.dez(frozen["ks_d"], 4), F.dez(frozen["ks_p"], 2)) == ("1,6161", "0,2787", "0,0071", "0,27")
    readme("Mittel 1,6161, sd 0,2787", "D = 0,0071, p = 0,27", "Zufalls-Seed 7301")


def test_gamma_chain_frozen():
    rows = REF.messungen()["gamma_ketten"]
    assert [r["schritt_sd"] for r in rows] == [0.5, 1.0, 2.4, 5.0, 12.0]
    assert [F.dez(r["ess_je_schritt"], 3) for r in rows] == ["0,041", "0,124", "0,230", "0,173", "0,068"]
    assert [F.dez(r["annahmerate"], 2) for r in rows] == ["0,84", "0,70", "0,44", "0,24", "0,10"]
    assert all((r["ketten"], r["schritte"]) == (400, 5000) for r in rows)
    assert all(r["ess_je_schritt"] < 1.0 for r in rows)                          # nie besser als die direkte Ziehung (ESS je Schritt 1), alle fünf Schrittweiten
    assert max(r["ess_je_schritt"] for r in rows) < 0.25
    assert F.rund_ganz(1e4 / rows[2]["ess_je_schritt"], 3) == "43 500"
    assert F.dez(REF.messungen()["gamma_ueberstreut"]["rmse_in_sd"], 3) == "0,031"
    readme("**0,041 / 0,124 / 0,230 / 0,173 / 0,068**", "0,84 / 0,70 / 0,44 / 0,24 / 0,10", "**43 500 Kettenschritte**", "RMSE des Mittels 0,031 sd", "10 000 unabhängige Ziehungen")
    # neu gerechnet im Band (Faktor 1,5 bei 400 Ketten × 1 000 Schritten nicht nötig: 150 Ketten genügen für ±15 % Standardfehler der ESS)
    post = PO.gamma_poisson_posterior(2, 1.0, 32, 20.0)
    x0 = S.gamma_direct(34, 21.0, 150, 77)
    run = S.rwm_1d(PO.gamma_log_density(post), x0, 2.4 * post.sd, 2000, 78, lower=0.0, checkpoints=(2000,))
    ess = S.ess_per_step_from_rmse(S.rmse_in_sd(run.means_at[2000], post.mean, post.sd), 2000)
    assert 0.5 * rows[2]["ess_je_schritt"] <= ess <= 1.6 * rows[2]["ess_je_schritt"]


def test_live_ess_estimate_scatters_by_about_a_quarter_at_60_chains():
    """README: die ESS je Schritt streut bei 60 Ketten um etwa ein Fünftel bis ein Viertel (Schritt 1 sd, zwölf Wiederholungen)."""
    post = PO.gamma_poisson_posterior(2, 1.0, 32, 20.0)
    logp = PO.gamma_log_density(post)
    est = []
    for seed in range(12):
        x0 = S.gamma_direct(34, 21.0, 60, 1000 + seed)
        run = S.rwm_1d(logp, x0, 1.0 * post.sd, 1000, 5000 + seed, lower=0.0, checkpoints=(1000,))
        est.append(S.ess_per_step_from_rmse(S.rmse_in_sd(run.means_at[1000], post.mean, post.sd), 1000))
    rel = float(np.std(est, ddof=1) / np.mean(est))
    assert 0.15 <= rel <= 0.35


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 3
# ----------------------------------------------------------------------------------------------------------------------------------------
def test_lognormal_grid():
    prior = PO.Prior1D("lognormal", 2.0, 0.5)
    ref = PO.reference_moments_1d(32, 20.0, prior)
    assert (F.dez(ref.mean_u, 5), F.dez(ref.sd_u, 5), F.dez(ref.mean_lambda, 5)) == ("0,48240", "0,16564", "1,64215")
    assert F.dez(ref.mean_lambda - 34 / 21, 4) == "0,0231"
    errs = {g: PO.grid_errors_in_sd(32, 20.0, prior, g) for g in (4, 6, 8, 10, 12)}
    want = {6: ("9,2·10⁻³", "0,21", "8,0·10⁻²"), 8: ("6,1·10⁻³", "1,7·10⁻²", "1,4·10⁻²"), 10: ("2,5·10⁻⁴", "1,0·10⁻⁴", "4,0·10⁻⁴"), 12: ("1,0·10⁻⁶", "1,0·10⁻⁵", "1,1·10⁻⁶")}
    for g, (a, b, c) in want.items():
        assert F.potenz(errs[g][0]) == a and F.potenz(errs[g][2]) == c
        assert (F.dez(errs[g][1], 2) if g == 6 else F.potenz(errs[g][1])) == b
    assert max(errs[4]) == pytest.approx(0.95, abs=0.005) and max(errs[4]) > 0.3
    assert all(max(errs[g]) >= 0.01 for g in (4, 6, 8)) and max(errs[10]) < 0.01 and max(errs[12]) < 0.01           # G = 10 ist das kleinste G der Liste unter 0,01 sd
    readme("**1,64215**", "1,61905", "0,0231", "E[u] = 0,48240, sd[u] = 0,16564", "9,2·10⁻³ / 0,21 / 8,0·10⁻²", "6,1·10⁻³ / 1,7·10⁻² / 1,4·10⁻²",
           "**G = 10: 2,5·10⁻⁴ / 1,0·10⁻⁴ / 4,0·10⁻⁴**", "1,0·10⁻⁶ / 1,0·10⁻⁵ / 1,1·10⁻⁶", "bis 0,95 (zu grob)")


def test_lognormal_chain_frozen():
    row = REF.messungen()["lognormal_kette"]
    assert (row["ketten"], row["schritte"], row["schritt_sd"]) == (400, 16000, 2.4)
    assert [F.dez(v, 3) for v in row["rmse_in_sd"].values()] == ["0,135", "0,068", "0,033", "0,017"]
    assert F.dez(row["ess_je_schritt"], 3) == "0,224" and F.dez(row["annahmerate"], 3) == "0,442"
    assert F.rund_ganz(1e4 / row["ess_je_schritt"], 3) == "44 600"
    assert F.potenz(1e8 / row["ess_je_schritt"]) == "4,5·10⁸"
    readme("0,135 / 0,068 / 0,033 / 0,017 sd", "ESS je Schritt 0,224, Annahmerate 0,442", "**44 600 Schritte**", "4,5·10⁸ Schritte", "G = 10 Dichteauswertungen")


# ----------------------------------------------------------------------------------------------------------------------------------------
# Experiment 4
# ----------------------------------------------------------------------------------------------------------------------------------------
def test_dimension_table_frozen():
    dim = REF.dimension()
    assert dim["meta"]["ketten"] == 240 and dim["meta"]["schritte"] == 64000
    want = {
        1: ("46 900", "49 300", "44 200", "0,213", "0,213", "0,445", "2,8·10⁻⁴, 1,2·10⁻⁴"),
        2: ("89 100", "83 300", "65 800", "0,112", "0,225", "0,339", "3,3·10⁻⁴, 7,9·10⁻⁵"),
        3: ("116 000", "132 000", "102 000", "0,086", "0,259", "0,298", "3,5·10⁻⁴, 2,0·10⁻⁴"),
        4: ("156 000", "181 000", "136 000", "0,064", "0,256", "0,275", "3,5·10⁻⁴, 2,2·10⁻⁴"),
        5: ("194 000", "210 000", "152 000", "0,051", "0,257", "0,259", "3,5·10⁻⁴, 2,2·10⁻⁴"),
        6: ("246 000", "260 000", "196 000", "0,041", "0,244", "0,250", "3,5·10⁻⁴, 2,3·10⁻⁴"),
    }
    for d, (a, b, pre, ess, essd, acc, gerr) in want.items():
        ch = dim["d"][str(d)]["ketten"]
        assert (F.groesse(1e4 / ch["iso_a"]["ess_per_step"]), F.groesse(1e4 / ch["iso_b"]["ess_per_step"]), F.groesse(1e4 / ch["pre_a"]["ess_per_step"])) == (a, b, pre)
        assert F.dez(ch["iso_a"]["ess_per_step"], 3) == ess and F.dez(ch["iso_a"]["ess_per_step"] * d, 3) == essd and F.dez(ch["iso_a"]["accept"], 3) == acc
        e = dim["d"][str(d)]["gitter_fehler"]["10"]
        assert f"{F.potenz(e[0])}, {F.potenz(e[1])}" == gerr
        gp = f"{10 ** d:,}".replace(",", " ")                          # README gruppiert ab vier Stellen („1 000“), F.ganz erst ab fünf
        row = [ln for ln in README.splitlines() if ln.startswith(f"| {d} | {gp} |")]
        assert row and all(x in row[0] for x in (a, b, pre, ess, essd, acc, gerr)), d
    # Kreuzung: Gitter G = 10 gegen Kette, in jedem der drei Läufe und in jeder Zeile gezählt
    for run in ("iso_a", "iso_b", "pre_a"):
        grid_wins = [10 ** d < 1e4 / dim["d"][str(d)]["ketten"][run]["ess_per_step"] for d in range(1, 7)]
        assert grid_wins == [True] * 5 + [False], run                   # bis d = 5 hat das Gitter weniger Auswertungen, ab d = 6 die Kette
        assert REF.crossing_dimension(run, 10, 0.01) == 6
    # Gitterfehler G = 10 in allen sechs d unter 0,01 sd und in der genannten Spanne
    errs = [dim["d"][str(d)]["gitter_fehler"]["10"] for d in range(1, 7)]
    assert all(max(e) < 0.01 for e in errs)
    assert (F.potenz(min(e[0] for e in errs)), F.potenz(max(e[0] for e in errs)), F.potenz(min(e[1] for e in errs)), F.potenz(max(e[1] for e in errs))) == ("2,8·10⁻⁴", "3,5·10⁻⁴", "7,9·10⁻⁵", "2,3·10⁻⁴")
    # ESS je Schritt × d in 18 Zeilen zwischen 0,20 und 0,33; Abweichung der beiden isotropen Läufe höchstens 16 %; vorkonditioniert spart 6 % (d = 1) bis 26 % (d = 2) gegen Folge A
    essd = [dim["d"][str(d)]["ketten"][r]["ess_per_step"] * d for d in range(1, 7) for r in ("iso_a", "iso_b", "pre_a")]
    assert len(essd) == 18 and F.dez(min(essd), 2) == "0,20" and F.dez(max(essd), 2) == "0,33"
    dev = [abs(dim["d"][str(d)]["ketten"]["iso_a"]["ess_per_step"] / dim["d"][str(d)]["ketten"]["iso_b"]["ess_per_step"] - 1) for d in range(1, 7)]
    assert round(100 * max(dev)) == 16
    saving = [1 - dim["d"][str(d)]["ketten"]["iso_a"]["ess_per_step"] / dim["d"][str(d)]["ketten"]["pre_a"]["ess_per_step"] for d in range(1, 7)]
    assert round(100 * min(saving)) == 6 and saving.index(min(saving)) == 0 and round(100 * max(saving)) == 26 and saving.index(max(saving)) == 1
    # RMSE fällt in jeder Zeile wie 1/√N: Verhältnis N = 1 000 → 64 000 (Faktor 64, Wurzel 8) im Band
    for d in range(1, 7):
        for run in ("iso_a", "iso_b", "pre_a"):
            r = dim["d"][str(d)]["ketten"][run]["rmse_in_sd"]
            assert 6.0 <= r["1000"] / r["64000"] <= 10.0, (d, run)
    g8 = [dim["d"][str(d)]["gitter_fehler"]["8"][1] for d in range(1, 7)]                        # G = 8 reicht in keiner Dimension (sd[u_j] relativ ≥ 0,01)
    assert min(g8) >= 0.01 and F.potenz(min(g8)) == "1,7·10⁻²" and F.potenz(max(g8)) == "1,9·10⁻²"
    r6 = dim["d"]["6"]["ketten"]["iso_a"]["rmse_in_sd"]
    assert [F.dez(r6[k], 3) for k in ("1000", "4000", "16000", "64000")] == ["0,165", "0,079", "0,038", "0,020"]
    readme("**Bis d = 5 hat das Gitter weniger Dichteauswertungen als die Kette Schritte, ab d = 6 die Kette**", "in allen 18 Zeilen wie 1/√N", "1,7·10⁻² bis 1,9·10⁻²", "2,8·10⁻⁴ bis 3,5·10⁻⁴ sd", "7,9·10⁻⁵ bis 2,3·10⁻⁴",
           "0,165 / 0,079 / 0,038 / 0,020", "zwischen 0,20 und 0,33", "bis zu 16 %", "6 % (d = 1) bis 26 % (d = 2)")


def test_dimension_extrapolation():
    ess = {d: REF.dimension()["d"][str(d)]["ketten"]["iso_a"]["ess_per_step"] for d in range(1, 7)}
    mean_essd = sum(ess[d] * d for d in ess) / 6
    # 10⁻⁴ sd, G = 12: Gitter 12^d gegen Kette 10⁸/(Mittel(ESS · d)/d); erste Dimension mit mehr Gitterpunkten als Kettenschritten
    first = next(d for d in range(1, 13) if 12 ** d > 1e8 / ((ess[d] if d in ess else mean_essd / d)))
    assert first == 9 and REF.crossing_dimension("iso_a", 12, 1e-4) == 9
    assert 12 ** 8 < 1e8 / (mean_essd / 8) and 12 ** 9 > 1e8 / (mean_essd / 9)
    readme("**d = 9**", "hochgerechnet")


# ----------------------------------------------------------------------------------------------------------------------------------------
# Ring
# ----------------------------------------------------------------------------------------------------------------------------------------
def test_ring_table():
    rows = EV.ring_rows()
    assert [(n, states) for n, states, _, _ in rows] == [(6, 18), (12, 322), (20, 15127)]
    assert [round(zt) for _, _, zt, _ in rows] == [154, 22258, 17580751] and [round(ze) for _, _, _, ze in rows] == [154, 22258, 17580751]
    readme("**154 / 22 258 / 17 580 751**", "18 / 322 / 15 127 Belegungen", "nicht für die Mischzeit")


def test_frozen_files_are_valid_json_with_all_sections():
    for name, keys in (("xab_dimension.json", {"meta", "d"}), ("xab_messungen.json", {"meta", "gate_ueberdeckung", "gamma_direkt", "gamma_ketten", "gamma_ueberstreut", "lognormal_kette"})):
        data = json.loads((ROOT / name).read_text(encoding="utf-8"))
        assert keys <= set(data), name


def test_readme_has_no_open_website_remark_and_names_its_limits():
    assert not re.search(r"(?i)einbindung[^.\n]*offen|offen[^.\n]*einbindung", README)
    readme("## Ehrliche Grenzen", "## Befunde und Korrekturen gegenüber der ersten Messreihe", "## Bewusst nicht umgesetzt")
