"""Verdikte an von Hand gesetzten Werten (DEMO-PLAYBOOK §12): jede Zahl im Text und jede Schwelle der Art steht hier.

Ein Verdikt ist eine reine Funktion (Art, Text) und bekommt Zahlen, nicht Seeds. Damit prüft der Test die Grenzen (p > 0,01, G < 6, worst < 0,01) und jede genannte Zahl direkt.
Dazu je Schnellstart-Szenario die erwartete Verdikt-Zeile mit den Rechnungen der App (kein Preset ohne Wirkung meldet eine Wirkung)."""

import pytest

import xab_constants as C
import xab_evaluation as ev
import xab_posterior as PO


# --- Gate ----------------------------------------------------------------------------------------------------------------------------------
def test_gate_verdict_names_loss_equations_mixing_and_steps():
    kind, text = ev.verdict_gate(0.0760, 7, 7.0e-14, 10, 0.151, 0.5, 71585.4)
    assert kind == "success"
    assert "**7,60 %**" in text and "7 Gleichungen" in text and "7,0·10⁻¹⁴" in text
    assert "t_mix(1/4) = 10 Schritten" in text and "**0,151**" in text and "±0,5 Prozentpunkte" in text and "**71 600**" in text
    assert "Vorgerechnet" not in text


def test_gate_verdict_without_mixing_within_the_limit_is_a_warning():
    kind, text = ev.verdict_gate(0.5, 5, 1e-14, None, 0.01, 1.0, 1.0e6)
    assert kind == "warning" and "in höchstens 5000 Schritten nicht" in text


def test_gate_verdict_adds_the_frozen_coverage_rows():
    rows = [{"ueberdeckung": 0.9525, "start": "leer", "ketten": 400, "t_schritte": 71586}, {"ueberdeckung": 0.9525, "start": "pi", "ketten": 400, "t_schritte": 71586}]
    _, text = ev.verdict_gate(0.076, 7, 1e-14, 10, 0.151, 0.5, 71585.0, rows)
    assert "400 Ketten (T = 71 586 Schritte): 95,2 % bei Start leer, 95,2 % bei Start aus π" in text


def test_frozen_coverage_only_for_the_default_gate_and_listed_eps():
    assert ev.frozen_coverage(2, 6, 80, 0.5) and len(ev.frozen_coverage(2, 6, 80, 0.5)) == 2
    assert ev.frozen_coverage(2, 6, 80, 1.0) and len(ev.frozen_coverage(2, 6, 80, 1.0)) == 2
    assert ev.frozen_coverage(2, 6, 80, 0.7) is None and ev.frozen_coverage(2, 6, 85, 0.5) is None and ev.frozen_coverage(1, 6, 80, 0.5) is None


# --- Gamma–Poisson -------------------------------------------------------------------------------------------------------------------------
TABLE = ((0.5, 0.84, 0.04), (1.0, 0.70, 0.12), (2.4, 0.44, 0.23), (5.0, 0.24, 0.17), (12.0, 0.10, 0.07))


def test_gamma_verdict_numbers_and_thresholds():
    post = PO.gamma_poisson_posterior(2, 1.0, 32, 20.0)
    kind, text = ev.verdict_gamma(post, 2, 1.0, 32, 20.0, 0.27, 2000, 2.4, 0.44, 0.231, TABLE, 43290.0)
    assert kind == "success"
    assert "Prior Gamma(2, 1), 32 Ankünfte in 20 Abfertigungsdauern" in text and "**Gamma(34, 21)**" in text
    assert "Mittel 1,619, sd 0,278, Intervall (1,121; 2,207)" in text and "p = 0,27, 2000 Ziehungen" in text
    assert "Schritt 2,4 sd (Annahmerate 0,44)" in text and "**0,231**" in text
    assert "0,5 sd: 0,04; 1,0 sd: 0,12; 2,4 sd: 0,23; 5,0 sd: 0,17; 12,0 sd: 0,07" in text
    assert "**43 300** Kettenschritte" in text
    assert "Achtung" not in text and "Rauschen der Schätzung" not in text


def test_gamma_verdict_threshold_of_the_ks_test_is_p_greater_than_0_01():
    post = PO.gamma_poisson_posterior(2, 1.0, 32, 20.0)
    assert ev.verdict_gamma(post, 2, 1.0, 32, 20.0, 0.0101, 2000, 2.4, 0.44, 0.2, TABLE, 5e4)[0] == "success"
    kind, text = ev.verdict_gamma(post, 2, 1.0, 32, 20.0, 0.01, 2000, 2.4, 0.44, 0.2, TABLE, 5e4)
    assert kind == "warning" and "Achtung" in text


def test_gamma_verdict_flags_a_chain_that_reaches_one_draw_per_step():
    post = PO.gamma_poisson_posterior(2, 1.0, 32, 20.0)
    _, text = ev.verdict_gamma(post, 2, 1.0, 32, 20.0, 0.5, 2000, 2.4, 0.44, 0.2, ((0.5, 0.8, 1.0),), 5e4)
    assert "Rauschen der Schätzung" in text
    _, text2 = ev.verdict_gamma(post, 2, 1.0, 32, 20.0, 0.5, 2000, 2.4, 0.44, 0.2, ((0.5, 0.8, 0.99),), 5e4)
    assert "Rauschen der Schätzung" not in text2


def test_gamma_verdict_adds_frozen_ess_only_when_given():
    post = PO.gamma_poisson_posterior(2, 1.0, 32, 20.0)
    frozen = [{"ketten": 400, "schritte": 5000, "ess_je_schritt": 0.041, "schritt_sd": 0.5}, {"ketten": 400, "schritte": 5000, "ess_je_schritt": 0.2297, "schritt_sd": 2.4}]
    _, text = ev.verdict_gamma(post, 2, 1.0, 32, 20.0, 0.5, 2000, 2.4, 0.44, 0.2, TABLE, 5e4, frozen)
    assert "Vorgerechnet mit 400 Ketten zu 5000 Schritten" in text and "0,041 bei 0,5 sd, 0,230 bei 2,4 sd" in text
    assert ev.frozen_gamma_table(2, 1.0, 32, 20.0) and ev.frozen_gamma_table(2, 1.0, 3, 2.0) is None


# --- Gitter ----------------------------------------------------------------------------------------------------------------------------------
def test_grid_verdict_success_for_lognormal_names_means_and_chain_steps():
    kind, text = ev.verdict_grid("Lognormal-Prior (Median 2,0, s₀ = 0,50)", True, 10, (2.5e-4, 1.0e-4, 4.0e-4), 10, 1.64215, 1.61905, 44600.0)
    assert kind == "success"
    assert "kein Gamma (E[λ] = 1,642 statt 1,619" in text and "G = 10 Punkten" in text and "0,01 sd (größter Fehler 4,0·10⁻⁴ sd)" in text and "**44 600** Schritte" in text


def test_grid_verdict_thresholds_g_below_6_and_error_at_0_01():
    assert ev.verdict_grid("x", True, 5, (0.04, 0.5, 0.1), 10, 1.6, 1.6, 4e4)[0] == "warning"
    assert "Gitter zu grob" in ev.verdict_grid("x", True, 5, (0.04, 0.5, 0.1), 10, 1.6, 1.6, 4e4)[1]
    assert ev.verdict_grid("x", True, 6, (0.0099, 0.0099, 0.0099), 10, 1.6, 1.6, 4e4)[0] == "success"          # genau unter der Schwelle
    kind, text = ev.verdict_grid("x", True, 6, (0.009, 0.21, 0.08), 10, 1.6, 1.6, 4e4)
    assert kind == "info" and "noch bei 0,210 sd Fehler" in text and "reicht erst G = 10" in text
    kind, text = ev.verdict_grid("x", True, 8, (0.01, 0.0, 0.0), None, 1.6, 1.6, 4e4)
    assert kind == "info" and "reicht erst" not in text                                                       # genau 0,01 ist nicht mehr darunter


def test_grid_verdict_for_the_conjugate_prior_says_it_is_a_gamma():
    _, text = ev.verdict_grid("Gamma(2, 1)-Prior", False, 10, (2.9e-4, 2.6e-5, 4.3e-4), 10, 1.61905, 1.61905, 4.6e4)
    assert "ist der Posterior ein Gamma, E[λ] = 1,619" in text and "kein Gamma" not in text


# --- Dimension -------------------------------------------------------------------------------------------------------------------------------
ROWS = ((1, 10, 4.7e4, 4.9e4, 4.4e4, 0.21), (2, 100, 8.9e4, 8.3e4, 6.6e4, 0.11), (3, 1e3, 1.16e5, 1.3e5, 1.0e5, 0.09), (4, 1e4, 1.56e5, 1.8e5, 1.4e5, 0.06),
        (5, 1e5, 1.94e5, 2.1e5, 1.5e5, 0.05), (6, 1e6, 2.46e5, 2.6e5, 2.0e5, 0.04))


def test_dimension_verdict_with_crossing_at_6():
    kind, text = ev.verdict_dimension(3, 10, (3.5e-4, 2.0e-4), ROWS, {"iso_a": 6, "iso_b": 6, "pre_a": 6}, "0,01 sd", 10)
    assert kind == "info"
    assert "Bis d = 5 hat das Gitter weniger Dichteauswertungen (100 000 gegen 150 000 bis 210 000 Kettenschritte je nach Lauf, für 0,01 sd)" in text
    assert "ab d = 6 die Kette (1 000 000 gegen 200 000 bis 260 000)" in text and "in allen drei vorgerechneten Läufen" in text
    assert "d = 3, G = 10: 1000 Gitterpunkte" in text and "3,5·10⁻⁴ sd" in text and "2,0·10⁻⁴ (relativ)" in text


def test_dimension_verdict_names_runs_that_disagree():
    _, text = ev.verdict_dimension(2, 10, (1e-4, 1e-4), ROWS, {"iso_a": 6, "iso_b": 5, "pre_a": 6}, "0,01 sd", 10)
    assert "nicht bei derselben Dimension: iso_a bei d = 6, iso_b bei d = 5, pre_a bei d = 6" in text and "in allen drei" not in text


def test_dimension_verdict_extrapolated_crossing_and_edge_cases():
    _, text = ev.verdict_dimension(6, 12, (1e-6, 1e-5), ROWS, {"iso_a": 9, "iso_b": 9, "pre_a": 9}, "10⁻⁴ sd (hochgerechnet)", 12)
    assert "in allen gemessenen Zeilen (d ≤ 6)" in text and "kreuzt erst bei d = 9 (hochgerechnet, nicht gemessen)" in text
    _, none_text = ev.verdict_dimension(1, 10, (1e-4, 1e-4), ROWS, {"iso_a": None, "iso_b": None, "pre_a": None}, "0,01 sd", 10)
    assert "bis d = 12 weniger als die Kette" in none_text
    _, one = ev.verdict_dimension(1, 10, (1e-4, 1e-4), ROWS, {"iso_a": 1, "iso_b": 1, "pre_a": 1}, "0,01 sd", 10)
    assert "schon bei d = 1 weniger Schritte" in one


# --- Schnellstart-Szenarien (je Preset die erwartete Zeile mit den Rechnungen der App) -------------------------------------------------------
def _settings(name):
    return ev.Settings(**C.PRESETS[name])


def test_default_preset_reports_what_it_measures():
    s = _settings("Gate: Voreinstellung (2 Spuren, 6 Plätze)")
    gate = ev.run_gate(s)
    kind, text = ev.verdict_gate(gate.exact.loss, gate.exact.equations, gate.exact.pi_deviation, gate.exact.t_mix, gate.exact.ess_loss, s.eps_pp, gate.t_need, ev.frozen_coverage(2, 6, 80, s.eps_pp))
    assert kind == "success" and "7,60 %" in text and "71 600" in text and "Vorgerechnet" in text


def test_overload_preset_reports_a_third_loss_and_204000_steps():
    gate = ev.run_gate(_settings("Gate: Überlast"))
    _, text = ev.verdict_gate(gate.exact.loss, gate.exact.equations, gate.exact.pi_deviation, gate.exact.t_mix, gate.exact.ess_loss, 0.5, gate.t_need, None)
    assert "35,96 %" in text and "204 000" in text and "t_mix(1/4) = 12" in text and "Vorgerechnet" not in text


def test_four_lane_preset_reports_erlang_b_and_no_waiting():
    s = _settings("Gate: vier Spuren ohne Warteplatz")
    gate = ev.run_gate(s)
    assert gate.erlang_b == pytest.approx(gate.exact.loss, rel=1e-12) and gate.metrics.mean_wait == 0.0
    assert ev.run_gate(_settings("Gate: Voreinstellung (2 Spuren, 6 Plätze)")).erlang_b is None


def test_small_data_preset_reports_gamma_5_3():
    s = _settings("Wenig Daten: 3 in 2 (Gamma(5, 3))")
    gp = ev.run_gamma(s)
    assert (gp.post.shape, gp.post.rate) == (5, 3.0)
    _, text = ev.verdict_gamma(gp.post, s.alpha, s.beta, s.n_obs, s.t_obs, gp.ks_p, len(gp.draws), s.gp_step, gp.accept, gp.ess, gp.table, gp.chain_steps_001, ev.frozen_gamma_table(s.alpha, s.beta, s.n_obs, s.t_obs))
    assert "**Gamma(5, 3)**" in text and "Vorgerechnet" not in text


def test_lognormal_preset_reports_a_non_gamma_posterior_that_a_grid_hits():
    s = _settings("Lognormal-Prior")
    grid = ev.run_grid(s)
    assert grid.ref.mean_lambda == pytest.approx(1.64215, abs=5e-6)
    kind, text = ev.verdict_grid("Lognormal-Prior", True, s.grid_g, grid.err, grid.g_need, grid.ref.mean_lambda, grid.conj_mean, grid.chain_steps_001)
    assert kind == "success" and "kein Gamma" in text and "E[λ] = 1,642 statt 1,619" in text


def test_six_layers_preset_reports_the_chain_winning():
    s = _settings("Sechs Schichten (d = 6)")
    dim = ev.run_dim(s)
    assert dim.crossing == {"iso_a": 6, "iso_b": 6, "pre_a": 6} and dim.points == 10 ** 6
    _, text = ev.verdict_dimension(s.dim_d, s.dim_g, dim.live_err, dim.rows, dim.crossing, s.dim_acc, dim.grid_needed)
    assert "ab d = 6 die Kette" in text


def test_core_questions_cover_the_three_questions():
    s = ev.Settings()
    qs = ev.core_questions(ev.run_gate(s), ev.run_gamma(s), ev.run_grid(s), ev.run_dim(s), s)
    assert [q[0] for q in qs] == ["Lässt sich das, was ich wissen will, ausrechnen?", "Wie viele Gleichungen, Auswertungen oder Schritte kostet es ausgerechnet, wie viele abgetastet?", "Ab wann lohnt die Kette?"]
    assert "7 Gleichungen" in qs[0][2] and "Gamma(34, 21)" in qs[0][2]
    assert "71 600" in qs[1][2] and "erst ab d = 6" in qs[2][2]
