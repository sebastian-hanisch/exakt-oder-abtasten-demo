"""Gate-Kette gegen die Handrechnung (Mini-Instanz: 2 Spuren, 3 Plätze insgesamt, Angebot a = 1, vier Zustände).

Von Hand: Gewichte π_n ∝ Π a/min(j, 2) = 1, 1, 1/2, 1/4 (Summe 11/4), also π = (4, 4, 2, 1)/11. Detailed Balance an jeder Kante, Verlust π_3 = 1/11, mittlere Zahl L = 1.
Uniformisiert mit Λ = a + c = 3: Zeilen (2/3, 1/3, 0, 0), (1/3, 1/3, 1/3, 0), (0, 2/3, 0, 1/3), (0, 0, 2/3, 1/3). Abstand d(0..7) = 10/11, 8/11, 4/11, 28/99, 148/891, 316/2673, 584/8019, 1216/24057
(d(0) = 1 − π_3; d(1) = ½(4/11 + 4/11 + |2/3 − 2/11| + |1/3 − 1/11|) = 8/11), also t_mix(1/4) = 4. Zweitgrößter Eigenwert 2/3 (det(P − 2/3·I) = 0). σ² = 310/1331 für die Verlustanzeige
(ESS je Schritt 11/31), σ² = 50/11 für die Zahl im System (ESS je Schritt 1/5)."""

from fractions import Fraction as Fr

import numpy as np
import pytest

import xab_chain as CH


def test_pi_of_the_mini_gate_is_4_4_2_1_over_11():
    pi = CH.pi_closed(2, 3, 1.0)
    assert np.allclose(pi, np.array([4, 4, 2, 1]) / 11, rtol=0, atol=1e-15)


def test_detailed_balance_and_global_balance_by_hand(mini_gate):
    pi = mini_gate.pi
    up = [Fr(1), Fr(1), Fr(1), Fr(0)]
    dn = [Fr(0), Fr(1), Fr(2), Fr(2)]
    for i in range(3):
        assert pi[i] * up[i] == pi[i + 1] * dn[i + 1]
    assert pi[3] == Fr(1, 11)


def test_uniformized_rows_of_the_mini_gate():
    p, lam = CH.uniformized(2, 3, 1.0)
    assert lam == 3.0
    expected = np.array([[2 / 3, 1 / 3, 0, 0], [1 / 3, 1 / 3, 1 / 3, 0], [0, 2 / 3, 0, 1 / 3], [0, 0, 2 / 3, 1 / 3]])
    assert np.allclose(p, expected, rtol=0, atol=1e-15)
    assert np.allclose(p.sum(axis=1), 1.0, rtol=0, atol=1e-15)


def test_tv_curve_by_hand():
    pi = CH.pi_closed(2, 3, 1.0)
    p, _ = CH.uniformized(2, 3, 1.0)
    d = CH.tv_curve(p, pi, 7)
    expected = [Fr(10, 11), Fr(8, 11), Fr(4, 11), Fr(28, 99), Fr(148, 891), Fr(316, 2673), Fr(584, 8019), Fr(1216, 24057)]
    assert np.allclose(d, [float(x) for x in expected], rtol=1e-12, atol=0)
    assert CH.t_mix(p, pi) == 4
    assert d[3] > 0.25 >= d[4]


def test_second_eigenvalue_is_two_thirds_and_t_rel_three():
    p, _ = CH.uniformized(2, 3, 1.0)
    ev = np.sort(np.linalg.eigvals(p).real)
    assert ev[-1] == pytest.approx(1.0, abs=1e-12)
    assert ev[-2] == pytest.approx(2 / 3, abs=1e-12)
    assert CH.t_rel(p) == pytest.approx(3.0, abs=1e-9)


def test_asymptotic_variance_and_ess_by_hand():
    p, _ = CH.uniformized(2, 3, 1.0)
    pi = CH.pi_closed(2, 3, 1.0)
    s2, var = CH.asymptotic_variance(p, pi, np.array([0.0, 0, 0, 1]))
    assert s2 == pytest.approx(310 / 1331, rel=1e-12) and var == pytest.approx(10 / 121, rel=1e-12)
    assert var / s2 == pytest.approx(11 / 31, rel=1e-12)
    s2n, varn = CH.asymptotic_variance(p, pi, np.arange(4.0))
    assert s2n == pytest.approx(50 / 11, rel=1e-12) and varn / s2n == pytest.approx(1 / 5, rel=1e-12)


def test_steps_for_band_by_hand():
    # T = 1,96² σ² / ε² mit σ² = 310/1331 und ε = 0,01
    assert round(CH.steps_for_band(310 / 1331, 0.01)) == 8947


def test_mini_gate_metrics():
    pi = CH.pi_closed(2, 3, 1.0)
    m = CH.gate_metrics(2, 3, 1.0, pi)
    assert m.blocking == pytest.approx(1 / 11, rel=1e-12)
    assert m.mean_n == pytest.approx(1.0, rel=1e-12)
    assert m.utilization == pytest.approx((10 / 11) / 2, rel=1e-12)       # E[min(n, 2)] = a (1 − π_K) = 10/11, geteilt durch c
    # Lq = Σ max(n − 2, 0) π_n = 1·π_3 = 1/11; Wartezeit nach Little = Lq / (a (1 − π_K)) = (1/11) / (10/11) = 1/10
    assert m.mean_queue == pytest.approx(1 / 11, rel=1e-12) and m.mean_wait == pytest.approx(0.1, rel=1e-12)


def test_exact_gate_bundle_uses_four_ways_and_counts_equations():
    ex = CH.exact_gate(2, 3, 1.0)
    assert ex.equations == 4 and ex.t_mix == 4
    assert ex.pi_deviation < 1e-12
    assert ex.loss == pytest.approx(1 / 11, rel=1e-12)
    assert ex.ess_loss == pytest.approx(11 / 31, rel=1e-12) and ex.ess_n == pytest.approx(0.2, rel=1e-12)
    assert ex.steps_for(0.01) == pytest.approx(CH.steps_for_band(310 / 1331, 0.01), rel=1e-12)
    assert ex.band(10000) == pytest.approx(1.96 * (310 / 1331 / 10000) ** 0.5, rel=1e-12)


def test_erlang_b_matches_the_chain_without_waiting_place():
    # K = c: kein Warteplatz, der Verlust ist die Erlang-B-Formel; c = 2, a = 1: B = (1/2)/(1 + 1 + 1/2) = 1/5
    assert CH.erlang_b(2, 1.0) == pytest.approx(0.2, rel=1e-12)
    pi = CH.pi_closed(2, 2, 1.0)
    assert pi[-1] == pytest.approx(0.2, rel=1e-12)
