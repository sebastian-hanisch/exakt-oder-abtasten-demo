"""Literaturwerte (ORACLE-PLAYBOOK §5.1): nur Werte aus Quellen, die eingesehen sind.

* README von markov-queue-demo (https://github.com/sebastian-hanisch/markov-queue-demo, Abschnitt „Befunde“, Stand der Einsicht 2026-11-18): Voreinstellung (zwei Spuren, 6 Plätze insgesamt,
  Last 80 %): Verlust 7,60 %, im Mittel 2,45 Lkw im System, Wartezeit der Angenommenen 1,98 min (3 min mittlere Abfertigung), Auslastung der Spuren 73,9 %; eine Spur, 6 Plätze, 80 %: Verlust 6,6 %,
  L = 2,14, Wartezeit 5,6 min; Überlast (150 %, zwei Spuren): Verlust 36,0 %, Spuren zu 96 % ausgelastet, Wartezeit 4,0 min; vier Spuren ohne Warteplatz: Verlust 22,8 %, genau Erlang B.
* Asanjarani, Nazarathy, Taylor, Queueing Systems 97 (2021), arXiv 2012.14614 (Abstract und Abschnitt 3.7, gelesen): bei einer Poisson-Ankunftsrate mit Gamma-Prior ist der Posterior wieder ein Gamma;
  die Form wächst um die Zahl der Ankünfte, die Rate um die Beobachtungsdauer.
Aus Quellen, die nicht gelesen sind (Norris, Handbook of Markov Chain Monte Carlo, Robert/Casella), steht hier kein Test."""

import pytest

import xab_chain as CH
import xab_evaluation as EV
import xab_posterior as PO


def _readme_values(c, k, last):
    ex = EV.gate_exact(c, k, last)
    m = CH.gate_metrics(c, k, ex.a, ex.pi)
    return ex, m


def test_readme_values_of_the_default_preset():
    ex, m = _readme_values(2, 6, 80)
    assert round(100 * ex.loss, 2) == 7.60
    assert round(m.mean_n, 2) == 2.45
    assert round(m.mean_wait * 3.0, 2) == 1.98
    assert round(100 * m.utilization, 1) == 73.9


def test_readme_values_of_one_lane():
    ex, m = _readme_values(1, 6, 80)
    assert round(100 * ex.loss, 1) == 6.6
    assert round(m.mean_n, 2) == 2.14
    assert round(m.mean_wait * 3.0, 1) == 5.6


def test_readme_values_of_the_overload_preset():
    ex, m = _readme_values(2, 6, 150)
    assert round(100 * ex.loss, 1) == 36.0
    assert round(100 * m.utilization) == 96
    assert round(m.mean_wait * 3.0, 1) == 4.0


def test_readme_value_of_four_lanes_without_waiting_place_is_erlang_b():
    ex, _ = _readme_values(4, 4, 80)
    assert round(100 * ex.loss, 1) == 22.8
    assert ex.loss == pytest.approx(CH.erlang_b(4, 3.2), rel=1e-12)


def test_gamma_prior_gives_gamma_posterior_with_shape_plus_n_and_rate_plus_t():
    for alpha, beta, n, t in ((2, 1.0, 32, 20.0), (1, 0.5, 0, 4.0), (7, 2.5, 100, 63.0)):
        post = PO.gamma_poisson_posterior(alpha, beta, n, t)
        assert (post.shape, post.rate) == (alpha + n, beta + t)
