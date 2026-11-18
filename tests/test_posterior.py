"""Gamma–Poisson und Gitter gegen die Handrechnung.

Von Hand (Prior Gamma(2, 1), 3 Ankünfte in 2 Zeiteinheiten): Likelihood (2λ)³e^{−2λ}/3! = (4/3)λ³e^{−2λ}, Posterior Gamma(5, 3): Mittel 5/3, Varianz 5/9, Modus 4/3.
Evidenz = (4/3)∫λ⁴e^{−3λ}dλ = (4/3)·4!/3⁵ = 32/243, ebenso negativ-binomial C(4, 3)(1/3)²(2/3)³ = 32/243. ∫λ⁴e^{−3λ}dλ = 8/81. Zwei Teildaten (1 in 1, dann 2 in 1) und
(2 in 1, dann 1 in 1) geben beide Gamma(5, 3). P(λ > 2) = 115·e^{−6} (Erlang-Summe 1 + 6 + 18 + 36 + 54 = 115). Abgeschnittener Prior (gleichverteilt auf (0, 3]): Posterior ∝ λ³e^{−2λ},
Mittel 2(1 − 115e^{−6})/(1 − 61e^{−6}) = 1,684606 (Σ_{k≤3} 6^k/k! = 61)."""

import math
from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy import integrate

import xab_posterior as PO


def test_gamma_poisson_posterior_by_hand():
    post = PO.gamma_poisson_posterior(2, 1.0, 3, 2.0)
    assert (post.shape, post.rate) == (5, 3.0)
    assert post.mean == pytest.approx(5 / 3, rel=1e-15)
    assert post.var == pytest.approx(5 / 9, rel=1e-15)
    assert post.mode == pytest.approx(4 / 3, rel=1e-15)


def test_evidence_by_hand_two_ways():
    assert PO.evidence(2, 1.0, 3, 2.0) == pytest.approx(32 / 243, rel=1e-12)
    # über das Integral: (4/3) ∫ λ⁴ e^{−3λ} dλ mit ∫ = 4!/3⁵ = 8/81
    val, _ = integrate.quad(lambda x: 4 / 3 * x ** 4 * math.exp(-3 * x), 0, 60, epsabs=0, epsrel=1e-13)
    assert val == pytest.approx(32 / 243, rel=1e-10)
    assert Fr(4, 3) * Fr(math.factorial(4), 3 ** 5) == Fr(32, 243)


def test_order_of_data_does_not_matter():
    a = PO.sequential_posterior(2, 1.0, [(1, 1.0), (2, 1.0)])
    b = PO.sequential_posterior(2, 1.0, [(2, 1.0), (1, 1.0)])
    one = PO.gamma_poisson_posterior(2, 1.0, 3, 2.0)
    assert (a.shape, a.rate) == (b.shape, b.rate) == (one.shape, one.rate)


def test_tail_probability_is_115_exp_minus_6():
    post = PO.gamma_poisson_posterior(2, 1.0, 3, 2.0)
    assert float(post.dist.sf(2.0)) == pytest.approx(115 * math.exp(-6), rel=1e-12)
    assert sum(Fr(6) ** k / math.factorial(k) for k in range(5)) == 115


def test_interval_is_the_central_95_percent_interval():
    post = PO.gamma_poisson_posterior(2, 1.0, 32, 20.0)         # Gamma(34, 21)
    lo, hi = post.interval()
    assert post.dist.cdf(lo) == pytest.approx(0.025, abs=1e-12) and post.dist.cdf(hi) == pytest.approx(0.975, abs=1e-12)
    assert (round(lo, 4), round(hi, 4)) == (1.1212, 2.2069)


def test_split_data_adds_up_to_the_whole():
    for n, t in ((32, 20.0), (3, 2.0), (0, 5.0), (100, 100.0)):
        parts = PO.split_data(n, t)
        assert sum(p[0] for p in parts) == n and sum(p[1] for p in parts) == pytest.approx(t, rel=1e-12)
        seq = PO.sequential_posterior(2, 1.0, parts)
        one = PO.gamma_poisson_posterior(2, 1.0, n, t)
        assert seq.shape == one.shape and seq.rate == pytest.approx(one.rate, rel=1e-12)


def test_truncated_prior_is_no_gamma_but_closed():
    # Prior gleichverteilt auf (0, 3]: Posterior ∝ λ³ e^{−2λ} auf (0, 3]; Mittel 2(1 − 115 e^{−6})/(1 − 61 e^{−6})
    e6 = math.exp(-6)
    mean = 2 * (1 - 115 * e6) / (1 - 61 * e6)
    assert mean == pytest.approx(1.684606, abs=1e-6)
    num, _ = integrate.quad(lambda x: x ** 4 * math.exp(-2 * x), 0, 3, epsabs=0, epsrel=1e-13)
    den, _ = integrate.quad(lambda x: x ** 3 * math.exp(-2 * x), 0, 3, epsabs=0, epsrel=1e-13)
    assert num / den == pytest.approx(mean, rel=1e-12)
    assert sum(Fr(6) ** k / math.factorial(k) for k in range(4)) == 61


def test_conjugate_prior_in_u_gives_the_gamma_posterior_through_the_grid():
    # Prior Gamma(2, 1) in u = ln λ: Posterior ∝ exp((α + n) u − (β + T) e^u) = Gamma(34, 21) in u; das Gitter trifft E[λ] = 34/21
    prior = PO.Prior1D("gamma", 2.0, 1.0)
    m = PO.grid_moments_1d(32, 20.0, prior, 24)
    assert m.mean_lambda == pytest.approx(34 / 21, rel=1e-6)
    post = PO.gamma_poisson_posterior(2, 1.0, 32, 20.0)
    assert m.sd_u == pytest.approx(float(np.sqrt(post.dist.expect(lambda x: math.log(x) ** 2) - post.dist.expect(lambda x: math.log(x)) ** 2)), rel=1e-5)


def test_laplace_mode_is_a_stationary_point():
    for prior in (PO.Prior1D("lognormal", 2.0, 0.5), PO.Prior1D("gamma", 2.0, 1.0), PO.Prior1D("lognormal", 0.5, 1.5)):
        u, sd = PO.laplace_1d(32, 20.0, prior)
        gp, _ = prior.grad_hess(u)
        assert 32 - 20.0 * math.exp(u) + gp == pytest.approx(0.0, abs=1e-9)
        assert sd > 0


def test_dim_model_symmetry_and_shapes():
    m = PO.make_dim_model(3)
    assert m.prec.shape == (3, 3) and np.allclose(m.prec, m.prec.T)
    # Kovarianz des Priors: s0² ρ^|i−j|
    cov = np.linalg.inv(m.prec)
    assert cov[0, 0] == pytest.approx(0.25, rel=1e-12) and cov[0, 2] == pytest.approx(0.25 * 0.49, rel=1e-12)
    with pytest.raises(ValueError):
        PO.make_dim_model(7)
