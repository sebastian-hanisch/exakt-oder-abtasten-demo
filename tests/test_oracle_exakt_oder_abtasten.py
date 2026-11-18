"""Orakel-Tests: jede Kernrechnung gegen einen anderen Rechenweg (ORACLE-PLAYBOOK §2, §5; DEMO-PLAYBOOK §12).

Orakel (jeweils ein anderer Rechenweg als im Kern):
  (a) π des Gates: numpy-Kern (geschlossene Form, lineares System, Eigenvektor, P⁵⁰⁰⁰) gegen Brüche (`fractions`, Gauß-Verfahren) für die fünf Voreinstellungen und 40 Zufallsinstanzen.
  (b) d(t) und t_mix: Matrixpotenzen in numpy gegen Matrixprodukt in Brüchen.
  (c) σ² und ESS je Schritt: Formel mit Z gegen Autokovarianzsumme und gegen Brüche.
  (d) Gamma–Poisson: Posterior gegen Quadratur der unnormierten Dichte (scipy.integrate.quad), Evidenz gegen negativ-binomial und Quadratur.
  (e) Gitter in einer Dimension gegen scipy.integrate.quad (Kastenfehler beachten).
  (f) Tensorgitter: d = 1 gegen quad, d = 2 gegen dblquad, d = 3 gegen ein feineres Gitter und gegen die vorgerechnete Referenz.
  (g) Ring: Z(λ) = Spur(Tⁿ) gegen Aufzählung aller Belegungen (ganzzahlig).
Das Orakel (a), (b), (c) wurde vorher an der Mini-Instanz von Hand geprüft (tests/test_chain.py)."""

import math
import random
from fractions import Fraction as Fr

import numpy as np
import pytest
from scipy import integrate, stats

import xab_chain as CH
import xab_evaluation as EV
import xab_posterior as PO
import xab_reference as REF
from fraktionen import FractionGate

PRESETS = [(2, 6, Fr(8, 5)), (1, 6, Fr(4, 5)), (2, 6, Fr(3)), (4, 4, Fr(16, 5)), (2, 5, Fr(8, 5))]       # (c, K, a): vier Voreinstellungen und K = 5


def _random_instances(n, seed=2024):
    rnd = random.Random(seed)
    out = []
    for _ in range(n):
        c = rnd.randint(1, 3)
        k = rnd.randint(c, 5)
        a = Fr(rnd.randint(1, 12), rnd.choice([2, 3, 4, 5]))
        out.append((c, k, a))
    return out


@pytest.mark.parametrize("c,k,a", PRESETS)
def test_pi_four_ways_match_fractions_for_the_presets(c, k, a):
    fg = FractionGate(c, k, a)
    assert fg.pi_by_linear_system() == fg.pi                          # Brüche: zwei Rechenwege gleich
    exact = np.array([float(x) for x in fg.pi])
    for name, vec in CH.pi_four_ways(c, k, float(a)).items():
        assert np.abs(vec - exact).max() < 1e-12, name
    assert CH.max_spread(CH.pi_four_ways(c, k, float(a))) < 1e-12


def test_pi_matches_fractions_on_random_instances():
    for c, k, a in _random_instances(40):
        fg = FractionGate(c, k, a)
        assert fg.pi_by_linear_system() == fg.pi
        assert np.allclose(CH.pi_closed(c, k, float(a)), [float(x) for x in fg.pi], rtol=0, atol=1e-13)


def test_pi_is_stationary_and_reversible_for_the_presets():
    for c, k, a in PRESETS:
        p, _ = CH.uniformized(c, k, float(a))
        pi = CH.pi_closed(c, k, float(a))
        assert np.abs(pi @ p - pi).max() < 1e-15
        flow = pi[:, None] * p
        assert np.abs(flow - flow.T).max() < 1e-15


def test_tv_curve_and_t_mix_match_fractions():
    for c, k, a in _random_instances(12, seed=7) + [(2, 3, Fr(1)), (2, 5, Fr(8, 5))]:
        fg = FractionGate(c, k, a)
        p, _ = CH.uniformized(c, k, float(a))
        pi = CH.pi_closed(c, k, float(a))
        curve = CH.tv_curve(p, pi, 10)
        exact = fg.d_curve(10)
        assert np.allclose(curve, [float(x) for x in exact], rtol=0, atol=1e-12)
        tm_exact = fg.t_mix(t_max=60)
        tm = CH.t_mix(p, pi)
        assert tm == tm_exact


def test_asymptotic_variance_matches_fractions_and_autocovariance_sum():
    for c, k, a in [(2, 3, Fr(1)), (2, 4, Fr(3, 2)), (1, 4, Fr(1, 2)), (3, 5, Fr(5, 2))]:
        fg = FractionGate(c, k, a)
        p, _ = CH.uniformized(c, k, float(a))
        pi = CH.pi_closed(c, k, float(a))
        for f in (np.eye(k + 1)[k], np.arange(k + 1.0)):
            s2, var = CH.asymptotic_variance(p, pi, f)
            fs2, fvar = fg.sigma2([Fr(int(x)) for x in f])
            assert s2 == pytest.approx(float(fs2), rel=1e-10) and var == pytest.approx(float(fvar), rel=1e-12)
            assert CH.autocov_sum_variance(p, pi, f, 2000) == pytest.approx(s2, rel=1e-10)


def test_asymptotic_variance_of_the_preset_loss_matches_the_autocovariance_sum():
    for c, k, a in PRESETS:
        p, _ = CH.uniformized(c, k, float(a))
        pi = CH.pi_closed(c, k, float(a))
        f = np.zeros(k + 1)
        f[k] = 1.0
        s2, _ = CH.asymptotic_variance(p, pi, f)
        assert CH.autocov_sum_variance(p, pi, f, 3000) == pytest.approx(s2, rel=1e-9)


def test_erlang_b_oracle_for_the_four_server_preset():
    pi = CH.pi_closed(4, 4, 3.2)
    assert pi[-1] == pytest.approx(CH.erlang_b(4, 3.2), rel=1e-12)


def test_gamma_poisson_posterior_matches_quadrature_of_the_unnormalised_density():
    alpha, beta, n, t = 2, 1.0, 32, 20.0
    post = PO.gamma_poisson_posterior(alpha, beta, n, t)
    unn = lambda lam: lam ** (alpha - 1 + n) * math.exp(-(beta + t) * lam)
    z, _ = integrate.quad(unn, 0, 60, epsabs=0, epsrel=1e-13, limit=400)
    assert z == pytest.approx(math.gamma(post.shape) / post.rate ** post.shape, rel=1e-10)
    m1, _ = integrate.quad(lambda x: x * unn(x), 0, 60, epsabs=0, epsrel=1e-13, limit=400)
    m2, _ = integrate.quad(lambda x: x * x * unn(x), 0, 60, epsabs=0, epsrel=1e-13, limit=400)
    assert m1 / z == pytest.approx(post.mean, rel=1e-10)
    assert math.sqrt(m2 / z - (m1 / z) ** 2) == pytest.approx(post.sd, rel=1e-9)
    lo, hi = post.interval()
    cdf_hi, _ = integrate.quad(lambda x: unn(x) / z, 0, hi, epsabs=0, epsrel=1e-12)
    assert cdf_hi == pytest.approx(0.975, abs=1e-9) and lo < post.mean < hi


def test_evidence_matches_negative_binomial_and_quadrature():
    alpha, beta, n, t = 2, 1.0, 32, 20.0
    prior = lambda lam: beta ** alpha / math.gamma(alpha) * lam ** (alpha - 1) * math.exp(-beta * lam)
    lik = lambda lam: (lam * t) ** n * math.exp(-lam * t) / math.factorial(n)
    ev_q, _ = integrate.quad(lambda x: prior(x) * lik(x), 0, 80, epsabs=0, epsrel=1e-12, limit=400)
    assert PO.evidence(alpha, beta, n, t) == pytest.approx(ev_q, rel=1e-9)
    assert PO.evidence(alpha, beta, n, t) == pytest.approx(1.570427e-2, rel=1e-6)


def _quad_moments_u(n, t, prior):
    u_star, sd = PO.laplace_1d(n, t, prior)
    lo, hi = u_star - 12 * sd, u_star + 12 * sd
    lpmax = float(PO.log_post_u(np.array([u_star]), n, t, prior)[0])
    dens = lambda x: math.exp(float(PO.log_post_u(np.array([x]), n, t, prior)[0]) - lpmax)
    z, _ = integrate.quad(dens, lo, hi, epsabs=0, epsrel=1e-13, limit=500, points=[u_star])
    mu, _ = integrate.quad(lambda x: x * dens(x), lo, hi, epsabs=0, epsrel=1e-13, limit=500, points=[u_star])
    mu /= z
    v, _ = integrate.quad(lambda x: (x - mu) ** 2 * dens(x), lo, hi, epsabs=0, epsrel=1e-13, limit=500, points=[u_star])
    ml, _ = integrate.quad(lambda x: math.exp(x) * dens(x), lo, hi, epsabs=0, epsrel=1e-13, limit=500, points=[u_star])
    return mu, math.sqrt(v / z), ml / z


@pytest.mark.parametrize("prior", [PO.Prior1D("lognormal", 2.0, 0.5), PO.Prior1D("lognormal", 0.7, 1.2), PO.Prior1D("gamma", 2.0, 1.0)])
def test_one_dimensional_grid_matches_scipy_quad(prior):
    ref = _quad_moments_u(32, 20.0, prior)
    for g, tol in ((16, 1e-5), (200, 1e-5)):                       # ab G = 16 bleibt nur der Kastenfehler (Kasten ± 6 sd)
        m = PO.grid_moments_1d(32, 20.0, prior, g)
        assert m.mean_u == pytest.approx(ref[0], abs=tol * ref[1])
        assert m.sd_u == pytest.approx(ref[1], rel=tol)
        assert m.mean_lambda == pytest.approx(ref[2], abs=tol * ref[1])
    fine = PO.reference_moments_1d(32, 20.0, prior)
    assert fine.mean_u == pytest.approx(ref[0], abs=1e-6 * ref[1]) and fine.mean_lambda == pytest.approx(ref[2], abs=1e-6 * ref[1])


def test_lognormal_posterior_mean_differs_from_the_conjugate_one():
    ref = _quad_moments_u(32, 20.0, PO.Prior1D("lognormal", 2.0, 0.5))
    assert ref[2] == pytest.approx(1.64215, abs=5e-6)                  # Quadratur: E[λ] unter Lognormal-Prior (Median 2, s₀ = 0,5)
    assert ref[2] - 34 / 21 == pytest.approx(0.0231, abs=5e-5)


def test_one_dimensional_grid_error_is_coarse_at_g4_and_small_at_g10():
    prior = PO.Prior1D("lognormal", 2.0, 0.5)
    assert max(PO.grid_errors_in_sd(32, 20.0, prior, 4)) > 0.3
    assert max(PO.grid_errors_in_sd(32, 20.0, prior, 10)) < 0.01


def test_tensor_grid_d1_matches_quad():
    model = PO.make_dim_model(1)
    umode, cov = PO.laplace_dim(model)
    sd = float(np.sqrt(cov[0, 0]))
    f = lambda x: math.exp(float(model.log_post(np.array([x]))) - float(model.log_post(umode)))
    z, _ = integrate.quad(f, umode[0] - 12 * sd, umode[0] + 12 * sd, epsabs=0, epsrel=1e-13, limit=500, points=[umode[0]])
    m, _ = integrate.quad(lambda x: x * f(x), umode[0] - 12 * sd, umode[0] + 12 * sd, epsabs=0, epsrel=1e-13, limit=500, points=[umode[0]])
    mean, sds = PO.tensor_grid_moments(model, 400)
    assert mean[0] == pytest.approx(m / z, abs=1e-6)


def test_tensor_grid_d2_matches_dblquad():
    model = PO.make_dim_model(2)
    umode, cov = PO.laplace_dim(model)
    sdl = np.sqrt(np.diag(cov))
    f = lambda y, x: math.exp(float(model.log_post(np.array([x, y]))) - float(model.log_post(umode)))
    lo, hi = umode - 9 * sdl, umode + 9 * sdl
    z, _ = integrate.dblquad(f, lo[0], hi[0], lambda x: lo[1], lambda x: hi[1], epsabs=1e-12, epsrel=1e-10)
    m, _ = integrate.dblquad(lambda y, x: x * f(y, x), lo[0], hi[0], lambda x: lo[1], lambda x: hi[1], epsabs=1e-12, epsrel=1e-10)
    mean, _ = PO.tensor_grid_moments(model, 100)
    assert mean[0] == pytest.approx(m / z, abs=1e-6)


def test_tensor_grid_d3_matches_a_finer_grid_and_the_frozen_reference():
    model = PO.make_dim_model(3)
    coarse = PO.tensor_grid_moments(model, 30)
    fine = PO.tensor_grid_moments(model, 50)
    assert np.allclose(coarse[0], fine[0], atol=1e-6) and np.allclose(coarse[1], fine[1], atol=1e-6)
    ref = REF.dimension()["d"]["3"]
    assert np.allclose(fine[0], ref["e_u"], atol=1e-6) and np.allclose(fine[1], ref["sd_u"], atol=1e-6)


def test_tensor_grid_is_symmetric_under_swapping_equal_layers():
    # gleiche Zählungen in zwei Schichten und symmetrischer Prior (Korrelation hängt am Abstand: Schichten 0 und 2 von d = 3 sind bei gleichen n_j vertauschbar)
    base = PO.make_dim_model(3)
    model = PO.DimModel(3, np.array([8.0, 6.0, 8.0]), base.t, base.m0, base.prec)
    mean, sd = PO.tensor_grid_moments(model, 20)
    assert mean[0] == pytest.approx(mean[2], abs=1e-12) and sd[0] == pytest.approx(sd[2], abs=1e-12)


def test_ring_partition_function_matches_enumeration_exactly():
    for n in (3, 6, 12, 20):
        lam = 3
        t = [[1, lam], [1, 0]]
        # ganzzahlige Matrixpotenz von Hand
        m = [[1, 0], [0, 1]]
        for _ in range(n):
            m = [[sum(m[i][k] * t[k][j] for k in range(2)) for j in range(2)] for i in range(2)]
        z_transfer = m[0][0] + m[1][1]
        z_enum = 0
        for x in range(1 << n):
            rot = ((x << 1) | (x >> (n - 1))) & ((1 << n) - 1)
            if x & rot == 0:
                z_enum += lam ** bin(x).count("1")
        assert z_transfer == z_enum
    assert EV.ring_z_transfer(6, 3.0) == pytest.approx(154, rel=1e-12)
    assert EV.ring_z_transfer(12, 3.0) == pytest.approx(22258, rel=1e-12)
    assert EV.ring_z_transfer(20, 3.0) == pytest.approx(17580751, rel=1e-12)   # von Hand per Rekursion Z_n = Z_{n-1} + 3 Z_{n-2} mit Z_1 = 1, Z_2 = 7: …, Z_12 = 22258, …, Z_20 = 17580751


def test_ring_enumeration_in_the_app_agrees_with_the_transfer_matrix():
    for n, states, zt, ze in EV.ring_rows():
        assert zt == pytest.approx(ze, rel=1e-12)
        # Zahl der Belegungen eines Rings aus n Plätzen ist die Lucas-Zahl L_n
        lucas = [2, 1]
        for _ in range(n):
            lucas.append(lucas[-1] + lucas[-2])
        assert states == lucas[n]


def test_ks_oracle_for_the_direct_gamma_draws_of_the_app():
    s = EV.Settings()
    gp = EV.run_gamma(s)
    assert gp.ks_p > 0.01
    assert stats.kstest(gp.draws, gp.post.dist.cdf).statistic == pytest.approx(gp.ks_d, rel=1e-12)
