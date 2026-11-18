"""Zufall und Ketten: die vektorisierte SplitMix64-Folge gleich der Vorlage, Übergänge der simulierten Gate-Kette gleich den Zeilen der Matrix, direkte Ziehung, Zufallsschritt-Metropolis."""

import math
from pathlib import Path

import numpy as np
import pytest
from scipy import stats

import xab_chain as CH
import xab_posterior as PO
import xab_rng
import xab_sampler as S

TEMPLATE = Path(__file__).resolve().parents[2] / "sebastianhanisch-website" / "tools" / "templates" / "rng_splitmix64_exact.py"


def test_rng_is_an_unchanged_copy_of_the_website_template():
    if not TEMPLATE.exists():
        pytest.skip("Vorlage liegt nicht neben dem Repo")
    mine = Path(xab_rng.__file__).read_text(encoding="utf-8").splitlines()
    template = TEMPLATE.read_text(encoding="utf-8").splitlines()
    assert mine[0].startswith("# Kopie von") and mine[1:] == template


def test_vectorized_stream_equals_the_scalar_generator():
    for seed in (0, 1, 35, 7301, 2 ** 63 + 5):
        scalar = xab_rng.SplitMix64(seed)
        expected = np.array([scalar.random() for _ in range(300)])
        assert np.array_equal(S.uniforms(seed, 300), expected)
        # mit Versatz: die Zahlen 101 … 300
        assert np.array_equal(S.uniforms(seed, 200, 100), expected[100:])


def test_uniforms_are_in_unit_interval_and_look_uniform():
    u = S.uniforms(5, 100000)
    assert u.min() >= 0.0 and u.max() < 1.0
    assert stats.kstest(u, "uniform").pvalue > 0.01


def test_box_muller_matches_the_scalar_normal_up_to_rounding():
    scalar = xab_rng.SplitMix64(9)
    z_scalar = np.array([scalar.normal() for _ in range(50)])
    u = S.uniforms(9, 100)
    z_vec = S.box_muller(u[0::2], u[1::2])
    assert np.allclose(z_vec, z_scalar, rtol=0, atol=1e-12)


def test_gate_thresholds_are_the_rows_of_the_matrix():
    for c, k, a in ((2, 3, 1.0), (2, 6, 1.6), (4, 4, 3.2), (1, 6, 0.8)):
        p, _ = CH.uniformized(c, k, a)
        up, dn = S.gate_thresholds(c, k, a)
        for s in range(k + 1):
            p_up = up[s]
            p_dn = 1.0 - dn[s]
            assert p_up == pytest.approx(p[s, s + 1] if s < k else 0.0, abs=1e-15)
            assert p_dn == pytest.approx(p[s, s - 1] if s > 0 else 0.0, abs=1e-15)
            assert p_up + p_dn <= 1.0


def test_simulated_one_step_distribution_matches_the_matrix_row():
    # 20 000 Ketten, ein Schritt aus Zustand 3 der Mini-Instanz: Zeile 3 = (0, 0, 2/3, 1/3)
    states = S.simulate_gate_states(2, 3, 1.0, 1, 20000, 5, start=3)
    freq = np.bincount(states, minlength=4) / 20000
    p, _ = CH.uniformized(2, 3, 1.0)
    se = np.sqrt(p[3] * (1 - p[3]) / 20000)
    assert np.all(np.abs(freq - p[3]) <= 4.5 * se + 1e-12)


def test_simulated_state_after_t_steps_matches_matrix_power():
    p, _ = CH.uniformized(2, 3, 1.0)
    for t in (2, 4):
        states = S.simulate_gate_states(2, 3, 1.0, t, 20000, 11, start=3)
        freq = np.bincount(states, minlength=4) / 20000
        exact = np.linalg.matrix_power(p, t)[3]
        assert np.abs(freq - exact).max() < 0.02


def test_simulated_share_is_reproducible_and_depends_on_the_seed():
    a = S.simulate_gate(2, 3, 1.0, 500, 3, 7)
    assert np.array_equal(a, S.simulate_gate(2, 3, 1.0, 500, 3, 7))
    assert not np.array_equal(a, S.simulate_gate(2, 3, 1.0, 500, 3, 8))


def test_simulated_shares_scatter_like_the_exact_asymptotic_variance():
    # 300 Ketten zu 2 000 Schritten, Start leer: Streuung der Anteile ≈ √(σ²/T) (Verhältnis in [0,8; 1,2]); Mittel innerhalb 4 Standardfehlern von π_K = 1/11
    ex = CH.exact_gate(2, 3, 1.0)
    shares = S.simulate_gate(2, 3, 1.0, 2000, 300, 1)
    sd_exact = math.sqrt(ex.sigma2_loss / 2000)
    assert 0.8 <= shares.std(ddof=1) / sd_exact <= 1.2
    assert abs(shares.mean() - 1 / 11) < 4 * sd_exact / math.sqrt(300)


def test_direct_gamma_draws_follow_gamma():
    draws = S.gamma_direct(5, 3.0, 4000, 21)
    assert stats.kstest(draws, stats.gamma(5, scale=1 / 3).cdf).pvalue > 0.01
    assert draws.mean() == pytest.approx(5 / 3, abs=4 * math.sqrt(5 / 9 / 4000))


def test_direct_gamma_needs_an_integer_shape():
    with pytest.raises(ValueError):
        S.gamma_direct(2.5, 1.0, 10, 1)
    with pytest.raises(ValueError):
        S.gamma_direct(0, 1.0, 10, 1)


def test_rwm_is_reproducible_and_independent_of_the_chunking(monkeypatch):
    post = PO.gamma_poisson_posterior(2, 1, 3, 2.0)
    logp = PO.gamma_log_density(post)
    x0 = S.gamma_direct(5, 3.0, 20, 3)
    a = S.rwm_1d(logp, x0, 1.0, 300, 9, lower=0.0, checkpoints=(300,))
    monkeypatch.setattr(S, "_CHUNK_VALUES", 90)        # viele kleine Blöcke
    b = S.rwm_1d(logp, x0, 1.0, 300, 9, lower=0.0, checkpoints=(300,))
    assert np.array_equal(a.x_final, b.x_final) and np.array_equal(a.means_at[300], b.means_at[300]) and a.accept_rate == b.accept_rate


def test_rwm_chain_has_the_right_stationary_mean_and_acceptance_in_range():
    post = PO.gamma_poisson_posterior(2, 1, 32, 20.0)
    x0 = S.gamma_direct(34, 21.0, 100, 4)
    run = S.rwm_1d(PO.gamma_log_density(post), x0, 2.4 * post.sd, 2000, 5, lower=0.0, checkpoints=(2000,))
    # Mittel über 100 Ketten innerhalb 4 Standardfehlern (ESS je Schritt höchstens 0,3 angenommen: konservativ)
    se = post.sd / math.sqrt(100 * 2000 * 0.1)
    assert abs(run.means_at[2000].mean() - post.mean) < 4 * se
    assert 0.3 < run.accept_rate < 0.6


def test_rwm_rejects_proposals_below_the_lower_bound():
    post = PO.gamma_poisson_posterior(1, 1.0, 0, 5.0)        # Gamma(1, 6) nahe 0
    run = S.rwm_1d(PO.gamma_log_density(post), np.full(30, 0.01), 5.0, 200, 1, lower=0.0, keep_trace=True)
    assert run.trace.min() > 0.0


def test_rwm_multi_matches_rwm_1d_statistically_for_d_equal_1():
    model = PO.make_dim_model(1)
    umode, cov = PO.laplace_dim(model)
    sd = float(np.sqrt(cov[0, 0]))
    x0 = np.full((60, 1), umode[0])
    run = S.rwm_multi(model.log_post, x0, np.array([[2.38 * sd]]), 1000, 3, checkpoints=(1000,))
    ref_mean, ref_sd = PO.tensor_grid_moments(model, 400)
    assert abs(run.means_at[1000].mean() - ref_mean[0]) < 4 * ref_sd[0] / math.sqrt(60 * 1000 * 0.1)
    assert 0.3 < run.accept_rate < 0.6


def test_ess_helper_inverts_the_rmse_law():
    # N = 1000, RMSE 0,1 sd entspricht (1/0,1)²/1000 = 0,1 unabhängigen Ziehungen je Schritt
    assert S.ess_per_step_from_rmse(0.1, 1000) == pytest.approx(0.1, rel=1e-12)
    assert S.rmse_in_sd(np.array([1.0, 3.0]), 2.0, 2.0) == pytest.approx(0.5, rel=1e-12)
