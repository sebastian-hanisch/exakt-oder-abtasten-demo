"""Randfälle (DEMO-PLAYBOOK §12): n = 0 Ankünfte, sehr viele Daten, α = 1, K = c (Erlang B), c = 1, Last 150 %, R = 1, T = 1, Schrittweiten 0,5 und 12 sd,
überstreuter Start, G = 4, d = 1, nichtganze Form, n_j = 0 in einer Schicht, Obergrenze R·T."""

import numpy as np
import pytest

import xab_chain as CH
import xab_constants as C
import xab_evaluation as ev
import xab_posterior as PO
import xab_sampler as S


def test_no_arrivals_gives_gamma_alpha_beta_plus_t():
    post = PO.gamma_poisson_posterior(2, 1.0, 0, 5.0)
    assert (post.shape, post.rate) == (2, 6.0)
    assert PO.evidence(2, 1.0, 0, 5.0) == pytest.approx((1 / 6) ** 2, rel=1e-12)        # P(n = 0) = (β/(β + T))^α
    gp = ev.run_gamma(ev.Settings(n_obs=0, t_obs=5.0))
    assert gp.post.shape == 2 and gp.ks_p > 0.001 and gp.trace.min() > 0


def test_very_many_arrivals():
    gp = ev.run_gamma(ev.Settings(alpha=10, n_obs=100, t_obs=100.0, gp_n_steps=500, gp_reps=10))
    assert (gp.post.shape, gp.post.rate) == (110, 101.0)
    assert gp.draws.min() > 0 and np.isfinite(gp.ess)


def test_alpha_one_is_the_exponential_prior():
    post = PO.gamma_poisson_posterior(1, 1.0, 3, 2.0)
    assert (post.shape, post.rate, post.mode) == (4, 3.0, 1.0)


def test_k_equals_c_is_erlang_b_and_uses_the_largest_exit_rate():
    ex = ev.gate_exact(4, 4, 80)
    assert ex.loss == pytest.approx(CH.erlang_b(4, 3.2), rel=1e-12)
    assert CH.uniformization_rate(4, 4, 3.2) == pytest.approx(3.2 + 3, rel=1e-12)           # a + c − 1 für K = c
    assert CH.uniformization_rate(2, 6, 1.6) == pytest.approx(3.6, rel=1e-12)               # a + c für K > c
    run = ev.run_gate(ev.Settings(c=4, k=4, r_chains=2, t_steps=2000))
    assert run.erlang_b == pytest.approx(run.exact.loss, rel=1e-12) and run.metrics.mean_wait == 0.0


def test_single_lane_and_overload_extremes_are_finite():
    for c, k, last in ((1, 4, 50), (1, 10, 150), (4, 10, 150), (4, 4, 50), (4, 10, 50)):
        ex = ev.gate_exact(c, k, last)
        assert ex.t_mix is not None and np.isfinite(ex.sigma2_loss) and 0 < ex.ess_loss <= 1 and 0 < ex.ess_n <= 1
        assert ex.pi_deviation < 1e-10 and abs(ex.pi.sum() - 1) < 1e-12


def test_loss_rises_with_load_and_falls_with_k():
    losses = [ev.gate_exact(2, 6, last).loss for last in range(50, 155, 5)]
    assert all(b > a for a, b in zip(losses, losses[1:]))
    by_k = [ev.gate_exact(2, k, 80).loss for k in range(4, 11)]
    assert all(b < a for a, b in zip(by_k, by_k[1:]))


def test_one_chain_and_one_step():
    run = ev.run_gate(ev.Settings(r_chains=1, t_steps=1000))
    assert run.shares is not None and len(run.shares) == 1 and run.coverage in (0.0, 1.0)
    shares = S.simulate_gate(2, 6, 1.6, 1, 5, 3)
    assert set(shares.tolist()) <= {0.0, 1.0}


def test_simulation_cap_gives_no_simulation_but_exact_values():
    over = ev.Settings(r_chains=400, t_steps=20000)
    assert not ev.sim_allowed(over.r_chains, over.t_steps)
    run = ev.run_gate(over)
    assert run.shares is None and run.coverage is None and run.exact.equations == 7
    assert ev.sim_allowed(1000, 1000) and not ev.sim_allowed(1000, 1001)
    assert ev.sim_allowed(C.GATE_R_MAX, C.SIM_CAP // C.GATE_R_MAX)


def test_extreme_step_sizes_of_the_chain_give_a_standing_chain():
    gp_small = ev.run_gamma(ev.Settings(gp_step=0.5, gp_n_steps=1000, gp_reps=30))
    gp_large = ev.run_gamma(ev.Settings(gp_step=12.0, gp_n_steps=1000, gp_reps=30))
    assert gp_small.accept > 0.7 and gp_large.accept < 0.2
    assert gp_small.ess < 0.15 and gp_large.ess < 0.15                  # „die Kette steht“: weit unter der direkten Ziehung


def test_overdispersed_start_without_burn_in_is_worse_than_the_equilibrium_start():
    post = PO.gamma_poisson_posterior(2, 1.0, 32, 20.0)
    logp = PO.gamma_log_density(post)
    over = np.array([0.3, 1.0, 2.5, 4.0] * 25)
    run_over = S.rwm_1d(logp, over, 2.4 * post.sd, 200, 3, lower=0.0, checkpoints=(200,))
    x0 = S.gamma_direct(34, 21.0, 100, 4)
    run_eq = S.rwm_1d(logp, x0, 2.4 * post.sd, 200, 3, lower=0.0, checkpoints=(200,))
    r_over = S.rmse_in_sd(run_over.means_at[200], post.mean, post.sd)
    r_eq = S.rmse_in_sd(run_eq.means_at[200], post.mean, post.sd)
    assert r_over > r_eq


def test_coarse_grid_g4_has_large_errors():
    run = ev.run_grid(ev.Settings(grid_g=4, ln_n_steps=500, ln_reps=10))
    assert max(run.err) > 0.1 and run.g_need == 10
    kind, text = ev.verdict_grid("x", False, 4, run.err, run.g_need, run.ref.mean_lambda, run.conj_mean, run.chain_steps_001)
    assert kind == "warning" and "Gitter zu grob" in text


def test_grid_g_200_hits_the_reference_closely():
    run = ev.run_grid(ev.Settings(grid_g=200, prior=C.PRIOR_LOGNORMAL, ln_n_steps=500, ln_reps=10))
    assert max(run.err) < 1e-5


def test_lognormal_prior_extremes_stay_finite():
    for median, s0 in ((0.5, 0.2), (5.0, 1.5), (0.5, 1.5), (5.0, 0.2)):
        run = ev.run_grid(ev.Settings(prior=C.PRIOR_LOGNORMAL, median=median, s0=s0, ln_n_steps=500, ln_reps=10))
        assert np.isfinite(run.err).all() and run.ref.sd_u > 0 and np.isfinite(run.ess)


def test_one_dimension_is_the_single_rate_posterior():
    dim = ev.run_dim(ev.Settings(dim_d=1, dim_g=10))
    assert dim.points == 10 and max(dim.live_err) < 1e-3
    assert dim.rows[0][1] == 10


def test_every_dimension_and_grid_pair_within_the_cap():
    assert C.DIM_G_MAX ** C.DIM_D_MAX <= C.DIM_POINT_CAP
    run = ev.run_dim(ev.Settings(dim_d=6, dim_g=12))
    assert run.points == 12 ** 6 and max(run.live_err) < 1e-4


def test_zero_count_in_a_layer_keeps_the_posterior_finite():
    base = PO.make_dim_model(3)
    model = PO.DimModel(3, np.array([8.0, 0.0, 11.0]), base.t, base.m0, base.prec)
    umode, cov = PO.laplace_dim(model)
    mean, sd = PO.tensor_grid_moments(model, 20)
    assert np.isfinite(umode).all() and np.isfinite(mean).all() and (sd > 0).all()
    assert mean[1] < mean[0] and mean[1] < mean[2]                    # die Schicht ohne Ankünfte hat die kleinste Rate


def test_non_integer_shape_is_not_offered():
    with pytest.raises(ValueError):
        S.gamma_direct(34.5, 21.0, 10, 1)
    assert C.GP_ALPHA_MIN >= 1 and isinstance(C.DEFAULTS["alpha"], int)
    spec_types = {type(C.DEFAULTS["alpha"]), type(C.DEFAULTS["n_obs"])}
    assert spec_types == {int}


def test_ring_enumeration_for_the_smallest_sizes():
    assert ev.ring_z_enumeration(3, 1.0) == (4.0, 4)                  # Ring aus 3 Plätzen: leer und drei Einzelplätze, Z(1) = 4 Belegungen
    assert ev.ring_z_transfer(3, 1.0) == pytest.approx(4.0)
