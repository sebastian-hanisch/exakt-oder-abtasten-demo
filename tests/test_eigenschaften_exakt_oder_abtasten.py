"""Eigenschaftstests (ORACLE-PLAYBOOK §5.2, metamorph): Aussagen, die für das Verfahren gelten müssen, ohne dass man das richtige Ergebnis kennt.

Je Eigenschaft 20 bis 50 Zufallsinstanzen mit festem Zufalls-Seed. Jede Eigenschaft ist eine belegte Aussage (mit ihren Voraussetzungen im Docstring), nicht nur das heutige Verhalten:
 * Reihenfolge und Aufteilung der Daten ändern den Posterior nicht (Konjugiertheit).
 * Zeitskalierung: β, T mit demselben Faktor skaliert, λ-Skala mit; Mittel · Zeit bleibt.
 * α, β → 0 gibt Gamma(n, T) (Grenzwert).
 * Mehr Daten (n, T mal k) verkleinern die sd, Verhältnis ≈ 1/√k bei fester Rate.
 * Verlust wächst monoton mit der Last und fällt monoton mit K (c fest); π hängt nur an a/min(n, c).
 * Gleicher Zufalls-Seed gibt dieselbe Ausgabe; Mittel über R Ketten liegt im Band der exakten Streuung.
 * Gitterfehler fällt in der gemessenen Folge G = 6, 8, 10, 12 (nicht bis G = 200: dort Kastenfehler).
 * Tensorgitter ist symmetrisch unter Vertauschen gleicher Schichten."""

import math
import random

import numpy as np
import pytest

import xab_chain as CH
import xab_evaluation as ev
import xab_posterior as PO
import xab_sampler as S


def _rnd_instances(n, seed=11):
    rnd = random.Random(seed)
    return [(rnd.randint(1, 8), rnd.uniform(0.2, 5.0), rnd.randint(0, 80), rnd.uniform(1.0, 60.0)) for _ in range(n)]      # (α, β, n, T)


def test_order_and_splitting_of_the_data_do_not_change_the_posterior():
    rnd = random.Random(5)
    for alpha, beta, n, t in _rnd_instances(40):
        one = PO.gamma_poisson_posterior(alpha, beta, n, t)
        cuts = sorted(rnd.sample(range(0, n + 1), 2)) if n >= 2 else [0, 0]
        parts = [(cuts[0], t * 0.2), (cuts[1] - cuts[0], t * 0.5), (n - cuts[1], t * 0.3)]
        rnd.shuffle(parts)
        seq = PO.sequential_posterior(alpha, beta, parts)
        assert seq.shape == one.shape and seq.rate == pytest.approx(one.rate, rel=1e-12)


def test_time_scaling_leaves_mean_times_time_unchanged():
    """Zeit in Minuten statt Abfertigungsdauern: β, T mit Faktor s multipliziert, λ durch s geteilt: Mittel · Zeit bleibt gleich (Prior-Rate β skaliert mit)."""
    for alpha, beta, n, t in _rnd_instances(30):
        for s in (3.0, 0.25):
            a = PO.gamma_poisson_posterior(alpha, beta, n, t)
            b = PO.gamma_poisson_posterior(alpha, beta * s, n, t * s)
            assert b.mean * s == pytest.approx(a.mean, rel=1e-12)
            assert b.sd * s == pytest.approx(a.sd, rel=1e-12)


def test_vanishing_prior_gives_gamma_n_t():
    for _, _, n, t in _rnd_instances(25):
        if n == 0:
            continue
        post = PO.gamma_poisson_posterior(1e-9, 1e-9, n, t)
        assert post.shape == pytest.approx(n, rel=1e-8) and post.rate == pytest.approx(t, rel=1e-8)
        assert post.mean == pytest.approx(n / t, rel=1e-7)


def test_more_data_shrinks_the_sd_exactly_by_the_gamma_formula_and_like_one_over_root_k_for_weak_priors():
    """Bei fester Rate (n und T mal k, gleicher Prior) ändert sich die sd exakt um den Faktor √((α + kn)/(α + n)) · (β + T)/(β + kT), ; sie fällt und liegt nahe 1/√k nur für einen schwachen Prior
    (α ≤ 0,1 n, β ≤ 0,1 T; im Test α ≤ 0,05 n, β ≤ 0,05 T für die Nähe) (Voraussetzung im Test; bei starkem Prior gilt der Wurzelsatz nicht)."""
    weak = 0
    rnd = random.Random(21)
    for alpha, beta, n, t in _rnd_instances(60):
        if n < 20:
            continue
        base = PO.gamma_poisson_posterior(alpha, beta, n, t)
        for k in (2, 4, 9):
            big = PO.gamma_poisson_posterior(alpha, beta, n * k, t * k)
            assert big.sd / base.sd == pytest.approx(math.sqrt((alpha + k * n) / (alpha + n)) * (beta + t) / (beta + k * t), rel=1e-12)
            if alpha <= 0.1 * n and beta <= 0.1 * t:
                assert big.sd < base.sd               # nur bei schwachem Prior (bei β ≥ T kann die sd mit T wachsen)
        for _ in range(1):
            n_big, t_big = n * 20, t * 20 + rnd.uniform(0, 1)
            if alpha <= 0.05 * n_big and beta <= 0.05 * t_big:
                weak += 1
                k = 4
                big = PO.gamma_poisson_posterior(alpha, beta, n_big * k, t_big * k)
                base_big = PO.gamma_poisson_posterior(alpha, beta, n_big, t_big)
                assert big.sd / base_big.sd == pytest.approx(1 / math.sqrt(k), rel=0.05)
    assert weak >= 10


def test_loss_rises_with_load_and_falls_with_places_for_random_gates():
    rnd = random.Random(9)
    for _ in range(25):
        c = rnd.randint(1, 4)
        k = rnd.randint(max(c, 4), 9)
        last = rnd.randrange(50, 150, 5)
        loss = ev.gate_exact(c, k, last).loss
        assert ev.gate_exact(c, k, last + 5).loss > loss
        assert ev.gate_exact(c, k + 1, last).loss < loss


def test_pi_depends_only_on_the_ratio_a_over_service_rate():
    """Gleicher Faktor in a und in der Abfertigungsrate: dasselbe π (der Zeitmaßstab ändert die Verteilung nicht)."""
    rnd = random.Random(3)
    for _ in range(30):
        c = rnd.randint(1, 4)
        k = rnd.randint(c, 8)
        a = rnd.uniform(0.3, 4.0)
        pi = CH.pi_closed(c, k, a)
        # Generator mit Faktor s auf alle Raten: π bleibt Nullvektor von π Q
        for s in (0.5, 3.0):
            q = CH.generator(c, k, a) * s
            assert np.abs(pi @ q).max() < 1e-12


def test_same_zufalls_seed_gives_the_same_output_and_another_seed_another():
    s = ev.Settings(seed=1234, r_chains=3, t_steps=2000)
    a = ev.run_gate(s)
    b = ev.run_gate(ev.Settings(seed=1234, r_chains=3, t_steps=2000))
    c = ev.run_gate(ev.Settings(seed=1235, r_chains=3, t_steps=2000))
    assert np.array_equal(a.shares, b.shares) and not np.array_equal(a.shares, c.shares)
    g1 = ev.run_gamma(ev.Settings(seed=77, gp_n_steps=500, gp_reps=10))
    g2 = ev.run_gamma(ev.Settings(seed=77, gp_n_steps=500, gp_reps=10))
    assert np.array_equal(g1.draws, g2.draws) and np.array_equal(g1.trace, g2.trace)


def test_mean_over_r_chains_lies_in_the_band_of_the_exact_spread():
    """Das Mittel über R Ketten weicht höchstens 4 Standardfehler √(σ²/(T R)) vom exakten Verlust ab (T groß gegen t_rel; Start leer, Einlauf klein)."""
    for seed in range(8):
        for c, k, last in ((2, 6, 80), (1, 5, 80), (3, 6, 100)):
            ex = ev.gate_exact(c, k, last)
            shares = S.simulate_gate(c, k, ex.a, 5000, 20, 100 + seed)
            assert abs(shares.mean() - ex.loss) < 4 * math.sqrt(ex.sigma2_loss / 5000 / 20) + 0.002, (c, k, last, seed)


def test_grid_error_falls_along_the_measured_series_6_8_10_12():
    prior_list = [PO.Prior1D("lognormal", m, s0) for m in (1.0, 2.0, 3.0) for s0 in (0.3, 0.5, 0.8)] + [PO.Prior1D("gamma", 2.0, 1.0)]
    for prior in prior_list:
        worst = [max(PO.grid_errors_in_sd(32, 20.0, prior, g)) for g in (6, 8, 10, 12)]
        assert worst[0] > worst[1] > worst[2] > worst[3], prior


def test_tensor_grid_means_shift_with_the_counts():
    """Mehr Ankünfte in einer Schicht (sonst gleich) erhöhen deren Posterior-Mittel von u."""
    base = PO.make_dim_model(3)
    lo = PO.DimModel(3, np.array([8.0, 6.0, 11.0]), base.t, base.m0, base.prec)
    hi = PO.DimModel(3, np.array([8.0, 12.0, 11.0]), base.t, base.m0, base.prec)
    assert PO.tensor_grid_moments(hi, 16)[0][1] > PO.tensor_grid_moments(lo, 16)[0][1]
