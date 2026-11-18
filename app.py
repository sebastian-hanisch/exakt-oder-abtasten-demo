"""Wann man nicht abtasten muss - interaktive Konzept-Demo
Sebastian Hanisch - Operations Research und Machine Learning

Dieselbe Frage zweimal: einmal ausgerechnet (Gleichungen, Formel, Gitter), einmal abgetastet (Kette). Zähleinheiten: Gleichungen, Dichteauswertungen, Schritte.
Vier Experimente: Gate-Kette, Posterior einer Ankunftsrate (Gamma–Poisson), nichtkonjugater Prior (Gitter), mehrere gekoppelte Raten (wo die Kette gebraucht wird).

Lauffähig mit: streamlit run app.py
"""

import numpy as np
import pandas as pd
import streamlit as st

import xab_chain as CH
import xab_constants as C
import xab_evaluation as ev
import xab_format as F
import xab_posterior as PO
import xab_reference as REF
import xab_texts as TX
from xab_layout import dunkles_design, kopf_css
from xab_presets import (
    PRESET_KEYS,
    bounds,
    init_session_state_defaults,
    load_permalink_settings,
    randomize_seed,
    sidebar_presets,
    sync_query_params,
)
from xab_visualization import (
    build_dim_cost,
    build_dim_rmse,
    build_ess_steps,
    build_gamma_density,
    build_gate_pi,
    build_gate_shares,
    build_grid_density,
    build_grid_error,
    build_overview,
)

st.set_page_config(page_title="Wann man nicht abtasten muss – Sebastian Hanisch", layout="wide")


@st.cache_data(show_spinner=False)
def _gate(s):
    return ev.run_gate(s)


@st.cache_data(show_spinner="Kette und direkte Ziehungen laufen …")
def _gamma(s):
    return ev.run_gamma(s)


@st.cache_data(show_spinner="Gitter und Kette laufen …")
def _grid(s):
    return ev.run_grid(s)


@st.cache_data(show_spinner="Tensorgitter wird ausgewertet …")
def _dim(s):
    return ev.run_dim(s)


@st.cache_data(show_spinner=False)
def _ring():
    return ev.ring_rows()


def _slider(label, key, step=None, fmt=None, help=None, disabled=False):
    lo, hi = bounds(key)
    kwargs = {}
    if step is not None:
        kwargs["step"] = step
    if fmt is not None:
        kwargs["format"] = fmt
    return st.slider(label, lo, hi, key=key, help=help, disabled=disabled, **kwargs)


# Fester Kopf: nur der Titel bleibt beim Scrollen oben (Stylesheet in xab_layout.py).
st.html(kopf_css(dunkles_design()))
with st.container(key="kopf"):
    st.title("🧮 Wann man nicht abtasten muss")
st.markdown(
    """
Bevor man eine Kette baut, um eine Verteilung abzutasten, lohnt die Frage: **lässt sich das Ziel ausrechnen?** Diese Seite stellt dieselbe Frage viermal (der Verlustanteil eines Gates,
der Posterior einer Ankunftsrate mit Gamma-Prior, mit einem anderen Prior und mit mehreren gekoppelten Raten), einmal **ausgerechnet** und einmal **abgetastet**, und zählt
**Gleichungen, Dichteauswertungen und Schritte**. Das Ergebnis ist unbequem: in diesen ausgedachten Modellen verliert die Kette bis zu fünf gekoppelten Parametern, erst bei sechs gewinnt sie (Genauigkeit 0,01 sd, Gitter mit 10 Punkten je Achse).

Weiter unten: **❓ Kernfragen**, vier Experimente mit je einem Verdikt, der Expander **🧰 Was hilft das Ausrechnen?**, der Abschnitt **🚧 Wo die Annahmen enden** und der Expander
**📐 Mathematische Formulierung**.
"""
)

# erklaerseite:einfach:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/exakt-oder-abtasten-erklaert.json, nicht von Hand ändern)
with st.expander("💬 In einfachen Worten"):
    st.markdown(
        """
Stellen Sie sich ein Abfertigungs-Gate vor und die Frage, in welchem Anteil der Zeit es voll ist. Man kann den Anteil **ausrechnen**, mit einem kleinen Gleichungssystem, oder eine **Zufallswanderung** durch die möglichen Zustände laufen lassen und zählen: das **Abtasten** mit einer Markov-Kette. Die Demo stellt dieselbe Frage mehrmals, an einem Gate und an Ankunftsraten, und zählt, was jeder Weg kostet: Gleichungen, Auswertungen der Dichte oder Schritte der Wanderung. Sie zeigt, dass in diesen ausgedachten Modellen das Ausrechnen bis zu fünf gekoppelte Raten reicht (Gitter mit 10 Punkten je Rate, Genauigkeit ein Hundertstel der Streuung) und die Kette erst ab sechs gewinnt, hier ein Zufallsschritt-Metropolis mit festem Schritt. Voraussetzung ist, dass sich das Ziel an jedem Punkt auswerten lässt und seine Normierung bekannt oder ausrechenbar ist. Alles gilt für diese Modelle, nicht für einen echten Betrieb.

[Ausführlich und verständlich erklärt, mit Fachteil](https://sebastianhanisch.net/exakt-oder-abtasten-erklaert.html)
        """
    )
# erklaerseite:einfach:end
st.info(TX.HONESTY_MARKDOWN)

with st.expander("So funktioniert die Frage „ausrechnen oder abtasten?“", expanded=True):
    st.markdown(TX.HOW_IT_WORKS_MARKDOWN)

st.caption("🔗 Die Adresszeile oben spiegelt Ihre aktuelle Konfiguration wider – einfach kopieren, um ein Szenario zu teilen.")

load_permalink_settings()
init_session_state_defaults()

with st.sidebar:
    sidebar_presets()  # Schnellstart oben, das aktive Preset ist hervorgehoben
    st.header("⚙️ Einstellungen")
    st.caption("Die Regler der Experimente stehen bei den Experimenten.")
    st.number_input("Zufalls-Seed", *bounds("seed_input"), key="seed_input", step=1, help="Startwert des Zufallsgenerators (SplitMix64) für alle Ketten und Ziehungen.")
    st.button("🎲 Neuer Zufalls-Seed", width="stretch", on_click=randomize_seed)

# --- Alle Regler vor der Rechnung auslesen (sie stehen bei den Experimenten, die Rechnung braucht sie schon oben) -----------------------------
# Streamlit zeichnet in Skriptreihenfolge; damit die Kernfragen vor den Experimenten stehen, zeichnen wir die Experimente in Containern, die vorher angelegt werden.
top = st.container()
kernfragen = st.container()
st.markdown("---")
box1 = st.container()
st.markdown("---")
box2 = st.container()
st.markdown("---")
box3 = st.container()
st.markdown("---")
box4 = st.container()
st.markdown("---")

# --- Experiment 1 ----------------------------------------------------------------------------------------------------------------------------
with box1:
    st.subheader("🔬 Experiment 1: Ein Gate – ausrechnen oder laufen lassen?")
    st.caption(
        "Gate aus [markov-queue-demo](" + TX.GITHUB + "/markov-queue-demo): Zustand = Zahl der Lkw im System (0 bis K), K = Plätze **insgesamt, Spuren eingeschlossen**, "
        "Angebot a = Last · c in Erlang. Ein Schritt der Kette ist 1/Λ Abfertigungsdauern, Λ = größte Austrittsrate (a + c, solange K > c)."
    )
    g1, g2, g3 = st.columns(3)
    with g1:
        _slider("Spuren c", "gate_c")
        _slider("Plätze insgesamt K", "gate_k", help="Spuren eingeschlossen: K = 6 heißt 6 Plätze insgesamt, nicht 6 Warteplätze.")
    with g2:
        _slider("Last in %", "gate_last", step=C.GATE_LAST_STEP, fmt="%d %%", help="Angebot a = Last mal Zahl der Spuren (Abfertigungsdauer 1).")
        _slider("Gewünschte Genauigkeit ε im Verlust (Prozentpunkte)", "gate_eps", step=C.GATE_EPS_STEP, fmt="%.1f")
    with g3:
        _slider("Schritte je Kette T", "gate_T", step=C.GATE_T_STEP)
        _slider("Zahl der Ketten R", "gate_R", help="R mal T darf höchstens 1 000 000 Schritte sein (live gerechnet in reiner Python-Schleife).")

# --- Experiment 2 ----------------------------------------------------------------------------------------------------------------------------
with box2:
    st.subheader("🔬 Experiment 2: Posterior einer Ankunftsrate – Gamma–Poisson")
    st.caption("Ankünfte in der Dauer T sind Poisson(λ·T), Prior Gamma(α, β) (Form α, Rate β). Der Posterior ist ausgerechnet Gamma(α + n, β + T); die Kette ist ein Messobjekt.")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        _slider("Prior-Form α", "gp_alpha", help="Ganz: nur dann ist die direkte Ziehung als Summe von Exponentialzahlen möglich.")
        _slider("Prior-Rate β", "gp_beta", step=C.GP_BETA_STEP, fmt="%.1f")
    with p2:
        _slider("Beobachtete Ankünfte n", "gp_n")
        _slider("Beobachtungsdauer T (Abfertigungsdauern)", "gp_T", step=C.GP_T_STEP, fmt="%.0f")
    with p3:
        _slider("Schrittweite der Kette (sd)", "gp_step", step=C.GP_STEP_STEP, fmt="%.1f", help="Vielfache der Standardabweichung des Posteriors.")
        _slider("Schritte N je Kette", "gp_N", step=C.GP_NSTEPS_STEP)
    with p4:
        _slider("Wiederholungen (Ketten) für die Messung", "gp_R", help="Aus R Ketten wird die ESS je Schritt gemessen; weniger Ketten, mehr Rauschen.")
        st.checkbox("Daten in zwei Teilen nacheinander verrechnen", key="gp_split", help="Zeigt, dass die Reihenfolge der Teildaten nichts ändert.")

# --- Experiment 3 ----------------------------------------------------------------------------------------------------------------------------
with box3:
    st.subheader("🔬 Experiment 3: Anderer Prior – ein Gitter genügt")
    st.caption(
        f"Feste ausgedachte Daten: {C.LN_N_OBS} Ankünfte in {C.LN_T_OBS:.0f} Abfertigungsdauern. Der Prior für ln λ ist gauß (Lognormal-Prior für λ) oder der Gamma(2, 1)-Prior des Experiments 2; "
        "gerechnet wird mit der logarithmierten Rate u = ln λ."
    )
    l1, l2, l3 = st.columns(3)
    with l1:
        prior_val = st.radio("Prior", C.PRIOR_CHOICES, key="ln_prior")
        _slider("Median des Lognormal-Priors", "ln_median", step=C.LN_MEDIAN_STEP, fmt="%.1f", disabled=prior_val != C.PRIOR_LOGNORMAL,
                             help="Wirkt nur beim Lognormal-Prior.")
    with l2:
        _slider("Streuung s₀ von ln λ im Prior", "ln_s0", step=C.LN_S0_STEP, fmt="%.2f", disabled=prior_val != C.PRIOR_LOGNORMAL, help="Wirkt nur beim Lognormal-Prior.")
        _slider("Gitterpunkte G", "ln_G", help="Jeder Punkt ist eine Dichteauswertung.")
    with l3:
        _slider("Schritte N je Kette", "ln_N", step=C.LN_NSTEPS_STEP)
        _slider("Wiederholungen (Ketten) für die Messung", "ln_R")

# --- Experiment 4 ----------------------------------------------------------------------------------------------------------------------------
with box4:
    st.subheader("🔬 Experiment 4: Wo die Kette gebraucht wird – mehrere gekoppelte Raten")
    st.caption(
        "d Schichten mit je einer Rate, über einen korrelierten Gauß-Prior auf ln λ gekoppelt (ausgedachte Zählungen 6 bis 11 in der Dauer 5). Der Posterior ist nicht konjugat. "
        "Die Ketten-Messungen (je 240 Ketten zu 64 000 Schritten) sind **vorgerechnet**; das Tensorgitter rechnet die App live."
    )
    d1, d2, d3 = st.columns(3)
    with d1:
        _slider("Zahl gekoppelter Raten d", "dim_d")
    with d2:
        _slider("Gitterpunkte G je Achse", "dim_G", help="Das Gitter hat G^d Punkte (höchstens 3 Millionen).")
    with d3:
        st.radio("Gewünschte Genauigkeit", C.DIM_ACC_CHOICES, key="dim_acc", help="Der Fehler des Mittels in Einheiten der Posterior-Standardabweichung.")

state = {name: st.session_state[sk] for name, sk in PRESET_KEYS.items()}
settings = ev.Settings(**state)
sync_query_params({sk: st.session_state[sk] for sk in PRESET_KEYS.values()})

gate = _gate(settings)
gp = _gamma(settings)
grid = _grid(settings)
dim = _dim(settings)
ex = gate.exact

# --- Ergebnis --------------------------------------------------------------------------------------------------------------------------------
with top:
    st.markdown("## 🎯 Ergebnis: ausgerechnet gegen abgetastet")
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Gate: Verlust (exakt)", F.prozent(ex.loss, 2), delta=f"{ex.equations} Gleichungen", delta_color="off")
    m2.metric("Rate: Posterior-Mittel (exakt)", F.dez(gp.post.mean, 3), delta=f"Gamma({F.zahl(gp.post.shape)}, {F.zahl(gp.post.rate)}): 2 Zahlen", delta_color="off")
    m3.metric("Anderer Prior: Gitterfehler", F.potenz(max(grid.err)) + " sd", delta=f"G = {settings.grid_g} Auswertungen", delta_color="off")
    cross = dim.crossing["iso_a"]
    m4.metric("Kette lohnt ab", f"d = {cross}" if cross else "d > 12", delta=f"{settings.dim_acc}, vorgerechnet", delta_color="off")
    st.plotly_chart(build_overview(gate, gp, grid, dim, settings), width="stretch", key="overview_chart")
    st.caption(
        "Kosten, die für die jeweilige Genauigkeit nötig sind (logarithmische Achse; Einheit je Zeile in Klammern): ausgerechnet (Gleichungen, Zahlen, Dichteauswertungen) gegen abgetastet (Schritte). "
        "Genauigkeit: ±ε im Verlust beim Gate, 0,01 sd bei den Posteriors (Kette: aus der gemessenen ESS je Schritt hochgerechnet, Wurzelgesetz). Gleichungen und Schritte sind verschiedene Operationen: der Vergleich im Gate sagt nur etwas über die Größenordnung; gleich gezählt sind Gitterpunkte und Kettenschritte."
    )

with kernfragen:
    st.markdown("## ❓ Kernfragen")
    for question, kind, text in ev.core_questions(gate, gp, grid, dim, settings):
        st.markdown(f"**{question}**")
        getattr(st, kind)(text)

# --- Experiment 1: Anzeige ---------------------------------------------------------------------------------------------------------------------
with box1:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_gate_pi(ex.pi), width="stretch", key="gate_pi_chart")
    with right:
        if gate.shares is None:
            st.warning(
                f"R · T = {F.ganz(settings.r_chains * settings.t_steps)} Schritte liegen über der Obergrenze von {F.ganz(C.SIM_CAP)}; die Simulation wird nicht gerechnet "
                "(Größenordnung 0,3 Mikrosekunden je Schritt oder darunter, rechnerabhängig; Zeit ist kein Befund). Die exakten Werte bleiben unverändert."
            )
        else:
            st.plotly_chart(build_gate_shares(gate.shares, ex.loss, gate.band, gate.eps, settings.t_steps), width="stretch", key="gate_shares_chart")
    steps_h = 1.0 / ex.lam
    rows = [
        ("Verlust π_K", F.prozent(ex.loss, 2), "exakt aus der Gleichgewichtsverteilung"),
        ("mittlere Zahl im System", F.dez(gate.metrics.mean_n, 2), "Lkw"),
        ("Wartezeit der Angenommenen", F.dez(gate.metrics.mean_wait * C.MEAN_SERVICE_MIN, 2), "Minuten bei 3 min mittlerer Abfertigung (Little: Lq / (a (1 − π_K)))"),
        ("Auslastung der Spuren", F.prozent(gate.metrics.utilization, 1), "E[min(n, c)] / c"),
        ("Mischzeit t_mix(1/4)", f"{ex.t_mix} Schritte" if ex.t_mix is not None else f"über {CH.T_MIX_MAX}",
         f"= {F.dez((ex.t_mix or 0) * steps_h, 2)} Abfertigungsdauern (ein Schritt = {F.dez(steps_h, 3)}); schlechtester Start"),
        ("Relaxationszeit t_rel", F.dez(ex.t_rel, 2) + " Schritte", "1/(1 − größter Betrag der übrigen Eigenwerte)"),
        ("ESS je Schritt: Verlustanzeige", F.dez(ex.ess_loss, 3), "Var(f)/σ² für f = 1{n = K}"),
        ("ESS je Schritt: Zahl im System", F.dez(ex.ess_n, 3), "für f = n: kleiner, denn die Größe ist langsamer"),
        (f"Schritte für ±{F.dez(settings.eps_pp, 1)} Prozentpunkte (95 %)", F.ganz(gate.t_need), "T = 1,96² σ² / ε² (exakt aus der Matrix)"),
    ]
    if gate.shares is not None:
        rows += [
            ("simulierter Verlust (Mittel der Ketten)", F.prozent(float(gate.shares.mean()), 2), f"{settings.r_chains} Ketten × {F.ganz(settings.t_steps)} Schritte, Start leer, ohne Einlauf"),
            ("Band ±1,96·√(σ²/T) (exakt)", F.dez(gate.band * 100, 3) + " Prozentpunkte", f"Überdeckung von ±ε in dieser Messung: {F.prozent(gate.coverage, 0)} von {settings.r_chains} Ketten"),
        ]
    if gate.erlang_b is not None:
        rows.append(("Erlang B (K = c)", F.prozent(gate.erlang_b, 2), "ohne Warteplatz ist der Verlust die Erlang-B-Formel: gleich π_K"))
    st.dataframe(pd.DataFrame(rows, columns=["Größe", "Wert", "Bemerkung"]), hide_index=True, width="stretch")
    kind, text = ev.verdict_gate(ex.loss, ex.equations, ex.pi_deviation, ex.t_mix, ex.ess_loss, settings.eps_pp, gate.t_need, ev.frozen_coverage(settings.c, settings.k, settings.last, settings.eps_pp))
    getattr(st, kind)(text)
    st.caption(
        "**Die Falle:** eine Kette bauen, bevor man geprüft hat, ob sich das Ziel ausrechnen lässt. Abtasten kostet 100-mal mehr Schritte für 10-mal mehr Genauigkeit (Wurzelgesetz); das Ausrechnen nicht. "
        "Die Mischzeit ist nicht das Problem, die Genauigkeit ist."
    )
    with st.expander("Vier Wege zu π (Gegenprobe)"):
        ways = CH.pi_four_ways(settings.c, settings.k, ex.a)
        st.dataframe(pd.DataFrame([[name, *[F.dez(v, 5) for v in vec]] for name, vec in ways.items()], columns=["Weg", *[f"π_{i}" for i in range(settings.k + 1)]]),
                     hide_index=True, width="stretch")
        st.caption(f"Geschlossene Form, lineares System (eine Gleichung durch die Normierung ersetzt), Eigenvektor zum Eigenwert 1 und Matrixpotenz P⁵⁰⁰⁰; größte Abweichung {F.potenz(ex.pi_deviation)}.")

# --- Experiment 2: Anzeige ---------------------------------------------------------------------------------------------------------------------
with box2:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_gamma_density(gp.post, gp.draws, gp.trace), width="stretch", key="gamma_density_chart")
    with right:
        st.plotly_chart(build_ess_steps(gp.table, settings.gp_step, gp.ess), width="stretch", key="ess_chart")
    lo, hi = gp.post.interval()
    q_draw = np.quantile(gp.draws, [0.025, 0.975])
    q_chain = np.quantile(gp.trace, [0.025, 0.975])
    tab = [
        ("exakt: Gamma(α + n, β + T)", F.dez(gp.post.mean, 4), F.dez(gp.post.sd, 4), f"({F.dez(lo, 3)}; {F.dez(hi, 3)})", "0", "2 Zahlen"),
        (f"direkte Ziehungen ({F.ganz(len(gp.draws))})", F.dez(gp.draws_mean, 4), F.dez(gp.draws_sd, 4), f"({F.dez(q_draw[0], 3)}; {F.dez(q_draw[1], 3)})", "10 000", "Ziehungen (ESS je Ziehung = 1)"),
        (f"Kette (eine Kette, {F.ganz(settings.gp_n_steps)} Schritte)", F.dez(float(gp.trace.mean()), 4), F.dez(float(gp.trace.std(ddof=1)), 4), f"({F.dez(q_chain[0], 3)}; {F.dez(q_chain[1], 3)})",
         F.groesse(gp.chain_steps_001), "Schritte (aus der gemessenen ESS je Schritt)"),
    ]
    st.dataframe(pd.DataFrame(tab, columns=["Weg", "Mittel", "sd", "95-%-Intervall", "Aufwand für 0,01 sd", "Einheit"]), hide_index=True, width="stretch")
    st.caption(
        f"Wahrscheinlichkeit der beobachteten Zählung unter dem Prior (negativ-binomial): {F.potenz(gp.evidence, 3)}. "
        f"Die Kette startet in der exakten Verteilung (direkte Ziehung), kein Einlauf wird verworfen; Annahmerate {F.dez(gp.accept, 2)}."
    )
    if settings.gp_split:
        parts = PO.split_data(settings.n_obs, settings.t_obs)
        a1, b1 = settings.alpha + parts[0][0], settings.beta + parts[0][1]
        st.markdown(
            f"**Reihenfolge der Daten egal:** erst {parts[0][0]} Ankünfte in {F.dez(parts[0][1], 2)} → Gamma({F.zahl(a1)}, {F.zahl(b1)}), dann {parts[1][0]} in {F.dez(parts[1][1], 2)} → "
            f"Gamma({F.zahl(gp.seq_post.shape)}, {F.zahl(gp.seq_post.rate)}). Alles in einem Schritt: Gamma({F.zahl(gp.post.shape)}, {F.zahl(gp.post.rate)}). Gleich."
        )
    kind, text = ev.verdict_gamma(gp.post, settings.alpha, settings.beta, settings.n_obs, settings.t_obs, gp.ks_p, len(gp.draws), settings.gp_step, gp.accept, gp.ess, gp.table, gp.chain_steps_001,
                                ev.frozen_gamma_table(settings.alpha, settings.beta, settings.n_obs, settings.t_obs))
    getattr(st, kind)(text)
    st.caption(
        "Ehrlicher Negativbefund: die Kette ist hier in den gemessenen Schrittweiten (0,5 bis 12 sd) nie besser als die direkte Ziehung (ESS je Schritt in der vorgerechneten Messung höchstens 0,230; die direkte Ziehung hat 1). "
        "Das gilt für diesen Posterior und den Zufallsschritt-Metropolis mit festem Schritt, nicht für jede Kette. Die ESS je Schritt ist hier aus R Ketten geschätzt und streut (bei 60 Ketten um etwa ein Fünftel bis ein Viertel); die vorgerechneten Werte mit 400 Ketten sind genauer."
    )

# --- Experiment 3: Anzeige ---------------------------------------------------------------------------------------------------------------------
with box3:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_grid_density(grid), width="stretch", key="grid_density_chart")
    with right:
        st.plotly_chart(build_grid_error(grid.errs_by_g, settings.grid_g, max(grid.err)), width="stretch", key="grid_error_chart")
    tab3 = [
        ("E[u] (Mittel von ln λ)", F.dez(grid.ref.mean_u, 5), F.dez(grid.grid.mean_u, 5), F.potenz(grid.err[0]) + " sd"),
        ("sd[u]", F.dez(grid.ref.sd_u, 5), F.dez(grid.grid.sd_u, 5), F.potenz(grid.err[1]) + " (relativ)"),
        ("E[λ] (Mittel der Rate)", F.dez(grid.ref.mean_lambda, 5), F.dez(grid.grid.mean_lambda, 5), F.potenz(grid.err[2]) + " sd"),
    ]
    st.dataframe(pd.DataFrame(tab3, columns=["Größe", "Referenz (feines Gitter, 4001 Punkte)", f"Gitter mit G = {settings.grid_g}", "Fehler"]), hide_index=True, width="stretch")
    kind, text = ev.verdict_grid("Lognormal-Prior (Median " + F.dez(settings.median, 1) + ", s₀ = " + F.dez(settings.s0, 2) + ")" if prior_val == C.PRIOR_LOGNORMAL else "Gamma(2, 1)-Prior",
                                prior_val == C.PRIOR_LOGNORMAL, settings.grid_g, grid.err, grid.g_need, grid.ref.mean_lambda, grid.conj_mean, grid.chain_steps_001)
    getattr(st, kind)(text)
    st.caption(
        f"Kette: Zufallsschritt-Metropolis in u, Schritt {C.LN_CHAIN_STEP_SD} sd, Start in der exakten Verteilung, ESS je Schritt etwa {F.dez(grid.ess, 3)} (aus {settings.ln_reps} Ketten), "
        f"Annahmerate {F.dez(grid.accept, 2)}. **Das Gitter setzt voraus, dass man den Gipfel kennt** (der Kasten liegt um den Gipfel); bei mehreren Gipfeln versagt dieses Vorgehen. "
        "Gemessen ist nur ein Gitter mit bekannter Lage der Masse."
    )

# --- Experiment 4: Anzeige ---------------------------------------------------------------------------------------------------------------------
with box4:
    left, right = st.columns(2)
    with left:
        st.plotly_chart(build_dim_cost(dim.rows, settings.dim_acc), width="stretch", key="dim_cost_chart")
    with right:
        st.plotly_chart(build_dim_rmse(settings.dim_d), width="stretch", key="dim_rmse_chart")
    st.markdown("**Vorgerechnete Tabelle** (je Zeile 240 Ketten zu 64 000 Schritten; Gitter: G Punkte je Achse, Fehler gegen das feine Tensorgitter)")
    frozen = REF.dimension()["d"]
    tab4 = []
    for r_ in dim.rows:
        dd = r_[0]
        ch = frozen[str(dd)]["ketten"]
        tab4.append((dd, F.groesse(r_[1]), F.groesse(r_[2]), F.groesse(r_[3]), F.groesse(r_[4]), F.dez(ch["iso_a"]["ess_per_step"], 3), F.dez(ch["iso_a"]["ess_per_step"] * dd, 3),
                     F.dez(ch["iso_a"]["accept"], 3), F.potenz(max(frozen[str(dd)]["gitter_fehler"]["10"]))))
    st.dataframe(pd.DataFrame(tab4, columns=["d", f"Gitterpunkte {dim.grid_needed}^d", "Kettenschritte (Folge A)", "Folge B", "vorkonditioniert", "ESS je Schritt (A)", "ESS je Schritt × d",
                                             "Annahmerate (A)", "Gitterfehler G = 10 (sd)"]), hide_index=True, width="stretch")
    kind, text = ev.verdict_dimension(settings.dim_d, settings.dim_g, dim.live_err, dim.rows, dim.crossing, settings.dim_acc, dim.grid_needed)
    getattr(st, kind)(text)
    st.caption(
        "**Falle und Negativbefund:** nicht die Nichtkonjugiertheit macht die Kette nötig, sondern die Zahl der Dimensionen. Das gilt für diese Zählungen (Poisson, unabhängige Ankünfte), diesen Prior, "
        "ein Gitter um den Gipfel mit G = 10 je Achse und Zufallsschritt-Metropolis mit festem Schritt; Zähleinheit Dichteauswertungen bzw. Schritte. Die Zeile für 10⁻⁴ sd ist hochgerechnet (ESS je Schritt ≈ Mittel(ESS · d)/d "
        "für d > 6, G = 12), nicht gemessen. Die Zeile d = 1 gehört zu einem anderen Modell (n = 8 in T = 5) als der Aufwand in Experiment 3 (n = 32 in T = 20); die beiden Kettenaufwände sind nicht dieselbe Messung."
    )

# --- Tabelle: auch die Vehikel brauchen keine Kette ---------------------------------------------------------------------------------------------
with st.expander("🧰 Was hilft das Ausrechnen? Auch die Beispiele der Nachbar-Demos brauchen keine Kette"):
    ring = _ring()
    st.markdown("Drei Beispiele, ausgerechnet statt abgetastet (Zähleinheiten: Gleichungen, Zahlen, Zeilenzustände):")
    tabv = [
        ("Gate (Experiment 1)", "π, Verlust, Mischzeit", f"{ex.equations} Gleichungen", "Linear lösen, Gleichgewicht geschlossen"),
        ("Gamma–Poisson (Experiment 2)", "Posterior der Rate", "2 Zahlen", "Gamma(α + n, β + T)"),
        ("Belegung eines Rings aus n Plätzen (keine zwei benachbarten belegt, Gewicht λ^k)", "Z(λ), Mittelwerte", "2 × 2-Übertragungsmatrix", "Z(λ) = μ₊ⁿ + μ₋ⁿ, μ± = (1 ± √(1 + 4λ))/2"),
    ]
    st.dataframe(pd.DataFrame(tabv, columns=["Beispiel", "Ausgerechnet", "Aufwand", "Wie"]), hide_index=True, width="stretch")
    st.dataframe(pd.DataFrame([(n, F.ganz(states), F.ganz(zt), F.ganz(ze)) for n, states, zt, ze in ring],
                              columns=["Plätze n im Ring (λ = 3)", "Belegungen (Aufzählung)", "Z(λ): Spur(Tⁿ)", "Z(λ): Aufzählung aller 2ⁿ Muster"]), hide_index=True, width="stretch")
    st.markdown(
        "**Der Ring ist exakt lösbar und dient den Nachbar-Demos als Orakel, nicht als Fall für einen Sampler; das gilt für Mittelwerte und Z(λ), nicht für die Mischzeit.** "
        f"Die Mischzeit des Rings: [mischzeit-exakt-demo]({TX.GITHUB}/mischzeit-exakt-demo). Torus und Direktsampler: [sampler-verfahrenswahl-demo]({TX.GITHUB}/sampler-verfahrenswahl-demo)."
    )

# --- Grenzen -----------------------------------------------------------------------------------------------------------------------------------
st.subheader("🚧 Wo die Annahmen enden")
st.markdown(TX.BOUNDARIES_MARKDOWN)
st.markdown(
    f"Verwandte Demos: [monte-carlo-schaetzer-demo]({TX.GITHUB}/monte-carlo-schaetzer-demo) (Fehler ∝ 1/√N für unabhängige Ziehungen), "
    f"[output-analysis-demo]({TX.GITHUB}/output-analysis-demo) (Batch Means, Warm-up), [markov-queue-demo]({TX.GITHUB}/markov-queue-demo) (das Gate)."
)

st.markdown("---")

with st.expander("📐 Mathematische Formulierung"):
    # erklaerseite:lesehilfe:start (erzeugt von tools/set_demo_explainer.py (Website-Repo) aus tools/content/explainers/exakt-oder-abtasten-erklaert.json, nicht von Hand ändern)
    st.markdown(
        r"""
**Lesehilfe.** Gegeben ist jeweils ein Ziel (ein Verlustanteil, ein Posterior), gesucht ein Mittelwert oder Anteil auf eine gewünschte Genauigkeit. Der Block ist ein **vollständiges Modell** der vier Rechnungen; was er weglässt, steht darunter.

| Symbol | Bedeutung |
|---|---|
| $c$, $K$, $a$ | Spuren, Plätze **insgesamt** (Spuren eingeschlossen), Angebot in Erlang (mittlere Abfertigungsdauer 1) |
| $\pi$, $P$, $\Lambda$ | stationäre Verteilung, uniformisierte Übergangsmatrix, Rate $\Lambda$ (größte Austrittsrate) |
| $d(t)$, $t_{\mathrm{mix}}$ | größter Abstand zu $\pi$ nach $t$ Schritten, Mischzeit $t_{\mathrm{mix}}(1/4)$ |
| $\sigma^2$, ESS je Schritt | asymptotische Varianz des Zeitmittels einer Anzeige $f$, $\mathrm{Var}_\pi(f)/\sigma^2$ |
| $\alpha$, $\beta$, $n$, $T$ | Prior-Form, Prior-Rate, beobachtete Ankünfte, Beobachtungsdauer (in Abfertigungsdauern) |
| $u = \ln \lambda$, $G$, $d$ | logarithmierte Rate, Gitterpunkte je Achse, Zahl der gekoppelten Raten |
| sd | Standardabweichung des Posteriors (Einheit der „Genauigkeit“) |

[Schritt für Schritt erklärt, mit Herleitung und Literatur](https://sebastianhanisch.net/exakt-oder-abtasten-erklaert.html#fachleute)
        """
    )
    # erklaerseite:lesehilfe:end
    st.markdown(TX.FORMULAS_MARKDOWN)

st.markdown("---")
st.caption(
    "Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – "
    "Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). "
    "Mehr zur Reihe: [MCMC: von der Kette zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-mcmc.html)."
)
