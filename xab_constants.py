"""Konstanten: Regler-Grenzen, feste Annahmen, Farben und Presets (Zufalls-Seed heißt in der Oberfläche „Zufalls-Seed“)."""

# --- Experiment 1: Gate-Kette (2 Spuren, K = 6 Plätze INSGESAMT, Last 80 % ist die Voreinstellung) ---------------------------------------------
GATE_C_MIN, GATE_C_MAX = 1, 4
GATE_K_MIN, GATE_K_MAX = 4, 10                  # K ≥ c: Plätze insgesamt, Spuren eingeschlossen
GATE_LAST_MIN, GATE_LAST_MAX, GATE_LAST_STEP = 50, 150, 5     # Last in % (Angebot a = Last · c)
GATE_T_MIN, GATE_T_MAX, GATE_T_STEP = 1000, 200000, 1000       # Schritte je Kette
GATE_R_MIN, GATE_R_MAX = 1, 400                 # Zahl der Ketten
GATE_EPS_MIN, GATE_EPS_MAX, GATE_EPS_STEP = 0.2, 2.0, 0.1      # gewünschte Genauigkeit im Verlust, Prozentpunkte
SIM_CAP = 1_000_000                             # höchstens R·T Schritte live (reine Python-Schleife, etwa 0,3 µs je Schritt, gemessen: Zähleinheit Schritte)
MEAN_SERVICE_MIN = 3.0                          # mittlere Abfertigungsdauer in Minuten (wie die Nachbar-Demo markov-queue-demo)

# --- Experiment 2: Gamma–Poisson ---------------------------------------------------------------------------------------------------------------
GP_ALPHA_MIN, GP_ALPHA_MAX = 1, 10
GP_BETA_MIN, GP_BETA_MAX, GP_BETA_STEP = 0.1, 5.0, 0.1
GP_N_MIN, GP_N_MAX = 0, 100                     # beobachtete Ankünfte
GP_T_MIN, GP_T_MAX, GP_T_STEP = 1.0, 100.0, 1.0 # Beobachtungsdauer in Abfertigungsdauern
GP_STEP_MIN, GP_STEP_MAX, GP_STEP_STEP = 0.5, 12.0, 0.1        # Schrittweite der Kette in Standardabweichungen des Posteriors
GP_NSTEPS_MIN, GP_NSTEPS_MAX, GP_NSTEPS_STEP = 500, 10000, 500
GP_REPS_MIN, GP_REPS_MAX = 10, 200
DIRECT_DRAWS = 2000                             # direkte Ziehungen live (ein exact_log je Ziehung, gemessen etwa 0,3 ms)
STEP_GRID_SD = (0.5, 1.0, 2.4, 5.0, 12.0)       # Schrittweiten der Vergleichstabelle (Vielfache der Posterior-sd)
STEP_GRID_N, STEP_GRID_R = 1000, 60             # Aufwand der Vergleichstabelle live

# --- Experiment 3: nichtkonjugater Prior -------------------------------------------------------------------------------------------------------
LN_N_OBS, LN_T_OBS = 32, 20.0                   # feste ausgedachte Daten: 32 Ankünfte in 20 Abfertigungsdauern
PRIOR_GAMMA = "Gamma(2, 1) (konjugat)"
PRIOR_LOGNORMAL = "Lognormal (Median, s₀)"
PRIOR_CHOICES = (PRIOR_GAMMA, PRIOR_LOGNORMAL)
GAMMA_PRIOR = (2.0, 1.0)
LN_MEDIAN_MIN, LN_MEDIAN_MAX, LN_MEDIAN_STEP = 0.5, 5.0, 0.1
LN_S0_MIN, LN_S0_MAX, LN_S0_STEP = 0.2, 1.5, 0.05
LN_G_MIN, LN_G_MAX = 4, 200                     # Gitterpunkte
LN_G_LIST = (6, 8, 10, 12)                      # Gitter der Fehlerkurve
LN_NSTEPS_MIN, LN_NSTEPS_MAX, LN_NSTEPS_STEP = 500, 10000, 500
LN_REPS_MIN, LN_REPS_MAX = 10, 200
LN_CHAIN_STEP_SD = 2.4
COARSE_GRID = 6                                 # ab G < 6 meldet die App „Gitter zu grob“ (Messung: Fehler bis 1,3 sd)

# --- Experiment 4: mehrere gekoppelte Raten -----------------------------------------------------------------------------------------------------
DIM_D_MIN, DIM_D_MAX = 1, 6
DIM_G_MIN, DIM_G_MAX = 4, 12
DIM_POINT_CAP = 3_000_000                       # höchstens G^d Punkte live (12^6 = 2 985 984)
DIM_ACC_001 = "0,01 sd"
DIM_ACC_1E4 = "10⁻⁴ sd (hochgerechnet)"
DIM_ACC_CHOICES = (DIM_ACC_001, DIM_ACC_1E4)
DIM_ACC_VALUE = {DIM_ACC_001: 0.01, DIM_ACC_1E4: 1e-4}
DIM_GRID_FOR_ACC = {DIM_ACC_001: 10, DIM_ACC_1E4: 12}           # kleinstes gemessenes G, das die Genauigkeit erreicht (Tabelle, Test)

SEED_MAX, DEFAULT_SEED = 999999, 35

# --- Farben --------------------------------------------------------------------------------------------------------------------------------------
COLOR_EXACT = "#2F6B65"
COLOR_SIM = "#d98b2b"
COLOR_GRID = "#4c78a8"
COLOR_CHAIN = "#b5543c"
COLOR_OTHER = "rgba(150,150,150,0.55)"

# --- Voreinstellung und Presets -----------------------------------------------------------------------------------------------------------------
DEFAULTS = dict(
    seed=DEFAULT_SEED,
    c=2, k=6, last=80, t_steps=20000, r_chains=10, eps_pp=0.5,
    alpha=2, beta=1.0, n_obs=32, t_obs=20.0, gp_step=2.4, gp_n_steps=2000, gp_reps=60, gp_split=False,
    prior=PRIOR_GAMMA, median=2.0, s0=0.5, grid_g=10, ln_n_steps=2000, ln_reps=60,
    dim_d=3, dim_g=10, dim_acc=DIM_ACC_001,
)


def _preset(**over) -> dict:
    p = dict(DEFAULTS)
    p.update(over)
    return p


PRESETS = {
    "Gate: Voreinstellung (2 Spuren, 6 Plätze)": _preset(),
    "Gate: Überlast": _preset(last=150),
    "Gate: vier Spuren ohne Warteplatz": _preset(c=4, k=4),
    "Gamma–Poisson: 32 in 20": _preset(gp_n_steps=5000, gp_reps=100),
    "Wenig Daten: 3 in 2 (Gamma(5, 3))": _preset(n_obs=3, t_obs=2.0),
    "Lognormal-Prior": _preset(prior=PRIOR_LOGNORMAL),
    "Sechs Schichten (d = 6)": _preset(dim_d=6),
}
PRESET_HELP = {
    "Gate: Voreinstellung (2 Spuren, 6 Plätze)": "Zwei Spuren, sechs Plätze insgesamt (Spuren eingeschlossen), Last 80 %: sieben Zustände. Die Kette läuft 10 Ketten zu je 20 000 Schritten.",
    "Gate: Überlast": "Wie die Voreinstellung, aber Last 150 %: der Verlust steigt auf über ein Drittel, die Kette braucht rund 204 000 Schritte für ±0,5 Prozentpunkte.",
    "Gate: vier Spuren ohne Warteplatz": "Vier Spuren, nur vier Plätze: kein Warteplatz, der Verlust ist die Erlang-B-Formel.",
    "Gamma–Poisson: 32 in 20": "Prior Gamma(2, 1), 32 Ankünfte in 20 Abfertigungsdauern: Posterior Gamma(34, 21). Die Kette läuft 100 Wiederholungen zu je 5 000 Schritten (engeres Band).",
    "Wenig Daten: 3 in 2 (Gamma(5, 3))": "Prior Gamma(2, 1), nur 3 Ankünfte in 2 Abfertigungsdauern: Posterior Gamma(5, 3), das Handbeispiel.",
    "Lognormal-Prior": "Prior für ln λ gauß (Median 2, s₀ = 0,5): der Posterior ist kein Gamma mehr, ein Gitter genügt trotzdem.",
    "Sechs Schichten (d = 6)": "Sechs gekoppelte Raten: das Gitter hat 10⁶ Punkte, die Kette gewinnt erstmals.",
}
