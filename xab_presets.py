"""Permalink (Begrenzen), Presets und Zufalls-Seed-Button (Standardmuster aus dem Demo-Portfolio).

Alle Regler stehen in SETTING_SPECS (auch die im Hauptbereich neben dem jeweiligen Experiment): das aktive Preset ist nur hervorgehoben, solange JEDER Regler
dem Preset entspricht. Settings-Felder (xab_evaluation.Settings) heißen wie die Preset-Schlüssel; PRESET_KEYS bildet sie auf die Schlüssel der Regler ab."""

import random
from dataclasses import dataclass

import streamlit as st

import xab_constants as C


def _to_bool(text) -> bool:
    return str(text).strip().lower() in ("1", "true", "ja", "wahr")


@dataclass(frozen=True)
class SettingSpec:
    url_param: str
    caster: object
    default: object
    lo: float | None = None
    hi: float | None = None
    choices: tuple | None = None


D = C.DEFAULTS
SETTING_SPECS = {
    "seed_input": SettingSpec("seed", int, D["seed"], 0, C.SEED_MAX),
    "gate_c": SettingSpec("c", int, D["c"], C.GATE_C_MIN, C.GATE_C_MAX),
    "gate_k": SettingSpec("k", int, D["k"], C.GATE_K_MIN, C.GATE_K_MAX),
    "gate_last": SettingSpec("last", int, D["last"], C.GATE_LAST_MIN, C.GATE_LAST_MAX),
    "gate_T": SettingSpec("t", int, D["t_steps"], C.GATE_T_MIN, C.GATE_T_MAX),
    "gate_R": SettingSpec("r", int, D["r_chains"], C.GATE_R_MIN, C.GATE_R_MAX),
    "gate_eps": SettingSpec("eps", float, D["eps_pp"], C.GATE_EPS_MIN, C.GATE_EPS_MAX),
    "gp_alpha": SettingSpec("alpha", int, D["alpha"], C.GP_ALPHA_MIN, C.GP_ALPHA_MAX),
    "gp_beta": SettingSpec("beta", float, D["beta"], C.GP_BETA_MIN, C.GP_BETA_MAX),
    "gp_n": SettingSpec("n", int, D["n_obs"], C.GP_N_MIN, C.GP_N_MAX),
    "gp_T": SettingSpec("dauer", float, D["t_obs"], C.GP_T_MIN, C.GP_T_MAX),
    "gp_step": SettingSpec("gstep", float, D["gp_step"], C.GP_STEP_MIN, C.GP_STEP_MAX),
    "gp_N": SettingSpec("gn", int, D["gp_n_steps"], C.GP_NSTEPS_MIN, C.GP_NSTEPS_MAX),
    "gp_R": SettingSpec("gr", int, D["gp_reps"], C.GP_REPS_MIN, C.GP_REPS_MAX),
    "gp_split": SettingSpec("split", _to_bool, D["gp_split"]),
    "ln_prior": SettingSpec("prior", str, D["prior"], choices=C.PRIOR_CHOICES),
    "ln_median": SettingSpec("median", float, D["median"], C.LN_MEDIAN_MIN, C.LN_MEDIAN_MAX),
    "ln_s0": SettingSpec("s0", float, D["s0"], C.LN_S0_MIN, C.LN_S0_MAX),
    "ln_G": SettingSpec("g", int, D["grid_g"], C.LN_G_MIN, C.LN_G_MAX),
    "ln_N": SettingSpec("ln", int, D["ln_n_steps"], C.LN_NSTEPS_MIN, C.LN_NSTEPS_MAX),
    "ln_R": SettingSpec("lr", int, D["ln_reps"], C.LN_REPS_MIN, C.LN_REPS_MAX),
    "dim_d": SettingSpec("d", int, D["dim_d"], C.DIM_D_MIN, C.DIM_D_MAX),
    "dim_G": SettingSpec("dg", int, D["dim_g"], C.DIM_G_MIN, C.DIM_G_MAX),
    "dim_acc": SettingSpec("acc", str, D["dim_acc"], choices=C.DIM_ACC_CHOICES),
}
PRESET_KEYS = {
    "seed": "seed_input", "c": "gate_c", "k": "gate_k", "last": "gate_last", "t_steps": "gate_T", "r_chains": "gate_R", "eps_pp": "gate_eps",
    "alpha": "gp_alpha", "beta": "gp_beta", "n_obs": "gp_n", "t_obs": "gp_T", "gp_step": "gp_step", "gp_n_steps": "gp_N", "gp_reps": "gp_R", "gp_split": "gp_split",
    "prior": "ln_prior", "median": "ln_median", "s0": "ln_s0", "grid_g": "ln_G", "ln_n_steps": "ln_N", "ln_reps": "ln_R",
    "dim_d": "dim_d", "dim_g": "dim_G", "dim_acc": "dim_acc",
}


def init_session_state_defaults():
    for state_key, spec in SETTING_SPECS.items():
        if state_key not in st.session_state:
            st.session_state[state_key] = spec.default


def bounds(state_key):
    spec = SETTING_SPECS[state_key]
    return spec.lo, spec.hi


def clamp_value(spec: SettingSpec, value):
    """Wert auf die Reglergrenzen begrenzen bzw. bei Auswahlreglern auf eine erlaubte Option prüfen (sonst Standardwert)."""
    if spec.choices is not None:
        return value if value in spec.choices else spec.default
    if spec.lo is not None:
        value = max(spec.lo, value)
    if spec.hi is not None:
        value = min(spec.hi, value)
    return value


def load_permalink_settings():
    """Einmal je Sitzung: Werte aus der Adresszeile übernehmen, auf die Reglergrenzen begrenzt (Auswahlregler: nur erlaubte Optionen)."""
    if "permalink_loaded" in st.session_state:
        return
    qp = st.query_params
    for state_key, spec in SETTING_SPECS.items():
        if spec.url_param in qp:
            try:
                value = spec.caster(qp[spec.url_param])
            except (ValueError, TypeError):
                continue
            st.session_state[state_key] = clamp_value(spec, value)
    # K muss mindestens c sein (Plätze insgesamt, Spuren eingeschlossen)
    if st.session_state.get("gate_k", C.GATE_K_MIN) < st.session_state.get("gate_c", C.GATE_C_MIN):
        st.session_state["gate_k"] = st.session_state["gate_c"]
    st.session_state["permalink_loaded"] = True


def sync_query_params(values):
    """`values`: {state_key: aktueller Wert}; die Adresszeile spiegelt die Konfiguration."""
    try:
        for state_key, value in values.items():
            st.query_params[SETTING_SPECS[state_key].url_param] = str(value)
    except Exception:
        pass


def apply_preset(name):
    for key, state_key in PRESET_KEYS.items():
        if key in C.PRESETS[name]:
            st.session_state[state_key] = C.PRESETS[name][key]


def aktives_preset():
    """Name des Presets, dessen Werte GENAU den aktuellen Einstellungen entsprechen, sonst None.

    Zustandslos: verglichen wird jeder Regler aus SETTING_SPECS (ein Preset ohne Eintrag für einen Regler meint dessen
    Standardwert) mit dem Session State. Ein später ergänzter Regler braucht nur seinen Eintrag in SETTING_SPECS,
    dann hebt auch seine Änderung die Markierung auf (tests/test_layout.py prüft das für jeden Regler)."""
    preset_key_of = {state_key: key for key, state_key in PRESET_KEYS.items()}
    for name, preset in C.PRESETS.items():
        if all(
            st.session_state.get(state_key, spec.default) == preset.get(preset_key_of.get(state_key), spec.default)
            for state_key, spec in SETTING_SPECS.items()
        ):
            return name
    return None


def sidebar_presets():
    """Schnellstart oben in der Seitenleiste; das aktive Preset ist hervorgehoben (`type="primary"`)."""
    if not C.PRESETS:
        return
    st.header("🎯 Schnellstart")
    st.caption("Ein Beispielszenario laden:")
    aktiv = aktives_preset()
    for name in C.PRESETS:
        st.button(
            name, width="stretch", on_click=apply_preset, args=(name,), help=C.PRESET_HELP.get(name, ""), key=f"preset_{name}",
            type="primary" if aktiv == name else "secondary",
        )


def randomize_seed():
    st.session_state["seed_input"] = random.randint(0, C.SEED_MAX)
