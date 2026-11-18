"""Presets: Vollständigkeit, gültige Werte, Standard-Preset = Standardeinstellungen, Permalink-Begrenzung."""

import pytest

import xab_constants as C
import xab_evaluation as ev
import xab_presets as P


def test_every_preset_has_help_and_all_keys():
    assert set(C.PRESETS) == set(C.PRESET_HELP)
    for name, preset in C.PRESETS.items():
        assert set(preset) == set(P.PRESET_KEYS), name
        assert C.PRESET_HELP[name]


def test_preset_values_are_inside_the_regler_bounds_or_choices():
    for name, preset in C.PRESETS.items():
        for key, state_key in P.PRESET_KEYS.items():
            spec = P.SETTING_SPECS[state_key]
            value = preset[key]
            if spec.choices is not None:
                assert value in spec.choices, (name, key)
            elif spec.lo is not None:
                assert spec.lo <= value <= spec.hi, (name, key)
            else:
                assert isinstance(value, bool), (name, key)


def test_default_preset_equals_the_default_settings():
    assert ev.Settings(**C.PRESETS["Gate: Voreinstellung (2 Spuren, 6 Plätze)"]) == ev.Settings()
    assert next(iter(C.PRESETS)) == "Gate: Voreinstellung (2 Spuren, 6 Plätze)"


def test_presets_are_pairwise_different():
    seen = set()
    for name, preset in C.PRESETS.items():
        key = tuple(sorted(preset.items()))
        assert key not in seen, name
        seen.add(key)


def test_setting_specs_cover_every_preset_key_and_every_settings_field():
    assert set(P.PRESET_KEYS.values()) == set(P.SETTING_SPECS)
    assert set(P.PRESET_KEYS) == set(ev.Settings.__dataclass_fields__)


def test_defaults_are_inside_their_bounds_and_k_is_at_least_c():
    for state_key, spec in P.SETTING_SPECS.items():
        if spec.lo is not None:
            assert spec.lo <= spec.default <= spec.hi, state_key
    assert C.GATE_K_MIN >= C.GATE_C_MAX                        # K ≥ c in jeder Einstellung (Plätze insgesamt, Spuren eingeschlossen)


def test_permalink_clamps_to_the_bounds_and_rejects_unknown_options():
    spec = P.SETTING_SPECS["gate_c"]
    assert P.clamp_value(spec, 99) == C.GATE_C_MAX and P.clamp_value(spec, -3) == C.GATE_C_MIN and P.clamp_value(spec, 3) == 3
    choice = P.SETTING_SPECS["ln_prior"]
    assert P.clamp_value(choice, "Unsinn") == C.DEFAULTS["prior"] and P.clamp_value(choice, C.PRIOR_LOGNORMAL) == C.PRIOR_LOGNORMAL
    assert P.SETTING_SPECS["gp_split"].caster("True") is True and P.SETTING_SPECS["gp_split"].caster("False") is False


def test_slider_steps_hit_the_default_values():
    """Jeder Standardwert liegt auf dem Raster seines Reglers (sonst zeigt der Regler beim Start einen anderen Wert)."""
    grids = {"gate_last": (C.GATE_LAST_MIN, C.GATE_LAST_STEP), "gate_T": (C.GATE_T_MIN, C.GATE_T_STEP), "gate_eps": (C.GATE_EPS_MIN, C.GATE_EPS_STEP),
             "gp_beta": (C.GP_BETA_MIN, C.GP_BETA_STEP), "gp_T": (C.GP_T_MIN, C.GP_T_STEP), "gp_step": (C.GP_STEP_MIN, C.GP_STEP_STEP),
             "gp_N": (C.GP_NSTEPS_MIN, C.GP_NSTEPS_STEP), "ln_median": (C.LN_MEDIAN_MIN, C.LN_MEDIAN_STEP), "ln_s0": (C.LN_S0_MIN, C.LN_S0_STEP), "ln_N": (C.LN_NSTEPS_MIN, C.LN_NSTEPS_STEP)}
    for state_key, (lo, step) in grids.items():
        for value in [P.SETTING_SPECS[state_key].default] + [p[k] for p in C.PRESETS.values() for k, sk in P.PRESET_KEYS.items() if sk == state_key]:
            assert (value - lo) / step == pytest.approx(round((value - lo) / step), abs=1e-9), (state_key, value)
