"""Layout (DEMO-PLAYBOOK §2): fester Kopf, Schnellstart oben in der Seitenleiste, aktives Preset hervorgehoben, Kernfragen vor dem Formel-Expander.

Die Hervorhebung ist zustandslos (`aktives_preset` vergleicht jeden Regler aus SETTING_SPECS mit dem Preset). Die Regler der Experimente stehen im Hauptbereich beim Experiment,
nur der Zufalls-Seed steht in der Seitenleiste; der Test unten stellt sicher, dass wirklich JEDER Regler der Seite in SETTING_SPECS steht (sonst bliebe das Preset trotz Änderung
markiert) und dass jede Änderung die Markierung aufhebt."""

from pathlib import Path

import pytest
from streamlit.testing.v1 import AppTest

import xab_constants as C
import xab_presets as P

APP = str(Path(__file__).resolve().parent.parent / "app.py")
REGLER_ARTEN = ("slider", "number_input", "select_slider", "selectbox", "radio", "toggle", "checkbox", "text_input")
DEFAULT_PRESET = next(iter(C.PRESETS))
LOGNORMAL_ONLY = ("ln_median", "ln_s0")           # im Gamma-Prior-Fall bewusst deaktiviert (nicht änderbar)


def _app() -> AppTest:
    at = AppTest.from_file(APP, default_timeout=300)
    at.run()
    assert not at.exception, [str(e) for e in at.exception]
    return at


def _typen(at) -> dict:
    """Button-Typ je Preset-Name (die Preset-Buttons der Seitenleiste)."""
    return {b.label: b.proto.type for b in at.sidebar.button if b.label in C.PRESETS}


def _regler(at, key):
    for art in REGLER_ARTEN:
        try:
            return getattr(at, art)(key=key)
        except KeyError:
            continue
    raise AssertionError(f"Regler {key} steht nicht auf der Seite")


def test_presets_stehen_in_der_seitenleiste_und_nicht_im_hauptbereich():
    at = _app()
    assert [b.label for b in at.sidebar.button][: len(C.PRESETS)] == list(C.PRESETS)  # die ersten Buttons der Seitenleiste
    assert not [b for b in at.main.button if b.label in C.PRESETS]


def test_titel_steht_im_festen_kopf():
    at = _app()
    assert "st-key-kopf" in " ".join(h.proto.body for h in at.get("html"))  # das Stylesheet setzt an der Klasse des Containers an
    assert at.title[0].value


def test_experimentregler_stehen_im_hauptbereich_und_nur_der_zufalls_seed_in_der_seitenleiste():
    at = _app()
    side = {w.key for art in REGLER_ARTEN for w in getattr(at.sidebar, art) if w.key}
    assert side == {"seed_input"}
    main = {w.key for art in REGLER_ARTEN for w in getattr(at.main, art) if w.key}
    assert main == set(P.SETTING_SPECS) - {"seed_input"}


def test_kernfragen_stehen_vor_den_experimenten_und_dem_formel_expander():
    at = _app()
    heads = [m.value for m in at.markdown if m.value.startswith("## ")]
    assert any("Kernfragen" in h for h in heads) and any("Ergebnis" in h for h in heads)
    subs = [s.value for s in at.subheader]
    assert [s.split(":")[0] for s in subs if s.startswith("🔬")] == [f"🔬 Experiment {i}" for i in (1, 2, 3, 4)]
    assert subs[-1].startswith("🚧")
    labels = [e.label for e in at.expander]
    assert labels.index(next(x for x in labels if x.startswith("📐"))) == len(labels) - 1          # der Formel-Expander ist der letzte


def test_beim_start_ist_genau_das_standard_preset_hervorgehoben():
    typen = _typen(_app())
    assert [name for name, typ in typen.items() if typ == "primary"] == [DEFAULT_PRESET]


@pytest.mark.parametrize("name", list(C.PRESETS))
def test_angeklicktes_preset_ist_hervorgehoben_und_bleibt_es_im_naechsten_lauf(name):
    at = _app()
    at.sidebar.button(key=f"preset_{name}").click().run()
    assert not at.exception
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]
    at.run()
    assert [n for n, typ in _typen(at).items() if typ == "primary"] == [name]


def test_jeder_regler_der_seite_steht_in_setting_specs():
    at = _app()
    schluessel = {w.key for art in REGLER_ARTEN for w in getattr(at, art) if w.key}
    fehlt = schluessel - set(P.SETTING_SPECS)
    assert not fehlt, f"Regler {sorted(fehlt)} fehlen in SETTING_SPECS: ihre Änderung hebt das aktive Preset nicht auf"
    assert set(P.SETTING_SPECS) <= schluessel


def _aenderung(state_key):
    """(Preset, neuer Wert): ein Preset als Start und ein neuer Wert des Reglers, der NICHT zu einem anderen Preset führt (sonst wäre dieses zu Recht markiert)."""
    spec = P.SETTING_SPECS[state_key]
    preset_key = next(k for k, sk in P.PRESET_KEYS.items() if sk == state_key)
    for name, preset in C.PRESETS.items():
        if state_key in LOGNORMAL_ONLY and preset["prior"] != C.PRIOR_LOGNORMAL:
            continue                                                  # nur beim Lognormal-Prior änderbar
        if spec.choices is not None:
            kandidaten = list(spec.choices)
        elif isinstance(spec.default, bool):
            kandidaten = [not preset[preset_key]]
        else:
            kandidaten = [spec.hi, spec.lo]
        for kandidat in kandidaten:
            if kandidat == preset[preset_key]:
                continue
            neu = dict(preset)
            neu[preset_key] = kandidat
            if neu not in C.PRESETS.values():
                return name, kandidat
    raise AssertionError(f"keine Änderung von {state_key} führt aus allen Presets heraus")


@pytest.mark.parametrize("state_key", list(P.SETTING_SPECS))
def test_jede_aenderung_eines_reglers_hebt_die_hervorhebung_auf(state_key):
    at = _app()
    start, wert = _aenderung(state_key)
    at.sidebar.button(key=f"preset_{start}").click().run()
    assert _typen(at)[start] == "primary"
    _regler(at, state_key).set_value(wert).run()
    assert not at.exception, [str(e) for e in at.exception]
    assert "primary" not in _typen(at).values()
    # erneutes Anklicken markiert wieder
    at.sidebar.button(key=f"preset_{DEFAULT_PRESET}").click().run()
    assert _typen(at)[DEFAULT_PRESET] == "primary"
