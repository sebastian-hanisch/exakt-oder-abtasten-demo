"""End-to-end Smoke-Test via Streamlits offiziellem AppTest-Framework: lädt app.py mit den Standardeinstellungen, klickt jeden Preset-Button, fährt jeden Slider an seine Grenzen und prüft,
dass kein Python-Fehler auftritt - insbesondere `streamlit.errors.StreamlitDuplicateElementId` (mehrere st.plotly_chart-Aufrufe ohne eindeutiges key= können zufällig identischen Inhalt rendern
und kollidieren) und `StreamlitAPIException` bei Slidern mit berechneten Grenzen (min == max).

Dazu: Anzeigen der Experimente (Verdikte, Tabellen, Hinweise bei Überschreiten der Obergrenze R·T), Permalink-Begrenzung."""

import os

from streamlit.testing.v1 import AppTest

import xab_constants as C

APP_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "app.py")
TIMEOUT = int(os.environ.get("APPTEST_TIMEOUT", "300"))  # unter Last (parallele Läufe) reichen 180 s nicht immer; Umgebungsvariable zum Anpassen
AUTOPLAY_MARKER = "▶"  # Auto-Play-Buttons (Animation mit sleep) im Smoke-Test auslassen


def _fresh_app() -> AppTest:
    at = AppTest.from_file(APP_PATH)
    at.run(timeout=TIMEOUT)
    assert not at.exception, [str(e) for e in at.exception]
    return at


def test_app_loads_without_exception():
    _fresh_app()


def test_every_button_click_does_not_raise():
    labels = [b.label for b in _fresh_app().button]
    assert labels, "keine Buttons gefunden"
    for index, label in enumerate(labels):
        if AUTOPLAY_MARKER in label:
            continue
        at = _fresh_app()
        at.button[index].click().run(timeout=TIMEOUT)
        assert not at.exception, (label, [str(e) for e in at.exception])


def test_every_slider_at_its_min_and_max_does_not_raise():
    labels = [s.label for s in _fresh_app().slider]
    assert labels, "keine Slider gefunden"
    for index, label in enumerate(labels):
        for edge in ("min", "max"):
            at = _fresh_app()
            if at.slider[index].disabled:       # nur beim Lognormal-Prior änderbar (ein Nutzer kann ein deaktiviertes Element nicht bedienen)
                at.radio(key="ln_prior").set_value(C.PRIOR_LOGNORMAL).run(timeout=TIMEOUT)
            slider = at.slider[index]
            value = slider.min if edge == "min" else slider.max
            if isinstance(slider.value, int):
                value = int(value)
            slider.set_value(value).run(timeout=TIMEOUT)
            assert not at.exception, (label, edge, [str(e) for e in at.exception])


def test_default_page_shows_every_experiment_with_a_verdict_and_the_honesty_note():
    at = _fresh_app()
    assert len(at.subheader) >= 5
    texts = " ".join(e.value for kind in (at.success, at.info, at.warning) for e in kind)
    assert "Gamma(34, 21)" in texts and "7,60 %" in texts and "Bis d = 5 hat das Gitter" in texts and "Es gibt keine reale Datenquelle" in texts
    assert len(at.dataframe) >= 7


def test_simulation_cap_shows_a_warning_instead_of_a_simulation():
    at = _fresh_app()
    at.slider(key="gate_R").set_value(400).run(timeout=TIMEOUT)
    assert not at.exception
    assert any("liegen über der Obergrenze" in w.value for w in at.warning)
    at.slider(key="gate_R").set_value(1).run(timeout=TIMEOUT)
    assert not any("liegen über der Obergrenze" in w.value for w in at.warning)


def test_lognormal_regler_are_disabled_with_the_gamma_prior_and_enabled_with_lognormal():
    at = _fresh_app()
    assert at.slider(key="ln_median").disabled and at.slider(key="ln_s0").disabled
    at.radio(key="ln_prior").set_value(C.PRIOR_LOGNORMAL).run(timeout=TIMEOUT)
    assert not at.exception and not at.slider(key="ln_median").disabled and not at.slider(key="ln_s0").disabled


def test_split_checkbox_shows_the_sequential_update():
    at = _fresh_app()
    assert not any("Reihenfolge der Daten egal" in m.value for m in at.markdown)
    at.checkbox(key="gp_split").set_value(True).run(timeout=TIMEOUT)
    assert not at.exception
    assert any("Reihenfolge der Daten egal" in m.value and "Gamma(34, 21)" in m.value for m in at.markdown)
