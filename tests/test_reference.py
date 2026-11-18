"""Vorgerechnete Dateien (xab_dimension.json, xab_messungen.json): Aufbau, Zusammenhang mit den Konstanten und die `--check`-Läufe der Skripte (mit kleinem Aufwand)."""

import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest

import xab_posterior as PO
import xab_reference as REF

ROOT = Path(__file__).resolve().parent.parent


def _tool(name):
    spec = importlib.util.spec_from_file_location(name, ROOT / "tools" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_dimension_file_has_all_dimensions_runs_and_grids():
    d = REF.dimension()
    assert sorted(d["d"], key=int) == ["1", "2", "3", "4", "5", "6"]
    assert d["meta"]["gitter_g"] == [4, 5, 6, 8, 10, 12] and d["meta"]["kontrollpunkte"] == [1000, 4000, 16000, 64000]
    assert d["meta"]["zaehlungen"] == list(PO.DIM_DATA) and d["meta"]["dauer"] == PO.DIM_T
    for e in d["d"].values():
        assert set(e["ketten"]) == {"iso_a", "iso_b", "pre_a"}
        assert set(e["gitter_fehler"]) == {"4", "5", "6", "8", "10", "12"}
        assert len(e["e_u"]) == len(e["sd_u"])
        for run in e["ketten"].values():
            assert run["r"] == 240 and run["steps"] == 64000 and 0.0 < run["ess_per_step"] < 1.0 and 0.0 < run["accept"] < 1.0


def test_dimension_reference_is_the_tensor_grid_of_the_code():
    for d in (1, 2, 3):
        e = REF.dimension()["d"][str(d)]
        mean, sd = PO.tensor_grid_moments(PO.make_dim_model(d), e["referenz_g"])
        assert np.allclose(mean, e["e_u"], rtol=1e-9, atol=1e-12) and np.allclose(sd, e["sd_u"], rtol=1e-9)


def test_dimension_grid_errors_fall_with_g_in_the_measured_series():
    for d in range(1, 7):
        err = REF.dimension()["d"][str(d)]["gitter_fehler"]
        e_u = [err[g][0] for g in ("4", "5", "6", "8", "10", "12")]
        assert e_u[0] > e_u[1] > e_u[4] > e_u[5]                       # nur die Punkte 4, 5, 10, 12; bei G = 6 und 8 steht ein Kastenfehler (nicht monoton über die ganze Folge)


def test_messungen_file_has_all_sections():
    m = REF.messungen()
    assert len(m["gate_ueberdeckung"]) == 4 and len(m["gamma_ketten"]) == 5
    assert m["gamma_direkt"]["ziehungen"] == 20000 and m["lognormal_kette"]["ketten"] == 400


def test_dimension_check_script_passes_with_small_effort():
    tool = _tool("xab_make_dimension")
    assert tool.check(dims=(1, 2), r=30, n_steps=2000) == []


def test_dimension_check_script_detects_a_changed_reference(tmp_path):
    tool = _tool("xab_make_dimension")
    data = json.loads((ROOT / "xab_dimension.json").read_text(encoding="utf-8"))
    data["d"]["1"]["e_u"][0] += 0.01
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    problems = tool.check(path=bad, dims=(1,), r=20, n_steps=1000)
    assert any("Referenz-Momente" in p for p in problems)


def test_messungen_check_script_passes():
    tool = _tool("xab_make_messungen")
    assert tool.check() == []


def test_messungen_check_script_detects_a_changed_ess(tmp_path):
    tool = _tool("xab_make_messungen")
    data = json.loads((ROOT / "xab_messungen.json").read_text(encoding="utf-8"))
    data["gamma_ketten"][2]["ess_je_schritt"] *= 5
    bad = tmp_path / "bad.json"
    bad.write_text(json.dumps(data), encoding="utf-8")
    assert any("Schritt 2.4 sd" in p for p in tool.check(path=bad))


@pytest.mark.parametrize("name", ["xab_make_dimension", "xab_make_messungen"])
def test_tools_import_nothing_from_other_demos(name):
    text = (ROOT / "tools" / f"{name}.py").read_text(encoding="utf-8")
    assert "import mkv" not in text and "from mkv" not in text
