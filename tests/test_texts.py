"""Texte: Wortliste (keine Begriffe aus der eigenen Planung, kein „Saatwert“, „trägt“, „Wärmebad“, „läuft lokal“), „Zufalls-Seed“, echte Umlaute, App-Text und README stimmen überein.

Die Zeile „Mehr zu den Demos: …“ bzw. „Mehr zur Reihe: …“ des Portfolio-Footers wird von set_demo_footer.py erzeugt und ist kein Planungstext: sie ist ausgenommen."""

import re
from pathlib import Path

import xab_texts as TX

ROOT = Path(__file__).resolve().parent.parent
FORBIDDEN = [r"\bStück\w*", r"\bLinie\w*", r"\bVormessung\w*", r"\bPlan\b", r"\bPläne\b", r"\bV1\b", r"\bmc_v\w*", r"\bPhase 0\b", r"\bKantenabgleich\w*", r"\bgeplant\w*", r"\bStandardfall\b",
             r"\bSaatwert\w*", r"\bträgt\b", r"\btragen\b", r"\bWärmebad\b", r"läuft lokal", r"\bSchwanz\b", r"\bReihe\b"]
SELF = Path(__file__).name


def _lines(path):
    for i, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if "Diese Demo ist Teil des Portfolios" in line or "ueber-mich.html" in line or "Mehr zur Reihe:" in line:
            continue
        yield i, line


def _files():
    files = [ROOT / "README.md"] + sorted(ROOT.glob("*.py")) + sorted((ROOT / "tests").glob("*.py")) + sorted((ROOT / "tools").glob("*.py"))
    return [f for f in files if f.name != SELF]


def test_no_planning_words_and_no_forbidden_words_in_any_text():
    hits = []
    for f in _files():
        for i, line in _lines(f):
            for pat in FORBIDDEN:
                if re.search(pat, line):
                    hits.append(f"{f.name}:{i}: {pat}")
    assert not hits, hits


def test_zufalls_seed_is_never_written_as_a_bare_seed_in_texts():
    hits = []
    for f in [ROOT / "README.md", ROOT / "app.py", ROOT / "xab_texts.py", ROOT / "xab_constants.py", ROOT / "xab_evaluation.py"]:
        for i, line in _lines(f):
            if re.search(r"(?<![A-Za-z_\-])Seed\b", line) and "Zufalls-Seed" not in line:
                hits.append(f"{f.name}:{i}")
    assert not hits, hits


def test_no_ascii_replacements_of_umlauts_in_texts():
    hits = []
    for f in _files():
        for i, line in _lines(f):
            if re.search(r"\b(fuer|ueber|Ueber|waehl\w*|koennen|muessen|Loesung\w*|Aenderung\w*|Uebergang\w*)\b", line):
                hits.append(f"{f.name}:{i}")
    assert not hits, hits


def test_honesty_text_of_the_app_is_in_the_readme():
    normalised = " ".join(re.sub(r"\*\*", "", (ROOT / "README.md").read_text(encoding="utf-8")).split())
    assert " ".join(TX.HONESTY_MARKDOWN.split()) in normalised


def test_app_names_the_expanders_and_sections_it_announces():
    app = (ROOT / "app.py").read_text(encoding="utf-8")
    for name in ("🧰 Was hilft das Ausrechnen?", "🚧 Wo die Annahmen enden", "📐 Mathematische Formulierung", "❓ Kernfragen"):
        assert name in app


def test_formula_block_has_one_display_equation_per_dollar_block_and_escaped_percent():
    text = TX.FORMULAS_MARKDOWN
    assert text.count("$$") % 2 == 0
    for block in re.findall(r"\$\$(.*?)\$\$", text, flags=re.S):
        assert "%" not in block.replace("\\%", "")
        assert "$$" not in block and block.count("\n\n") == 0
    app = (ROOT / "app.py").read_text(encoding="utf-8")                     # die Lesehilfe steht im erzeugten Block der app.py (set_demo_explainer.py), nicht mehr im Formelblock
    assert "Lesehilfe" not in text and "Lesehilfe" in app and "| Symbol | Bedeutung |" in app and "vollständiges Modell" in app


def test_boundaries_table_links_only_neighbour_demos_without_planning_remarks():
    for url in re.findall(r"\]\((https://github\.com/sebastian-hanisch/[a-z0-9-]+)\)", TX.BOUNDARIES_MARKDOWN):
        assert url.endswith("-demo")
    assert "geplant" not in TX.BOUNDARIES_MARKDOWN
