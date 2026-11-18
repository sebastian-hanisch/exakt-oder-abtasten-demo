"""Gemeinsame Fixtures: die Mini-Instanzen, an denen jede Einheit von Hand geprüft wird (DEMO-PLAYBOOK §1b)."""

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests"))

from fraktionen import FractionGate  # noqa: E402


@pytest.fixture
def mini_gate():
    """Gate mit 2 Spuren, 3 Plätzen insgesamt, Angebot a = 1: vier Zustände. Handrechnung: π = (4, 4, 2, 1)/11, Verlust 1/11, t_mix(1/4) = 4 (siehe tests/test_chain.py)."""
    return FractionGate(2, 3, 1)
