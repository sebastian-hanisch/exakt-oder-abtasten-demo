"""Zahlen im deutschen Format: Dezimalkomma, Gruppierung mit Leerzeichen ab fünf Stellen (DIN 5008); eigene Formatfunktionen für Tabellen und Texte."""

from __future__ import annotations

import math


def ganz(n: float) -> str:
    """16796 → „16 796“, 1430 → „1430“."""
    s = str(abs(int(round(n))))
    if len(s) >= 5:
        gruppen = []
        while s:
            gruppen.append(s[-3:])
            s = s[:-3]
        s = " ".join(reversed(gruppen))
    return ("-" if n < 0 else "") + s


def dez(x: float, stellen: int = 2) -> str:
    return f"{x:.{stellen}f}".replace(".", ",")


def prozent(anteil: float, stellen: int = 1) -> str:
    """Anteil 0…1 als Prozentangabe: 0,0760 → „7,6 %“ bei einer Stelle."""
    return f"{dez(anteil * 100, stellen)} %"


def potenz(x: float, stellen: int = 1) -> str:
    """Kleine oder große Zahl in der Schreibweise 7,4·10⁻⁶."""
    if x == 0:
        return "0"
    mant, exp = f"{x:.{stellen}e}".split("e")
    hoch = str(abs(int(exp))).translate(str.maketrans("0123456789", "⁰¹²³⁴⁵⁶⁷⁸⁹"))
    return f"{mant.replace('.', ',')}·10{'⁻' if int(exp) < 0 else ''}{hoch}"


def rund_ganz(x: float, gueltige: int = 3) -> str:
    """Ganzzahl auf `gueltige` gültige Ziffern gerundet und gruppiert: 71585,5 → „71 600“."""
    if x == 0:
        return "0"
    e = int(math.floor(math.log10(abs(x))))
    q = 10 ** max(e - gueltige + 1, 0)
    return ganz(round(x / q) * q)


def groesse(x: float, gueltige: int = 3) -> str:
    """Anzahl in Dezimalschreibweise bis 10⁶ (gerundet), darüber als Zehnerpotenz."""
    return rund_ganz(x, gueltige) if abs(x) < 1e7 else potenz(x, 1)


def faktor(x: float) -> str:
    """Verhältnis als „2,1-fach“ (eine Stelle ab 1, sonst zwei)."""
    return f"{dez(x, 1 if x >= 1 else 2)}-fach"


def zahl(x: float, stellen: int = 2) -> str:
    """Ganze Zahlen ohne Nachkommastellen, sonst bis `stellen` Stellen ohne überflüssige Nullen: 21,0 → „21“, 2,5 → „2,5“."""
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    return dez(x, stellen).rstrip("0").rstrip(",")
