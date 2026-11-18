# Wann man nicht abtasten muss – Streamlit-Demo

<!-- TODO nach dem Deploy: direkt hierher den Live-Link setzen (Leerzeile davor und danach):
**[→ Demo live ausprobieren](https://sebastianhanisch-exakt-oder-abtasten-demo.streamlit.app/)** -->

Einstieg in die „Konzepte“-Themen zum Abtasten von Verteilungen mit Markov-Ketten (MCMC) im Portfolio von [Sebastian Hanisch](https://sebastianhanisch.net) (Operations Research und Machine Learning).
Die Demo hebt nichts auf, sie stellt dieselbe Frage zweimal: **Wie groß ist der Anteil, der Mittelwert, der Posterior?** Einmal **ausgerechnet** (Gleichungen, Formel, Gitter), einmal **abgetastet**
(Kette). Die Annahme, die sie durchgehend macht und die die folgenden Demos aufheben: **das Ziel lässt sich an jedem Punkt auswerten, und seine Normierung ist bekannt oder ausrechenbar.**
Die nächste Demo kehrt die Richtung um (ein Gleichgewicht ist vorgegeben, die Kette wird dazu gebaut): [kette-gleichgewicht-demo](https://github.com/sebastian-hanisch/kette-gleichgewicht-demo).

**Keine Neuheit:** alles hier ist Lehrbuchstoff (Levin/Peres/Wilmer für Markov-Ketten und Mischzeit, konjugate Posteriors, Gitterquadratur). Neu ist nur die Vorführbarkeit mit einer Zähleinheit
(Gleichungen, Dichteauswertungen, Schritte) und exakten Orakeln.

## Warum dieses Problem

Wer eine Verteilung nicht ziehen kann, greift zur Kette. Die Falle: eine Kette bauen, bevor man geprüft hat, ob sich das Ziel ausrechnen lässt. Die Demo zeigt in vier Experimenten, wo das Ausrechnen reicht und
wo erst die Kette sinnvoll wird, und der **Negativbefund** steht vorn: in diesen ausgedachten Modellen verliert die Kette bis zu fünf gekoppelten Raten, erst bei sechs gewinnt sie (Genauigkeit 0,01 sd, Gitter mit G = 10 Punkten je Achse, Zufallsschritt-Metropolis mit festem Schritt; Zähleinheit Dichteauswertungen bzw. Schritte).

Kernfragen der Seite: (1) Lässt sich das, was ich wissen will, ausrechnen? (2) Wie viele Gleichungen, Dichteauswertungen oder Schritte kostet es ausgerechnet, wie viele abgetastet? (3) Ab wann lohnt die Kette?

## Modell

**Es gibt keine reale Datenquelle:** die Zählungen sind ausgedacht (32 Ankünfte in 20 Abfertigungsdauern; sechs Schichten mit je 6 bis 11 Ankünften in 5), und die Gate-Kette stammt aus [markov-queue-demo](https://github.com/sebastian-hanisch/markov-queue-demo). In den gelesenen Quellen ist gezeigt, dass sich solche Posteriors mit MCMC berechnen lassen, nicht, dass das bessere Betriebsentscheidungen liefert. Gezählt werden Gleichungen, Dichteauswertungen und Schritte, nicht Sekunden.

- **Gate-Kette** (`xab_chain.py`, neu geschrieben nach den Formeln der Nachbar-Demo): Zustand n = Zahl der Lkw im System (0 bis K), K = Plätze **insgesamt, Spuren eingeschlossen** (Voreinstellung: 2 Spuren, K = 6, also 7 Zustände; „6 Plätze“ heißt
  nicht „6 Warteplätze“), Angebot a = Last · c in Erlang (mittlere Abfertigungsdauer 1), Abgangsrate min(n, c). **Uniformisierte** Kette P = I + Q/Λ mit Λ = größte Austrittsrate (gleich a + c, solange K > c): ein Schritt ist 1/Λ Abfertigungsdauern
  (Voreinstellung: 1/3,6). **Verlust** = π_K (der Zeitanteil im Zustand K; gleich dem Anteil abgewiesener Ankünfte, wenn die Ankünfte Poisson sind, so führt es die Nachbar-Demo).
- **TV-Mischzeit** t_mix(1/4) mit d(t) = max_x TV(P^t(x, ·), π) (Levin/Peres/Wilmer; die d̄-Fassung kommt hier nicht vor). **ESS je Schritt** = Var(f)/σ² für eine Anzeige f, σ² = **exakte** asymptotische Varianz der stationären Kette
  (1 = so gut wie unabhängige Ziehungen): das **Orakel**, nicht der Schätzer (Schätzer, R̂ und Burn-in: [posterior-sampling-demo](https://github.com/sebastian-hanisch/posterior-sampling-demo)).
- **Gamma–Poisson:** Ankünfte in der Dauer T sind Poisson(λ·T), Prior Gamma(α, β) (Form α, Rate β), Posterior Gamma(α + n, β + T). **Nichtkonjugater Prior:** ln λ ~ N(ln Median, s₀²). **Mehrere Raten (Modell D, ausgedacht):** d Schichten mit u_j = ln λ_j,
  Prior gauß (Median 2, sd 0,5, Korrelation 0,7^|i−j|), Zählungen n = (8, 6, 11, 9, 7, 10) in T = 5.
- **„Genauigkeit“** = RMSE des Mittels in Einheiten der Posterior-Standardabweichung (sd): 0,01 sd heißt, der Fehler des Mittels ist ein Hundertstel der Streuung des Posteriors. In Experiment 4 (mehrere Raten) ist es die Wurzel aus dem mittleren quadratischen Fehler, gemittelt über alle Ketten und alle Koordinaten, jeder Fehler in Einheiten der sd seiner Koordinate.
- **Zähleinheiten:** *Schritte* (Kette), *Gleichungen* (Zustände), *Dichteauswertungen* (Gitterpunkte bzw. Kettenschritte, eine Auswertung je Schritt), *Ziehungen* (unabhängig), *Zeilenzustände* (Übertragungsmatrix). Gleichungen und Schritte sind verschiedene Operationen: der Vergleich im Gate sagt nur etwas über die Größenordnung; gleich gezählt sind Gitterpunkte und Kettenschritte (je eine Dichteauswertung).

## Methodik

- **Experiment 1, ein Gate:** π auf **vier Wegen** (geschlossene Form, lineares System, Eigenvektor, Matrixpotenz P⁵⁰⁰⁰), t_mix(1/4) und t_rel aus Matrixpotenzen und Eigenwerten, σ² und ESS je Schritt aus σ² = 2⟨f̃, Z f̃⟩_π − ⟨f̃, f̃⟩_π mit Z = (I − P + 1π)⁻¹
  (eigene Rechnung, gegen die Autokovarianzsumme und gegen Brüche geprüft), Schritte T = 1,96² σ²/ε² für ein 95-%-Band ±ε. Simulation: R Ketten zu T Schritten (Start leer, kein Einlauf), Zeitanteil im Zustand K, Überdeckung des Bandes.
- **Experiment 2, Gamma–Poisson:** Posterior aus zwei Zahlen; direkte Ziehung als Summe von α + n Exponentialzahlen (`−ln(Produkt der Zufallszahlen)/(β + T)`); Zufallsschritt-Metropolis mit festem Schritt (Vielfache der Posterior-sd) in λ selbst, ein Vorschlag λ ≤ 0 wird verworfen; Start in der exakten
  Verteilung; ESS je Schritt = (1/RMSE)²/N aus R Ketten.
- **Experiment 3, nichtkonjugater Prior:** Gitter in u = ln λ auf Gipfel ± 6 Laplace-Standardabweichungen mit Trapezregel, Referenz ein Gitter mit 4 001 Punkten auf ± 8 sd (in den Tests gegen `scipy.integrate.quad`).
- **Experiment 4, mehrere Raten:** Tensorgitter mit G^d Punkten (live bis 3 Millionen); die Ketten (240 Ketten × 64 000 Schritte, zwei Zufallsfolgen mit isotropem Schritt und ein vorkonditionierter Schritt; Start jeder Kette: Gipfel der Laplace-Näherung plus Laplace-sd mal Normalzahl je Koordinate, kein Einlauf verworfen, anders als in Experiment 2 und 3, die in der exakten Verteilung starten) sind **vorgerechnet** (`xab_dimension.json`, Skript `tools/xab_make_dimension.py`
  mit `--check`). Die 400-Ketten-Messungen der Experimente 1 bis 3 sind ebenfalls vorgerechnet (`xab_messungen.json`, `tools/xab_make_messungen.py`).
- **Zufall nur aus dem ganzzahligen SplitMix64** (`xab_rng.py`, unveränderte Kopie der Vorlage der Website; `xab_sampler.uniforms` ist die vektorisierte Folge, Test gegen die Vorlage). Ganzzahl- und Gleichverteilungsfolge sind plattformfest, `np.log`/`np.cos`/`np.exp` in den Ketten
  nicht in der letzten Stelle: Zufallsanteile stehen als Band, nie als exakte Zahl. Der Zufallsschritt-Metropolis ist hier ein **Messobjekt**, kein Lehrgegenstand (Verfahren und Vorschlagsfallen: [metropolis-kleiner-raum-demo](https://github.com/sebastian-hanisch/metropolis-kleiner-raum-demo)).

## Befunde (gemessen, keine Behauptungen)

Zahlen mit ihrer Voraussetzung; die Spalte „Beleg“ nennt den Test oder die eingefrorene Messung (`tests/test_claims.py`, wenn nichts anderes steht). Exakte Werte sind Brüche oder Ganzzahlen bzw. Gleitkommazahlen auf 10⁻⁹ relativ; Zufallsanteile sind Bänder.

**Experiment 1 – die fünf Gate-Fälle** (Zähleinheit Schritte der uniformisierten Kette; Genauigkeit ±0,005 im Verlust, 95 %, asymptotisch; ESS je Schritt für die stationäre Kette):

| Gate (Spuren c, Plätze K, Last) | Verlust | t_mix(1/4) | t_rel | ESS je Schritt (Verlust) | ESS je Schritt (Zahl im System) | Schritte für ±0,005 |
|---|---|---|---|---|---|---|
| Voreinstellung (2, 6, 80 %) | 7,60 % | 10 | 8,58 | 0,151 | 0,063 | 71 585 |
| eine Spur (1, 6, 80 %) | 6,63 % | 12 | 9,56 | 0,140 | 0,056 | 67 886 |
| Überlast (2, 6, 150 %) | 35,96 % | 12 | 7,57 | 0,174 | 0,073 | 203 884 |
| vier Spuren, kein Warteplatz (4, 4, 80 %) | 22,81 % (= Erlang B) | 5 | 3,73 | 0,358 | 0,161 | 75 684 |
| Kette mit 6 Zuständen (2, 5, 80 %) | 10,27 % | 7 | 6,39 | 0,182 | 0,086 | 77 908 |

Beleg: `test_gate_table`. Die Voreinstellung (K = 6) hat t_mix(1/4) = 10; die Zahl 7 gilt für K = 5 (6 Zustände).

| Frage | Befund | Beleg |
|---|---|---|
| Wie viele Gleichungen für π? | Voreinstellung: **7 Gleichungen** (K + 1 Zustände); die vier Wege stimmen auf unter 10⁻¹² überein, πP = π und Detailed Balance auf 10⁻¹⁵ | `test_gate_four_ways` |
| Wie viele Schritte für ±0,5 Prozentpunkte? | rund **71 600** (71 585) bei 2 Spuren, K = 6, 80 %: 10-mal genauer kostet 100-mal mehr Schritte (Wurzelgesetz: T ∝ 1/ε²) | `test_gate_table`, `test_square_root_law` |
| Ist die Mischzeit das Problem? | Nein: nach 10 Schritten ist d(t) ≤ 1/4, aber die ESS je Schritt ist 0,151 (Verlustanzeige) und 0,063 (Zahl im System): „die Kette braucht T Schritte“ hängt an der geschätzten Größe | `test_gate_table` |
| Deckt das Band? | Vorgerechnet (400 Ketten, Voreinstellung, SplitMix64): ε = 0,01 (T = 17 897): **96,5 %** (Start leer, kein Einlauf) und 95,75 % (Start aus π); ε = 0,005 (T = 71 586): **95,25 %** und 95,25 %; Erwartung 95 % ± 1,1 Prozentpunkte | `test_gate_coverage_frozen` |
| Folgt die Streuung dem Orakel? | Streuung der Anteile 0,00248 gegen exakt √(σ²/T) = 0,00255 (T = 71 586, Start leer, 400 Ketten) | `test_gate_coverage_frozen` |

**Experiment 2 – Gamma–Poisson** (Prior Gamma(2, 1), 32 Ankünfte in 20 Abfertigungsdauern; ausgedacht):

| Frage | Befund | Beleg |
|---|---|---|
| Posterior? | Exakt **Gamma(34, 21)**: Mittel 1,619, sd 0,278, 95-%-Intervall (1,121; 2,207); Wahrscheinlichkeit der Zählung (Evidenz) 1,570427·10⁻² auf zwei Wegen (Quadratur, negativ-binomial) | `test_gamma_poisson_numbers` |
| Reihenfolge der Daten? | 10 Ankünfte in 7, dann 22 in 13 gibt dasselbe Gamma(34, 21) wie 32 in 20 in einem Schritt | `test_gamma_poisson_numbers` |
| Direkte Ziehung? | 20 000 Ziehungen (Summe von 34 Exponentialzahlen/21, Zufalls-Seed 7301): Mittel 1,6161, sd 0,2787; Kolmogorow-Smirnow D = 0,0071, p = 0,27 | `test_direct_draws_frozen` |
| Kette? | Zufallsschritt-Metropolis, Start in der exakten Verteilung, 400 Ketten × 5 000 Schritte: ESS je Schritt **0,041 / 0,124 / 0,230 / 0,173 / 0,068** bei Schritt 0,5 / 1 / 2,4 / 5 / 12 sd (Annahmerate 0,84 / 0,70 / 0,44 / 0,24 / 0,10; die ESS-Werte aus 400 Ketten streuen von Lauf zu Lauf, die Reihenfolge der Schrittweiten nicht); **in diesem Messbereich (0,5 bis 12 sd) nie besser als die direkte Ziehung** (1) | `test_gamma_chain_frozen` |
| Aufwand für 0,01 sd? | 10 000 unabhängige Ziehungen gegen rund **43 500 Kettenschritte** (Schritt 2,4 sd, aus der ESS je Schritt, Wurzelgesetz; ein Neu-Lauf mit anderen Zufalls-Seeds kann um einige Prozent abweichen) gegen 0 (exakt) | `test_gamma_chain_frozen` |
| Start außerhalb? | Mit überstreutem Start (λ = 0,3 / 1 / 2,5 / 4, je 100 der 400 Ketten, Schritt 2,4 sd, 5 000 Schritte, kein Einlauf verworfen) ist der RMSE des Mittels 0,031 sd | `test_gamma_chain_frozen` |
| Wenig Daten? | 3 Ankünfte in 2: Gamma(5, 3), Mittel 5/3, Varianz 5/9, Evidenz 32/243 (Handbeispiel) | `tests/test_posterior.py` |

**Experiment 3 – anderer Prior** (n = 32 in T = 20, Prior ln λ ~ N(ln 2, 0,5²), Gitter in u auf Gipfel ± 6 Laplace-sd):

| Frage | Befund | Beleg |
|---|---|---|
| Ist der Posterior ein Gamma? | Nein: E[λ] = **1,64215** gegen 1,61905 im konjugaten Fall (Differenz 0,0231); Referenz E[u] = 0,48240, sd[u] = 0,16564 (Quadratur) | `test_lognormal_grid` |
| Reicht ein Gitter? | Fehler (E[u] in sd / sd[u] relativ / E[λ] in sd): G = 6: 9,2·10⁻³ / 0,21 / 8,0·10⁻²; G = 8: 6,1·10⁻³ / 1,7·10⁻² / 1,4·10⁻²; **G = 10: 2,5·10⁻⁴ / 1,0·10⁻⁴ / 4,0·10⁻⁴**; G = 12: 1,0·10⁻⁶ / 1,0·10⁻⁵ / 1,1·10⁻⁶; G = 4: bis 0,95 (zu grob) | `test_lognormal_grid` |
| Kette? | Schritt 2,4 sd, 400 Ketten × 16 000 Schritte: RMSE von E[u] 0,135 / 0,068 / 0,033 / 0,017 sd bei N = 250 / 1 000 / 4 000 / 16 000, ESS je Schritt 0,224, Annahmerate 0,442 | `test_lognormal_chain_frozen` |
| Aufwand für 0,01 sd? | **G = 10 Dichteauswertungen** gegen rund **44 600 Schritte**; für 10⁻⁴ sd G = 12 gegen rund 4,5·10⁸ Schritte (**hochgerechnet** nach dem Wurzelgesetz, nicht simuliert) | `test_lognormal_chain_frozen` |

**Experiment 4 – mehrere gekoppelte Raten** (Modell D; Gitter G = 10 je Achse; 240 Ketten × 64 000 Schritte je Zeile, vorgerechnet; Kettenschritte für 0,01 sd = 10 000/ESS je Schritt; „Folge A“ und „Folge B“: zwei Zufallsfolgen mit isotropem Schritt 2,38/√d · mittlere sd; „vorkonditioniert“: Schritt 2,38/√d mal Cholesky-Faktor der Laplace-Kovarianz):

| d | Gitterpunkte 10^d | Kettenschritte Folge A | Folge B | vorkonditioniert | ESS je Schritt (A) | ESS je Schritt × d (A) | Annahmerate (A) | Gitterfehler G = 10 (E[u], sd[u]) |
|---|---|---|---|---|---|---|---|---|
| 1 | 10 | 46 900 | 49 300 | 44 200 | 0,213 | 0,213 | 0,445 | 2,8·10⁻⁴, 1,2·10⁻⁴ |
| 2 | 100 | 89 100 | 83 300 | 65 800 | 0,112 | 0,225 | 0,339 | 3,3·10⁻⁴, 7,9·10⁻⁵ |
| 3 | 1 000 | 116 000 | 132 000 | 102 000 | 0,086 | 0,259 | 0,298 | 3,5·10⁻⁴, 2,0·10⁻⁴ |
| 4 | 10 000 | 156 000 | 181 000 | 136 000 | 0,064 | 0,256 | 0,275 | 3,5·10⁻⁴, 2,2·10⁻⁴ |
| 5 | 100 000 | 194 000 | 210 000 | 152 000 | 0,051 | 0,257 | 0,259 | 3,5·10⁻⁴, 2,2·10⁻⁴ |
| 6 | 1 000 000 | 246 000 | 260 000 | 196 000 | 0,041 | 0,244 | 0,250 | 3,5·10⁻⁴, 2,3·10⁻⁴ |

Beleg: `test_dimension_table_frozen`. Die Zeile d = 1 (46 900 Schritte, ESS je Schritt 0,213) und der Aufwand aus Experiment 3 (rund 44 600 Schritte, ESS je Schritt 0,224) gehören zu verschiedenen Modellen und Ensembles (n = 8 in T = 5 gegen 32 in 20, 240 gegen 400 Ketten, anderer Start, andere Zufallsfolge) und sind nicht dieselbe Messung. **Bis d = 5 hat das Gitter weniger Dichteauswertungen als die Kette Schritte, ab d = 6 die Kette**, in allen drei vorgerechneten Läufen. In der Gitterliste 4, 5, 6, 8, 10, 12 (die Rechenskripte nehmen alle sechs, die Fehlerkurve der App nur 6, 8, 10, 12; G = 4 und 5 reichen nicht) ist G = 10 je Achse das kleinste G, das 0,01 sd erreicht, **unabhängig von d** (gemessen d ≤ 6; G = 8 reicht nicht, die relative Abweichung von sd[u_j] liegt dort bei 1,7·10⁻² bis 1,9·10⁻²; Werte dazwischen, etwa G = 9, sind nicht ausgewertet): größter Fehler in E[u_j]
2,8·10⁻⁴ bis 3,5·10⁻⁴ sd, in sd[u_j] relativ 7,9·10⁻⁵ bis 2,3·10⁻⁴. Der RMSE fällt in allen 18 Zeilen wie 1/√N (Verhältnis N = 1 000 zu N = 64 000 zwischen 6 und 10, erwartet 8; d = 6, Folge A: 0,165 / 0,079 / 0,038 / 0,020 bei N = 1 000 / 4 000 / 16 000 / 64 000). Für 10⁻⁴ sd (G = 12) wäre die Kreuzung erst bei **d = 9**
(**hochgerechnet** mit ESS je Schritt ≈ Mittel(ESS · d)/d über d = 1 bis 6, nicht gemessen; `test_dimension_extrapolation`; die Dimension hängt an der Gitterliste: mit einem kleineren G zwischen 10 und 12, das 10⁻⁴ sd erreicht, läge die Kreuzung anders). ESS je Schritt × d liegt in 18 Zeilen zwischen 0,20 und 0,33 (nur Beobachtung).

| Frage | Befund | Beleg |
|---|---|---|
| Ring ohne Sampler? | Z(λ) = Spur(Tⁿ) mit T = [[1, λ], [1, 0]] gleich der Aufzählung aller Belegungen, ganzzahlig: λ = 3, n = 6 / 12 / 20: **154 / 22 258 / 17 580 751** bei 18 / 322 / 15 127 Belegungen. Der Ring ist exakt lösbar und dient den Nachbar-Demos als Orakel; das gilt für Mittelwerte und Z(λ), nicht für die Mischzeit | `test_ring_table` |
| Passen die Werte der Nachbar-Demo? | Voreinstellung: Verlust 7,60 %, 2,45 Lkw im System, Wartezeit 1,98 min (Annahme: 3 min mittlere Abfertigung, `MEAN_SERVICE_MIN`), Auslastung 73,9 %; vier Spuren: 22,8 % = Erlang B (Stand der Einsicht 2026-11-18) | `tests/test_literatur_gleichgewicht.py` |

## Befunde und Korrekturen gegenüber der ersten Messreihe

Die frühere Messreihe (gleiche Modelle, Zufall aus numpy) ist eine Erwartung, keine README-Zahl; neu mit SplitMix64 gemessen:

- **Überdeckung:** 96,5 % / 95,75 % / 95,25 % / 95,25 % statt 93,8 % / 95,0 % / 94,5 % / 95,8 % der früheren Messreihe; die vier neuen Werte liegen innerhalb von zwei Standardfehlern um 95 % (ein Standardfehler sind 1,1 Prozentpunkte bei 400 Ketten); `test_gate_coverage_frozen`.
- **Gate mit K = c:** Λ ist die größte Austrittsrate, nicht a + c (für K > c gleich). Für vier Spuren ohne Warteplatz (Λ = 6,2, nicht 7,2) gilt t_mix(1/4) = 5, wie in der früheren Messreihe.
- **Gitterfehler bei G = 12** (1·10⁻⁶ bis 1·10⁻⁵ statt 2,6·10⁻⁷ bis 6,7·10⁻⁶): die Trapezregel gewichtet die Randpunkte mit ½, die frühere Messreihe summierte ungewichtet; Aussage (G = 10 reicht für 0,01 sd) unverändert.
- **Zweite Zufallsfolge, R = 240 und vorkonditionierter Schritt für Experiment 4:** die Kreuzung bei d = 6 gilt in allen drei Läufen; die Kettenschritte der beiden isotropen Läufe weichen um bis zu 16 % voneinander ab (d = 4: 156 000 gegen 181 000), der vorkonditionierte Schritt braucht gegenüber Folge A 6 % (d = 1) bis 26 % (d = 2) weniger Schritte, verschiebt die Kreuzung aber nicht (`test_dimension_table_frozen`).
- **ESS je Schritt × d** schwankt zwischen den Läufen (0,20 bis 0,33), die frühere Aussage „bleibt bei 0,22 bis 0,27“ gilt für die isotropen Läufe nicht mehr als Band; es bleibt eine Beobachtung.
- **Live-Grenzen:** R · T höchstens 1 000 000 Schritte in Experiment 1 (reine Python-Schleife); 2 000 direkte Ziehungen live (ein `exact_log` je Ziehung); Experimente 2 und 3 live mit höchstens 10 000 Schritten × 200 Ketten.
- **ESS je Schritt in der App ist geschätzt** (aus R Ketten) und streut bei 60 Ketten um etwa ein Fünftel bis ein Viertel (Messung: zwölf Wiederholungen, Schritt 1 sd); die vorgerechneten Werte (400 Ketten) stehen daneben.

## Ehrliche Grenzen

- **Ausgedachte Modelle:** Poisson-Zählungen, unabhängige Ankünfte, konstante Rate, ein Gate; keine reale Datenquelle. Aussagen gelten für diese Zählungen, diese Priors, ein Gitter um den bekannten Gipfel, Zufallsschritt-Metropolis mit festem Schritt (isotrop, in Experiment 4 auch vorkonditioniert), Zähleinheit Dichteauswertungen bzw. Schritte.
- **„Keine Kette nötig“** heißt: in diesen Modellen, bis zu fünf gekoppelte Raten, G = 10 je Achse, 0,01 sd. Mit besserer Kette (Gibbs, Langevin, HMC) verschiebt sich die Kreuzung zu weniger Parametern; nicht gemessen.
- **Das Gitter setzt voraus, dass man weiß, wo die Masse liegt** (Gipfel und Laplace-Breite für den Kasten). Bei mehreren Gipfeln versagt dieses Vorgehen; ein Gitter mit unbekannter Lage, schiefe und mehrgipfelige Posteriors sind nicht gemessen.
- **Hochrechnungen** (10⁻⁴ sd, d ≥ 7) sind nicht simuliert; die ESS je Schritt × d ist eine Beobachtung, kein Gesetz (nur sechs Dimensionen, drei Läufe).
- **Nicht gemessen:** latente Variablen (fehlende Abgangszeiten), M/G/1 und Weibull-Modelle (dort ist nach den gelesenen Quellen MCMC nötig: Sutton/Jordan, Shestopaloff/Neal; deren Beschleunigungsfaktoren sind Aussagen der Quellen, hier nicht nachgerechnet); Entscheidungsgüte; Zeiten auf Streamlit Cloud.
- **Zahlen nur auf diesem Stand gemessen** (Windows, Python 3.14, numpy 2.5, scipy 1.18); Ganzzahlen und Brüche sind plattformfest, Gleitkommazahlen und Zufallsanteile stehen als Band.
- **Quellen so, wie sie gelesen sind:** Levin/Peres/Wilmer (ganzes Buch), Asanjarani/Nazarathy/Taylor (Abstract und Abschnitt 3.7), Aktekin/Soyer (Abschnitte 1, 4, Anhang), Sutton/Jordan (Abstract, Abschnitt 3), Shestopaloff/Neal (Abstract, Abschnitt 1). Nicht gelesen: Norris, Handbook of Markov Chain Monte Carlo, Robert/Casella.
  Die Formel für σ² ist Lehrbuchstoff, hier eigene Rechnung (Quelle nicht gelesen).

## Verifikation

- **Kern gegen Handrechnung** (`tests/test_chain.py`, `tests/test_posterior.py`): Mini-Gate (2 Spuren, 3 Plätze, Angebot 1): π = (4, 4, 2, 1)/11, d(0…7), t_mix = 4, σ² = 310/1331, ESS 11/31; Gamma(5, 3), Evidenz 32/243, abgeschnittener Prior 1,684606.
- **Orakel-Tests** (`tests/test_oracle_exakt_oder_abtasten.py`, anderer Rechenweg): π des Gates auf vier Wegen gegen Brüche (fünf Voreinstellungen, 40 Zufallsinstanzen); d(t) und t_mix gegen Matrixprodukt in Brüchen; σ² gegen Brüche und Autokovarianzsumme; Gamma-Posterior und Evidenz gegen
  Quadratur und negativ-binomial; Gitter gegen `scipy.integrate.quad`, `dblquad` und feineres Gitter; Ring gegen Aufzählung. Literaturwerte: `tests/test_literatur_gleichgewicht.py`. Eigenschaftstests (Datenreihenfolge, Zeitskalierung, α, β → 0, Monotonie): `tests/test_eigenschaften_exakt_oder_abtasten.py`.
- **Zufall:** vektorisierte SplitMix64-Folge gleich der Vorlage, simulierte Übergänge gleich den Zeilen der Matrix, Streuung der Anteile gleich √(σ²/T) (`tests/test_sampler.py`).
- **Claims, Verdikte, Randfälle:** `tests/test_claims.py` (jede Zahl dieser README), `tests/test_verdicts.py` (Verdikte als reine Funktionen, jeder Zweig), `tests/test_edge_cases.py` (n = 0, K = c, c = 1, Last 150 %, G = 4, R = 1 …), `tests/test_texts.py` (Texte, Wortliste).
- **Oberfläche:** AppTest-Rauchtest (jeder Button, jeder Regler am Minimum und Maximum), Presets, Layout (Schnellstart in der Seitenleiste, aktives Preset hervorgehoben; jede Änderung eines Reglers hebt die Markierung auf), Footer, Laufzeit-Importe.
- **Vorgerechnete Dateien** prüft `python tools/xab_make_dimension.py --check` und `python tools/xab_make_messungen.py --check` (Referenz und Gitterfehler exakt, Ketten als Größenordnung; `tests/test_reference.py`).
- **Testzahl:** 208 Tests.

## Dateistruktur

| Datei | Zweck |
|---|---|
| `.github/workflows/tests.yml` | CI: Push, Pull Request, wöchentlicher Canary |
| `.gitignore` | venv, Caches, Streamlit-Secrets |
| `.streamlit/config.toml` | Streamlit-Einstellungen (keine Nutzungsstatistik) |
| `app.py` | Streamlit-App: Kernfragen, vier Experimente, Grenzen, Formeln |
| `requirements.txt` | Laufzeit-Bibliotheken (ungepinnt): streamlit, numpy, scipy, pandas, plotly |
| `requirements-dev.txt` | Laufzeit plus pytest |
| `xab_chain.py` | Gate-Kette: π auf vier Wegen, d(t), t_mix, t_rel, σ², ESS je Schritt, Erlang B |
| `xab_posterior.py` | Gamma–Poisson, Gitter in u = ln λ, Tensorgitter, Laplace-Kasten |
| `xab_sampler.py` | vektorisierter SplitMix64, Gate-Simulation, direkte Gamma-Ziehung, Zufallsschritt-Metropolis |
| `xab_rng.py` | SplitMix64 (unveränderte Kopie der Vorlage der Website) |
| `xab_reference.py` | lädt die vorgerechneten Messungen |
| `xab_evaluation.py` | die vier Experimente, Verdikte als reine Funktionen, Ring-Tabelle |
| `xab_constants.py`, `xab_presets.py`, `xab_layout.py`, `xab_format.py`, `xab_texts.py`, `xab_visualization.py` | Regler-Grenzen und Presets, Permalink, fester Kopf, Zahlenformate, Texte, Abbildungen |
| `xab_dimension.json`, `xab_messungen.json` | vorgerechnete Messungen (Experiment 4; Experimente 1 bis 3 mit 400 Ketten) |
| `tools/xab_make_dimension.py`, `tools/xab_make_messungen.py` | rechnen die JSON-Dateien neu (`--check` prüft sie) |
| `tests/` | Handrechnung, Orakel, Literatur, Eigenschaften, Claims, Verdikte, Randfälle, Zufall, Presets, Layout, AppTest, Footer, Laufzeit-Importe |

## Bewusst nicht umgesetzt

- **Torus, Direktsampler und Ziehen aus der Übertragungsmatrix:** [sampler-verfahrenswahl-demo](https://github.com/sebastian-hanisch/sampler-verfahrenswahl-demo); hier nur der Ring als Orakel.
- **Metropolis als Verfahren, Vorschlagsfallen, Schrittweite und Dimension:** [metropolis-kleiner-raum-demo](https://github.com/sebastian-hanisch/metropolis-kleiner-raum-demo), [dimension-hmc-demo](https://github.com/sebastian-hanisch/dimension-hmc-demo); R̂, ESS-Schätzer, Burn-in, pCN: [posterior-sampling-demo](https://github.com/sebastian-hanisch/posterior-sampling-demo).
- **Mischzeit im Vergleich zur Spektrallücke:** [mischzeit-exakt-demo](https://github.com/sebastian-hanisch/mischzeit-exakt-demo); Batch Means und Warm-up: [output-analysis-demo](https://github.com/sebastian-hanisch/output-analysis-demo); unabhängige Ziehungen und der Fehler ∝ 1/√N: [monte-carlo-schaetzer-demo](https://github.com/sebastian-hanisch/monte-carlo-schaetzer-demo).
- **Gamma mit nichtganzer Form** (die Vorlage hat keinen Gamma-Sampler; die direkte Ziehung braucht ganze Form), **Gibbs, vorkonditionierter Schritt als Regler, Langevin, HMC**, latente Variablen, PDF-Export.

## Lokal ausführen

```bash
python3 -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate

pip install -r requirements.txt
streamlit run app.py
```

## Tests ausführen

```bash
pip install -r requirements-dev.txt
pytest tests/ -v
```

Gebaut mit Streamlit, Plotly, NumPy, SciPy und pandas.

---

Diese Demo ist Teil des Portfolios von [Sebastian Hanisch](https://sebastianhanisch.net) – Operations Research und Machine Learning ([Über mich](https://sebastianhanisch.net/ueber-mich.html)). Mehr zur Reihe: [MCMC: von der Kette zur Verfahrenswahl](https://sebastianhanisch.net/konzepte-mcmc.html).
