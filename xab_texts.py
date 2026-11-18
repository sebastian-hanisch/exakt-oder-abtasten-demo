"""Texte der App, die auch die README wörtlich führt (ein Test sucht sie dort): Ehrlichkeitshinweis, Grenzen, Formelblock.

Es gibt keine reale Datenquelle: Gate-Kette, Zählungen und Priors sind ausgedacht; jede Aussage gilt nur für diese Modelle."""

GITHUB = "https://github.com/sebastian-hanisch"

HONESTY_MARKDOWN = (
    "Es gibt keine reale Datenquelle: die Zählungen sind ausgedacht (32 Ankünfte in 20 Abfertigungsdauern; sechs Schichten mit je 6 bis 11 Ankünften in 5), "
    f"und die Gate-Kette stammt aus [markov-queue-demo]({GITHUB}/markov-queue-demo). In den gelesenen Quellen ist gezeigt, dass sich solche Posteriors mit MCMC "
    "berechnen lassen, nicht, dass das bessere Betriebsentscheidungen liefert. Gezählt werden Gleichungen, Dichteauswertungen und Schritte, nicht Sekunden."
)

HOW_IT_WORKS_MARKDOWN = """
1. **Das Ziel benennen:** ein Anteil (der Verlust eines Gates), ein Mittelwert oder ein Posterior (die Verteilung einer Ankunftsrate nach den Daten).
2. **Ausrechnen:** Gleichungen lösen (Gate), eine Formel auswerten (Gamma–Poisson) oder die Dichte auf einem Gitter auswerten (andere Priors). Gezählt wird, wie viele Gleichungen
   bzw. Dichteauswertungen das kostet.
3. **Abtasten:** eine Kette laufen lassen, die Zeitmittel bildet. Gezählt werden Schritte; die **ESS je Schritt** sagt, wie viele unabhängige Ziehungen ein Schritt wert ist
   (1 = so gut wie unabhängig). Das Zeitmittel wird mit 1/√N genauer: zehnmal genauer kostet hundertmal mehr Schritte.
4. **Vergleichen, in derselben Zähleinheit:** Gleichungen gegen Schritte (Experiment 1), Dichteauswertungen gegen Kettenschritte (Experimente 3 und 4). Zeiten in Sekunden sind kein Befund. Gleichungen und Schritte sind verschiedene Operationen: dieser Vergleich sagt nur etwas über die Größenordnung.
5. **Die Annahme, die diese Seite durchgehend macht:** das Ziel lässt sich an jedem Punkt auswerten, und seine Normierung ist bekannt oder ausrechenbar. Ohne sie geht es nicht
   mehr ums Ausrechnen.
"""

BOUNDARIES_MARKDOWN = f"""
| Annahme | Was passiert, wenn sie verletzt ist | Wer setzt an |
|---|---|---|
| **Das Ziel lässt sich auswerten, die Normierung ist bekannt oder ausrechenbar** | Ist die Normierung weder bekannt noch ausrechenbar, lässt sich das Ziel nicht ausrechnen; die Kette braucht nur Dichteverhältnisse. | [metropolis-kleiner-raum-demo]({GITHUB}/metropolis-kleiner-raum-demo) |
| **Ausgedachte Poisson-Zählungen, unabhängige Ankünfte, konstante Rate, ein Gate** | Gemessen ist nur dieses Modell (Zählungen 32 in 20, sechs Schichten mit 6 bis 11 Ankünften in 5, Gate mit 7 Zuständen); eine reale Datenquelle gibt es nicht. | – |
| **Das Gitter liegt um den bekannten Gipfel** (Gipfel ± 6 Laplace-Standardabweichungen) | Ein Gitter mit unbekannter Lage der Masse, schiefe oder mehrgipfelige Posteriors sind nicht gemessen; ein Gitter um nur einen Gipfel verfehlt die Masse der anderen. | [gipfel-tempering-demo]({GITHUB}/gipfel-tempering-demo) |
| **Die Kette ist ein Zufallsschritt-Metropolis mit festem, isotropem Schritt** (und ein vorkonditionierter Schritt in der vorgerechneten Tabelle) | Gibbs, Langevin und HMC sind nicht gemessen; mit besserer Kette verschiebt sich die Kreuzung zu weniger Parametern. | [dimension-hmc-demo]({GITHUB}/dimension-hmc-demo), [gibbs-glauber-demo]({GITHUB}/gibbs-glauber-demo) |
| **Genauigkeit 0,01 sd, Tensorgitter bis d = 6** | Die Kreuzung für 10⁻⁴ sd und d ≥ 7 ist hochgerechnet, nicht gemessen. | – |
| **Beobachtete Zählungen** | Latente Variablen (fehlende Abgangszeiten), M/G/1 und Weibull-Modelle sind nicht gemessen; dort ist MCMC nach den gelesenen Quellen nötig (Sutton/Jordan, Shestopaloff/Neal). | – |
| **Keine Entscheidungsgüte** | Dass ein abgetasteter Posterior bessere Betriebsentscheidungen liefert, ist in den gelesenen Quellen nicht gezeigt und hier nicht gemessen. | – |
| **R̂, ESS-Schätzer und Burn-in** | Hier steht die exakte ESS der Kette (aus der Matrix), kein Schätzer; Schätzer und Diagnostik: [posterior-sampling-demo]({GITHUB}/posterior-sampling-demo) (Vehtari u. a.: R̂ < 1,01, mindestens vier Ketten, ESS ≥ 400). | [posterior-sampling-demo]({GITHUB}/posterior-sampling-demo) |
"""

FORMULAS_MARKDOWN = r"""
**Gate-Kette.** Zustand $n \in \{0, \dots, K\}$ ist die Zahl der Lkw im System, die Aufstiegsrate ist $a$ (für $n < K$), die Abstiegsrate $\min(n, c)$. Mit der größten Austrittsrate $\Lambda = \max_n (\text{Aufstieg}_n + \text{Abstieg}_n)$ (das ist $a + c$, solange $K > c$) ist
$P = I + Q/\Lambda$; ein Schritt entspricht $1/\Lambda$ Abfertigungsdauern. Das Gleichgewicht ist
$$\pi_n \propto \prod_{j=1}^{n} \frac{a}{\min(j, c)}, \qquad \text{Verlust} = \pi_K.$$

**Abstand und Mischzeit.** Mit der Totalvariation $\mathrm{TV}(\mu, \nu) = \tfrac12 \sum_x |\mu(x) - \nu(x)|$:
$$d(t) = \max_x \mathrm{TV}\big(P^t(x, \cdot), \pi\big), \qquad t_{\mathrm{mix}} = \min\{t : d(t) \le 1/4\}.$$

**Streuung des Zeitmittels.** Für eine Anzeige $f$ (hier $f = \mathbf 1\{n = K\}$ oder $f = n$), $\tilde f = f - \pi f$ und $Z = (I - P + \mathbf 1 \pi)^{-1}$:
$$\sigma^2 = 2\,\langle \tilde f, Z \tilde f \rangle_\pi - \langle \tilde f, \tilde f \rangle_\pi, \qquad \mathrm{ESS\ je\ Schritt} = \mathrm{Var}_\pi(f)/\sigma^2.$$
Das 95-%-Band des Zeitanteils nach $T$ Schritten hat die Halbbreite $1{,}96\sqrt{\sigma^2/T}$; für ein Band $\pm\varepsilon$ braucht es
$$T = \frac{1{,}96^2\,\sigma^2}{\varepsilon^2} \quad \text{Schritte (asymptotisch, stationärer Start).}$$

**Gamma–Poisson.** Ankünfte in der Dauer $T$ sind Poisson($\lambda T$), der Prior ist Gamma($\alpha, \beta$) mit Dichte $\propto \lambda^{\alpha - 1} e^{-\beta \lambda}$. Dann ist
$$\lambda \mid n \;\sim\; \mathrm{Gamma}(\alpha + n,\; \beta + T), \qquad P(n) = \binom{\alpha + n - 1}{n} \Big(\frac{\beta}{\beta + T}\Big)^{\alpha} \Big(\frac{T}{\beta + T}\Big)^{n}.$$
Ist $\alpha + n$ ganz, ist eine direkte Ziehung die Summe von $\alpha + n$ Exponentialzahlen: $\lambda = -\ln(U_1 \cdots U_{\alpha + n})/(\beta + T)$ mit gleichverteilten $U_i$.

**Nichtkonjugater Prior und Gitter.** Mit $u = \ln \lambda$ und dem Prior $u \sim \mathcal N(\ln m, s_0^2)$ ist
$$\ln p(u \mid n) = n u - T e^{u} - \frac{(u - \ln m)^2}{2 s_0^2} + \text{const.}$$
Das Gitter nimmt $G$ gleich verteilte Punkte $u_i$ auf dem Kasten Gipfel $\pm 6$ Laplace-Standardabweichungen und bildet Momente mit der Trapezregel; jeder Punkt ist eine Dichteauswertung.

**Mehrere gekoppelte Raten.** Für $d$ Schichten mit Zählungen $n_j$ in der Dauer $T$ ist
$$\ln p(u) = \sum_{j=1}^{d} \big(n_j u_j - T e^{u_j}\big) - \tfrac12 (u - \mu_0)^{\top} \Sigma_0^{-1} (u - \mu_0) + \text{const}, \qquad (\Sigma_0)_{ij} = s_0^2\, \rho^{|i - j|}.$$
Das Tensorgitter hat $G^d$ Punkte, die Kette einen Schritt je Auswertung.

**Zufallsschritt-Metropolis.** Vorschlag $y = x + \sigma_p\, \xi$ mit $\xi \sim \mathcal N(0, I)$, Annahme mit Wahrscheinlichkeit $\min\{1, p(y)/p(x)\}$. Der Fehler des Mittels nach $N$ Schritten ist $\approx \mathrm{sd}/\sqrt{N \cdot \mathrm{ESS\ je\ Schritt}}$; für den Fehler $\delta$ (in sd) braucht es $N = 1/(\delta^2 \cdot \mathrm{ESS\ je\ Schritt})$ Schritte.

**Ring.** Belegungen eines Rings aus $n$ Plätzen (keine zwei benachbarten belegt, Gewicht $\lambda^k$ für $k$ belegte Plätze) haben $Z(\lambda) = \mathrm{Spur}(T_\lambda^{\,n})$ mit $T_\lambda = \begin{pmatrix} 1 & \lambda \\ 1 & 0 \end{pmatrix}$, also $Z(\lambda) = \mu_+^n + \mu_-^n$ mit $\mu_\pm = (1 \pm \sqrt{1 + 4\lambda})/2$.

**Was der Block bewusst nicht enthält:** die Gleichungen der Uniformisierung im Beweis (nur das Ergebnis $P = I + Q/\Lambda$), die Herleitung der Laplace-Näherung (Gipfel durch Newton-Verfahren auf $\ln p$, Standardabweichung aus der zweiten Ableitung) und die Schätzer der ESS aus einer einzelnen Kette; die App rechnet die exakte ESS aus der Matrix.

**Quellen** (so gelesen, wie genannt): Levin, Peres, Wilmer, *Markov Chains and Mixing Times*, 2. Aufl. (Totalvariation, Mischzeit, Detailed Balance); Asanjarani, Nazarathy, Taylor, *A Survey of Parameter and State Estimation in Queues*,
Queueing Systems 97 (2021), arXiv 2012.14614 (Abstract und Abschnitt 3.7: Gamma-Prior für die Poisson-Ankunftsrate gibt einen Gamma-Posterior); Aktekin, Soyer, *Bayesian Modeling of Queues in Call Centers*, GWU Technical Report TR-2010-3 (Abschnitte 1 und 4, Anhang:
unabhängige Gamma-Priors geben geschlossene Posteriors, bei einem Lognormal-Prior gibt es keine geschlossene Form). Die Formel für $\sigma^2$ ist Lehrbuchstoff der Statistik von Markov-Ketten; hier ist sie eigene Rechnung, gegen die Autokovarianzsumme geprüft (Quelle nicht gelesen).
"""
