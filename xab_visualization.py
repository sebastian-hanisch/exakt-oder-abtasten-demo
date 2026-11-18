"""Plotly-Abbildungen (alle Achsen fest: `fixedrange`, damit Touch-Scrollen nicht am Diagramm hängen bleibt)."""

from __future__ import annotations

import numpy as np
import plotly.graph_objects as go

import xab_constants as C
import xab_reference as REF

_MARGIN = dict(l=60, r=20, t=40, b=50)


def _fix(fig: go.Figure, height: int = 330, **layout) -> go.Figure:
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=height, margin=_MARGIN, legend=dict(orientation="h", y=-0.28), **layout)
    return fig


def build_gate_pi(pi: np.ndarray, title: str = "Gleichgewicht π (exakt)") -> go.Figure:
    n = list(range(len(pi)))
    colors = [C.COLOR_EXACT] * (len(pi) - 1) + [C.COLOR_CHAIN]
    fig = go.Figure(go.Bar(x=n, y=pi, marker_color=colors, hovertemplate="n = %{x}: π = %{y:.5f}<extra></extra>", name="π exakt"))
    fig.update_layout(title=title, xaxis_title="Zahl der Lkw im System n (rot: Zustand K, der Verlust)", yaxis_title="Wahrscheinlichkeit", showlegend=False)
    return _fix(fig)


def build_gate_shares(shares: np.ndarray, exact_loss: float, band: float, eps: float, t_steps: int) -> go.Figure:
    """Simulierter Zeitanteil im Zustand K je Kette gegen den exakten Wert, mit dem exakten 95-%-Band der Streuung und dem gewünschten ±ε."""
    r = len(shares)
    fig = go.Figure()
    fig.add_hrect(y0=exact_loss - band, y1=exact_loss + band, fillcolor="rgba(47,107,101,0.15)", line_width=0, annotation_text="Band ±1,96·√(σ²/T)", annotation_position="top left")
    fig.add_hline(y=exact_loss, line_color=C.COLOR_EXACT, line_width=2)
    for sign in (-1, 1):
        fig.add_hline(y=exact_loss + sign * eps, line_color=C.COLOR_OTHER, line_dash="dash")
    fig.add_trace(go.Scatter(x=list(range(1, r + 1)), y=shares, mode="markers", marker=dict(color=C.COLOR_SIM, size=8), name="Zeitanteil je Kette",
                             hovertemplate="Kette %{x}: %{y:.4f}<extra></extra>"))
    lo = min(shares.min(), exact_loss - max(band, eps)) - 0.002
    hi = max(shares.max(), exact_loss + max(band, eps)) + 0.002
    fig.update_layout(title=f"Simulierter Verlust je Kette nach T = {t_steps:,} Schritten".replace(",", " "), xaxis_title="Kette", yaxis_title="Zeitanteil im Zustand K",
                      yaxis_range=[lo, hi], xaxis_range=[0.5, r + 0.5], showlegend=False)
    return _fix(fig)


def build_gamma_density(post, draws: np.ndarray, trace: np.ndarray) -> go.Figure:
    xs = np.linspace(max(post.mean - 4.5 * post.sd, 1e-6), post.mean + 5 * post.sd, 300)
    fig = go.Figure()
    for data, name, color in ((draws, "direkte Ziehungen", C.COLOR_SIM), (trace, "Kette (eine Kette, alle Schritte)", C.COLOR_CHAIN)):
        fig.add_trace(go.Histogram(x=data, histnorm="probability density", name=name, marker_color=color, opacity=0.45, nbinsx=40, xbins=dict(start=float(xs[0]), end=float(xs[-1]))))
    fig.add_trace(go.Scatter(x=xs, y=post.dist.pdf(xs), mode="lines", name=f"exakt: Gamma({post.shape:g}, {post.rate:g})", line=dict(color=C.COLOR_EXACT, width=3)))
    fig.update_layout(barmode="overlay", title="Posterior der Ankunftsrate λ", xaxis_title="λ (Ankünfte je Abfertigungsdauer)", yaxis_title="Dichte", xaxis_range=[float(xs[0]), float(xs[-1])])
    return _fix(fig, 360)


def build_ess_steps(table: tuple, current_step: float, current_ess: float) -> go.Figure:
    steps = [r[0] for r in table]
    ess = [r[2] for r in table]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=steps, y=ess, mode="lines+markers", name="Kette (Tabelle)", line=dict(color=C.COLOR_CHAIN), hovertemplate="Schritt %{x} sd: %{y:.3f}<extra></extra>"))
    fig.add_trace(go.Scatter(x=[current_step], y=[current_ess], mode="markers", name="gewählter Schritt", marker=dict(color=C.COLOR_SIM, size=12, symbol="diamond")))
    fig.add_hline(y=1.0, line_color=C.COLOR_EXACT, line_dash="dash", annotation_text="direkte Ziehung: 1", annotation_position="bottom right")
    fig.update_layout(title="Unabhängige Ziehungen je Schritt (ESS) gegen Schrittweite", xaxis_title="Schrittweite (Vielfache der Posterior-sd)", yaxis_title="ESS je Schritt",
                      xaxis_type="log", yaxis_range=[0, 1.1])
    return _fix(fig)


def build_grid_density(run) -> go.Figure:
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=run.lam, y=run.dens, mode="lines", name="Posterior (feine Referenz)", line=dict(color=C.COLOR_EXACT, width=3)))
    if run.prior.kind == "lognormal":
        fig.add_trace(go.Scatter(x=run.lam, y=run.conj_dens, mode="lines", name="zum Vergleich: Gamma-Prior (konjugat)", line=dict(color=C.COLOR_OTHER, dash="dash")))
    fig.add_trace(go.Scatter(x=run.grid_lam, y=run.grid_dens, mode="markers", name=f"Auswertungspunkte des Gitters (G = {len(run.grid_lam)})",
                             marker=dict(color=C.COLOR_GRID, size=9, symbol="diamond"), hovertemplate="λ = %{x:.3f}: %{y:.3f}<extra></extra>"))
    fig.update_layout(title="Posterior der Rate und die Punkte, an denen das Gitter die Dichte auswertet", xaxis_title="λ", yaxis_title="Dichte")
    return _fix(fig, 360)


def build_grid_error(errs_by_g: tuple, current_g: int, current_err: float) -> go.Figure:
    gs = [g for g, _ in errs_by_g]
    worst = [max(max(e), 1e-12) for _, e in errs_by_g]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=gs, y=worst, mode="lines+markers", name="größter Fehler (Gitter 6, 8, 10, 12)", line=dict(color=C.COLOR_GRID), hovertemplate="G = %{x}: %{y:.2e} sd<extra></extra>"))
    fig.add_trace(go.Scatter(x=[current_g], y=[max(current_err, 1e-12)], mode="markers", name="gewähltes G", marker=dict(color=C.COLOR_SIM, size=12, symbol="diamond")))
    fig.add_hline(y=0.01, line_color=C.COLOR_OTHER, line_dash="dash", annotation_text="0,01 sd", annotation_position="top right")
    fig.update_layout(title="Fehler des Gitters gegen Gitterpunkte", xaxis_title="Gitterpunkte G", yaxis_title="Fehler (in sd)", yaxis_type="log")
    return _fix(fig)


def build_dim_cost(rows: tuple, accuracy_label: str) -> go.Figure:
    ds = [r[0] for r in rows]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=ds, y=[r[1] for r in rows], mode="lines+markers", name="Gitter: Dichteauswertungen", line=dict(color=C.COLOR_GRID, width=3),
                             hovertemplate="d = %{x}: %{y:.3g}<extra></extra>"))
    for idx, name, dash in ((2, "Kette, isotroper Schritt (Folge A)", "solid"), (3, "Kette, isotroper Schritt (Folge B)", "dot"), (4, "Kette, vorkonditionierter Schritt", "dash")):
        fig.add_trace(go.Scatter(x=ds, y=[r[idx] for r in rows], mode="lines+markers", name=name, line=dict(color=C.COLOR_CHAIN, dash=dash),
                                 hovertemplate="d = %{x}: %{y:.3g}<extra></extra>"))
    fig.update_layout(title=f"Kosten für {accuracy_label}: Gitter gegen Kette (vorgerechnet)", xaxis_title="Zahl der gekoppelten Raten d", yaxis_title="Dichteauswertungen",
                      yaxis_type="log", xaxis=dict(tickmode="linear", dtick=1))
    return _fix(fig, 360)


def build_dim_rmse(d: int) -> go.Figure:
    """Fehler des Kettenmittels gegen Schrittzahl (Doppellog), vorgerechnet, mit der Steigung −½ als Hilfslinie."""
    ch = REF.dimension()["d"][str(d)]["ketten"]
    fig = go.Figure()
    for name, label, color, dash in (("iso_a", "isotroper Schritt, Folge A", C.COLOR_CHAIN, "solid"), ("iso_b", "isotroper Schritt, Folge B", C.COLOR_CHAIN, "dot"),
                                      ("pre_a", "vorkonditionierter Schritt", C.COLOR_SIM, "dash")):
        ns = sorted(int(k) for k in ch[name]["rmse_in_sd"])
        fig.add_trace(go.Scatter(x=ns, y=[ch[name]["rmse_in_sd"][str(n)] for n in ns], mode="lines+markers", name=label, line=dict(color=color, dash=dash),
                                 hovertemplate="N = %{x}: %{y:.4f} sd<extra></extra>"))
    n0 = ns[0]
    y0 = ch["iso_a"]["rmse_in_sd"][str(n0)]
    fig.add_trace(go.Scatter(x=ns, y=[y0 * (n0 / n) ** 0.5 for n in ns], mode="lines", name="Steigung −½ (Wurzelgesetz)", line=dict(color=C.COLOR_OTHER, dash="dash")))
    fig.update_layout(title=f"Fehler des Kettenmittels gegen Schritte, d = {d} (vorgerechnet)", xaxis_title="Schritte N", yaxis_title="RMSE (in sd)", xaxis_type="log", yaxis_type="log")
    return _fix(fig)


def build_overview(gate, gp, grid, dim, s) -> go.Figure:
    """Kosten für die jeweilige Genauigkeit: ausgerechnet (Gleichungen, Zahlen, Dichteauswertungen) gegen abgetastet (Schritte), logarithmische Achse."""
    d = s.dim_d
    items = [
        (f"Gate-Verlust ±{s.eps_pp:g} Prozentpunkte<br>(Gleichungen | Schritte)", gate.exact.equations, gate.t_need),
        ("Rate, Gamma-Prior, 0,01 sd<br>(Zahlen | Schritte)", 2, gp.chain_steps_001),
        (f"Rate, anderer Prior, 0,01 sd<br>(Auswertungen G = {s.grid_g} | Schritte)", s.grid_g, grid.chain_steps_001),
        (f"{d} gekoppelte Raten, {s.dim_acc}<br>(Auswertungen G^d | Schritte)", dim.points, dim.rows[d - 1][2]),
    ]
    labels = [i[0] for i in items]
    fig = go.Figure()
    fig.add_trace(go.Bar(y=labels, x=[i[1] for i in items], orientation="h", name="ausgerechnet", marker_color=C.COLOR_EXACT, hovertemplate="%{x:,.0f}<extra>ausgerechnet</extra>"))
    fig.add_trace(go.Bar(y=labels, x=[i[2] for i in items], orientation="h", name="abgetastet (Kette)", marker_color=C.COLOR_CHAIN, hovertemplate="%{x:,.0f}<extra>abgetastet</extra>"))
    fig.update_layout(barmode="group", xaxis_type="log", xaxis_title="Kosten in der genannten Einheit (logarithmisch)", yaxis=dict(autorange="reversed"), title="Ausgerechnet gegen abgetastet")
    fig.update_layout(margin=dict(l=250, r=20, t=40, b=50))
    fig.update_xaxes(fixedrange=True)
    fig.update_yaxes(fixedrange=True)
    fig.update_layout(height=360, legend=dict(orientation="h", y=-0.25))
    return fig
