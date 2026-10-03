"""Figuras Plotly de ATTN (Persona 4)."""
import numpy as np
import plotly.graph_objects as go

from engine import SIGMA, MASSES, SEMIAXES, NULL_ID

BG = "#0e1117"
STATUS_COLORS = {
    "MONITORING": "#4b5563",
    "ANOMALY DETECTED": "#d97706",
    "TESTING HYPOTHESIS": "#2563eb",
    "HYPOTHESIS FALSIFIED": "#dc2626",
    "HIDDEN CAUSE FOUND": "#7c3aed",
    "REPLICATION PASSED": "#16a34a",
    "REPLICATION FAILED": "#991b1b",
    "INCONCLUSIVE": "#6b7280",
}
KIND_STYLE = {
    "initial": ("Observaciones iniciales", "#60a5fa", "circle"),
    "selected": ("Experimentos elegidos", "#f59e0b", "diamond"),
    "replication": ("Replicación (holdout)", "#22c55e", "square"),
}


def _layout(fig, title, height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15)), height=height,
        paper_bgcolor=BG, plot_bgcolor=BG, template="plotly_dark",
        margin=dict(l=10, r=10, t=45, b=10),
        legend=dict(orientation="h", yanchor="bottom", y=-0.25, font=dict(size=10)),
    )
    return fig


def _focus_id(bank, posterior):
    """Hipótesis de cuerpo oculto con mayor probabilidad (para dibujar su 'firma')."""
    hidden = [(posterior[h], h) for h in bank.ids if h != NULL_ID]
    return max(hidden)[1]


def orbit_figure(bank, frame, observations):
    fig = go.Figure()
    nul = bank.traj[NULL_ID]
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Estrella",
                             marker=dict(symbol="star", size=16, color="gold")))
    fig.add_trace(go.Scatter(x=nul["v2"][:, 0], y=nul["v2"][:, 1], mode="lines",
                             name="Planeta visible 2 (modelo)", line=dict(color="#6b7280", width=1)))
    fig.add_trace(go.Scatter(x=nul["v1"][:, 0], y=nul["v1"][:, 1], mode="lines",
                             name="Planeta visible 1 (modelo sin oculto)",
                             line=dict(color="#9ca3af", width=1, dash="dash")))
    # El cuerpo oculto SOLO se dibuja como hipótesis una vez que es probable (nunca la verdad)
    if frame["p_hidden"] >= 0.5 and frame["map_id"] != NULL_ID:
        tr = bank.traj[frame["map_id"]]
        fig.add_trace(go.Scatter(x=tr["hid"][:, 0], y=tr["hid"][:, 1], mode="lines",
                                 name=f"Cuerpo oculto (hipótesis {frame['map_id']})",
                                 line=dict(color="#ef4444", width=2, dash="dot")))
    for kind, (label, color, symbol) in KIND_STYLE.items():
        pts = [o for o in observations if o.get("kind") == kind]
        if pts:
            fig.add_trace(go.Scatter(
                x=[o["x"] for o in pts], y=[o["y"] for o in pts], mode="markers", name=label,
                text=[f"t={o['time']:.2f} yr" for o in pts], hoverinfo="text+x+y",
                marker=dict(color=color, symbol=symbol, size=9, line=dict(width=1, color="white")),
            ))
    fig.update_xaxes(title="x (AU)", zeroline=False)
    fig.update_yaxes(title="y (AU)", scaleanchor="x", scaleratio=1, zeroline=False)
    return _layout(fig, "Sistema orbital (observaciones parciales)", 430)


def residual_figure(bank, frame, observations):
    """
    Muestra la anomalía: |observación − modelo sin oculto| / σ.
    Línea: firma esperada de la hipótesis de cuerpo oculto más probable.
    """
    fig = go.Figure()
    nul = bank.traj[NULL_ID]["v1"]
    focus = _focus_id(bank, frame["posterior"])
    sig = np.linalg.norm(bank.traj[focus]["v1"] - nul, axis=1) / SIGMA
    mask = bank.times <= 10.0
    fig.add_trace(go.Scatter(x=bank.times[mask], y=sig[mask], mode="lines",
                             name=f"Firma esperada de {focus}",
                             line=dict(color="#ef4444", width=2)))
    fig.add_hline(y=3, line=dict(color="#9ca3af", dash="dash"),
                  annotation_text="3σ", annotation_position="top left")
    for kind, (label, color, symbol) in KIND_STYLE.items():
        pts = [o for o in observations if o.get("kind") == kind]
        if pts:
            ys = [float(np.linalg.norm(np.array([o["x"], o["y"]]) - nul[bank.idx(o["time"])]) / SIGMA)
                  for o in pts]
            fig.add_trace(go.Scatter(x=[o["time"] for o in pts], y=ys, mode="markers", name=label,
                                     marker=dict(color=color, symbol=symbol, size=9,
                                                 line=dict(width=1, color="white"))))
    exp = frame.get("experiment")
    if exp:
        fig.add_vline(x=exp["time"], line=dict(color="#f59e0b", dash="dot"))
    fig.update_xaxes(title="tiempo (años)")
    fig.update_yaxes(title="|obs − modelo sin oculto| / σ")
    return _layout(fig, "Anomalía: discrepancia con el modelo sin cuerpo oculto", 340)


def posterior_figure(frame, truth_cell=None):
    """Heatmap de probabilidad posterior sobre (masa, semieje) + p(sin oculto)."""
    z = np.zeros((len(MASSES), len(SEMIAXES)))
    for hid, p in frame["posterior"].items():
        if hid == NULL_ID:
            continue
        # id = H_m{m}_a{a:.2f}
        m_s, a_s = hid[3:].split("_a")
        i = int(np.argmin([abs(m * 1e5 - float(m_s)) for m in MASSES]))
        j = int(np.argmin([abs(a - float(a_s)) for a in SEMIAXES]))
        z[i, j] = p
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{a:.2f}" for a in SEMIAXES], y=[f"{m * 1e5:.0f}e-5" for m in MASSES],
        colorscale="Viridis", zmin=0, zmax=max(0.05, float(z.max())),
        text=[[f"{v:.2f}" for v in row] for row in z], texttemplate="%{text}",
        colorbar=dict(title="p"),
    ))
    if truth_cell is not None:  # solo al final de la corrida (evaluación)
        fig.add_trace(go.Scatter(x=[f"{SEMIAXES[truth_cell[1]]:.2f}"],
                                 y=[f"{MASSES[truth_cell[0]] * 1e5:.0f}e-5"], mode="markers",
                                 marker=dict(symbol="x", size=16, color="red"), name="verdad"))
    fig.update_xaxes(title="semieje a (AU)")
    fig.update_yaxes(title="masa (M☉)")
    p0 = frame["posterior"][NULL_ID]
    return _layout(fig, f"Hipótesis activas — P(sin cuerpo oculto) = {p0:.1%}", 330)


def belief_figure(frames):
    steps = [f for f in frames if f["type"] == "step"]
    fig = go.Figure()
    xs = [f["n_observations"] for f in steps]
    fig.add_trace(go.Scatter(x=xs, y=[f["p_hidden"] for f in steps], mode="lines+markers",
                             name="P(existe cuerpo oculto)", line=dict(color="#a78bfa")))
    fig.add_trace(go.Scatter(x=xs, y=[f["cluster_mass"] for f in steps], mode="lines+markers",
                             name="masa en torno a la hipótesis MAP", line=dict(color="#34d399")))
    fig.add_hline(y=0.99, line=dict(color="#9ca3af", dash="dash"), annotation_text="umbral")
    fig.update_xaxes(title="experimentos realizados", dtick=1)
    fig.update_yaxes(title="probabilidad", range=[0, 1.02])
    return _layout(fig, "Cambio de confianza", 300)


def scores_figure(frame):
    sc = frame.get("candidate_scores")
    if not sc:
        return None
    t = [s[0] for s in sc]
    v = [s[1] for s in sc]
    chosen = frame["experiment"]["time"]
    colors = ["#f59e0b" if abs(ti - chosen) < 1e-9 else "#4b5563" for ti in t]
    fig = go.Figure(go.Bar(x=[f"{ti:.2f}" for ti in t], y=v, marker_color=colors))
    fig.update_xaxes(title="tiempo candidato (años)")
    fig.update_yaxes(title="desacuerdo entre hipótesis (σ²)")
    return _layout(fig, "Dónde discrepan más las hipótesis (elegido en naranja)", 280)
