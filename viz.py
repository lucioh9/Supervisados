"""LYDH Plotly figures — mission-control style, plain-English labels."""
import numpy as np
import plotly.graph_objects as go

from engine import SIGMA, MASSES, SEMIAXES, NULL_ID

BG = "rgba(0,0,0,0)"
TXT = "#e8ecf8"
MUTED = "#8b95b8"
GRID = "rgba(139,149,184,0.12)"
CYAN, VIOLET, AMBER, GREEN, RED, PINK = "#22d3ee", "#8b5cf6", "#fbbf24", "#34d399", "#fb7185", "#f472b6"

STATUS_COLORS = {
    "MONITORING": "#64748b",
    "ANOMALY DETECTED": "#f59e0b",
    "TESTING HYPOTHESIS": "#3b82f6",
    "HYPOTHESIS FALSIFIED": "#ef4444",
    "HIDDEN CAUSE FOUND": "#8b5cf6",
    "REPLICATION PASSED": "#10b981",
    "REPLICATION FAILED": "#be123c",
    "INCONCLUSIVE": "#6b7280",
}
STATUS_LABELS = {
    "MONITORING": "WATCHING",
    "ANOMALY DETECTED": "SOMETHING DOESN'T ADD UP",
    "TESTING HYPOTHESIS": "TESTING AN IDEA",
    "HYPOTHESIS FALSIFIED": "IDEA RULED OUT",
    "HIDDEN CAUSE FOUND": "HIDDEN PLANET FOUND",
    "REPLICATION PASSED": "CONFIRMED ON NEW DATA",
    "REPLICATION FAILED": "FAILED THE DOUBLE-CHECK",
    "INCONCLUSIVE": "NOT ENOUGH EVIDENCE",
}
POLICY_NAMES = {"attn": "LYDH (smart pick)", "greedy": "Biggest error", "random": "Random"}
POLICY_HELP = {
    "attn": "Measures where our competing guesses disagree the most.",
    "greedy": "Measures where the current picture is most wrong.",
    "random": "Measures on a random date.",
}
KIND_STYLE = {
    "initial": ("First measurements", CYAN, "circle"),
    "selected": ("Measurements LYDH chose", AMBER, "diamond"),
    "replication": ("New data (kept hidden until the end)", GREEN, "square"),
}


def _layout(fig, title, height=420):
    fig.update_layout(
        title=dict(text=title, font=dict(size=14, color=TXT), x=0.01, xanchor="left"),
        height=height, paper_bgcolor=BG, plot_bgcolor=BG,
        font=dict(color=TXT, family="Inter, sans-serif", size=12),
        margin=dict(l=8, r=8, t=48, b=8),
        legend=dict(orientation="h", yanchor="top", y=-0.18, x=0, font=dict(size=10, color=MUTED),
                    bgcolor="rgba(0,0,0,0)"),
        hoverlabel=dict(bgcolor="#0e1426", font_color=TXT),
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(color=MUTED, size=11),
                     tickfont=dict(color=MUTED, size=10))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(color=MUTED, size=11),
                     tickfont=dict(color=MUTED, size=10))
    return fig


def _focus_id(bank, posterior):
    hidden = [(posterior[h], h) for h in bank.ids if h != NULL_ID]
    return max(hidden)[1]


def _glow_line(fig, x, y, color, name, width=2, dash=None, showlegend=True):
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", line=dict(color=color, width=width * 5),
                             opacity=0.12, hoverinfo="skip", showlegend=False))
    fig.add_trace(go.Scatter(x=x, y=y, mode="lines", name=name, showlegend=showlegend,
                             line=dict(color=color, width=width, dash=dash)))


def orbit_figure(bank, frame, observations):
    fig = go.Figure()
    nul = bank.traj[NULL_ID]
    _glow_line(fig, nul["v2"][:, 0], nul["v2"][:, 1], "#64748b", "Visible planet 2", 1)
    _glow_line(fig, nul["v1"][:, 0], nul["v1"][:, 1], CYAN, "Visible planet 1 (where it should be with no hidden planet)", 1.5, "dash")
    # El cuerpo oculto SOLO se dibuja como hipótesis una vez que es probable (nunca la verdad)
    if frame["p_hidden"] >= 0.5 and frame["map_id"] != NULL_ID:
        tr = bank.traj[frame["map_id"]]
        _glow_line(fig, tr["hid"][:, 0], tr["hid"][:, 1], RED, "Hidden planet (our best guess)", 2.2, "dot")
    fig.add_trace(go.Scatter(x=[0], y=[0], mode="markers", name="Star", hoverinfo="name",
                             marker=dict(symbol="star", size=20, color="#fde047",
                                         line=dict(width=6, color="rgba(253,224,71,0.25)"))))
    for kind, (label, color, symbol) in KIND_STYLE.items():
        pts = [o for o in observations if o.get("kind") == kind]
        if pts:
            fig.add_trace(go.Scatter(
                x=[o["x"] for o in pts], y=[o["y"] for o in pts], mode="markers", name=label,
                text=[f"t={o['time']:.2f} yr" for o in pts], hoverinfo="text+x+y",
                marker=dict(color=color, symbol=symbol, size=10, line=dict(width=1.5, color="white")),
            ))
    fig.update_xaxes(title="x (AU)", zeroline=False, showgrid=False)
    fig.update_yaxes(title="y (AU)", scaleanchor="x", scaleratio=1, zeroline=False, showgrid=False)
    return _layout(fig, "THE SOLAR SYSTEM · what we can see", 440)


def residual_figure(bank, frame, observations):
    fig = go.Figure()
    nul = bank.traj[NULL_ID]["v1"]
    focus = _focus_id(bank, frame["posterior"])
    sig = np.linalg.norm(bank.traj[focus]["v1"] - nul, axis=1) / SIGMA
    mask = bank.times <= 10.0
    fig.add_trace(go.Scatter(x=bank.times[mask], y=sig[mask], mode="lines", name="What a hidden planet would cause",
                             line=dict(color=RED, width=2), fill="tozeroy", fillcolor="rgba(251,113,133,0.08)"))
    fig.add_hline(y=3, line=dict(color=MUTED, dash="dash", width=1), annotation_text="clearly more than normal error",
                  annotation_position="top left", annotation_font_color=MUTED)
    for kind, (label, color, symbol) in KIND_STYLE.items():
        pts = [o for o in observations if o.get("kind") == kind]
        if pts:
            ys = [float(np.linalg.norm(np.array([o["x"], o["y"]]) - nul[bank.idx(o["time"])]) / SIGMA) for o in pts]
            fig.add_trace(go.Scatter(x=[o["time"] for o in pts], y=ys, mode="markers", name=label,
                                     marker=dict(color=color, symbol=symbol, size=10,
                                                 line=dict(width=1.5, color="white"))))
    exp = frame.get("experiment")
    if exp:
        fig.add_vline(x=exp["time"], line=dict(color=AMBER, dash="dot", width=1.5))
    fig.update_xaxes(title="time (years)")
    fig.update_yaxes(title="how far off (× normal measurement error)")
    return _layout(fig, "WHAT DOESN'T ADD UP · gap between what we measured and what we expected", 340)


def posterior_figure(frame, truth_cell=None):
    z = np.zeros((len(MASSES), len(SEMIAXES)))
    for hid, p in frame["posterior"].items():
        if hid == NULL_ID:
            continue
        m_s, a_s = hid[3:].split("_a")
        i = int(np.argmin([abs(m * 1e5 - float(m_s)) for m in MASSES]))
        j = int(np.argmin([abs(a - float(a_s)) for a in SEMIAXES]))
        z[i, j] = p
    n = len(MASSES)
    show_text = n <= 8
    fig = go.Figure(go.Heatmap(
        z=z, x=[f"{a:.2f}" for a in SEMIAXES], y=[f"{m * 1e5:.1f}e-5" for m in MASSES],
        colorscale=[[0, "#0b1020"], [0.25, "#2b1d6b"], [0.55, "#7c3aed"], [0.8, "#22d3ee"], [1, "#fef08a"]],
        zmin=0, zmax=max(0.05, float(z.max())), xgap=2, ygap=2,
        text=[[f"{v:.2f}" for v in row] for row in z] if show_text else None,
        texttemplate="%{text}" if show_text else None, textfont=dict(color="white", size=11),
        colorbar=dict(title=dict(text="p", font=dict(color=MUTED)), tickfont=dict(color=MUTED), thickness=10),
    ))
    if truth_cell is not None:
        fig.add_trace(go.Scatter(x=[f"{SEMIAXES[truth_cell[1]]:.2f}"], y=[f"{MASSES[truth_cell[0]] * 1e5:.1f}e-5"],
                                 mode="markers", name="real answer (used only for grading)",
                                 marker=dict(symbol="x", size=16, color=RED, line=dict(width=2, color="white"))))
    fig.update_xaxes(title="distance from the star (AU)", showgrid=False)
    fig.update_yaxes(title="planet mass (× Sun)", showgrid=False)
    p0 = frame["posterior"][NULL_ID]
    return _layout(fig, f"BEST GUESSES · chance there is NO hidden planet: {p0:.0%}", 340)


def belief_figure(frames):
    steps = [f for f in frames if f["type"] == "step"]
    fig = go.Figure()
    xs = [f["n_observations"] for f in steps]
    fig.add_trace(go.Scatter(x=xs, y=[f["p_hidden"] for f in steps], mode="lines+markers", name="Chance a hidden planet exists",
                             line=dict(color=VIOLET, width=3, shape="spline"), marker=dict(size=8),
                             fill="tozeroy", fillcolor="rgba(139,92,246,0.10)"))
    fig.add_trace(go.Scatter(x=xs, y=[f["cluster_mass"] for f in steps], mode="lines+markers",
                             name="How much we agree on one answer", line=dict(color=GREEN, width=3, shape="spline"),
                             marker=dict(size=8)))
    fig.add_hline(y=0.99, line=dict(color=MUTED, dash="dash", width=1), annotation_text="sure enough to decide",
                  annotation_font_color=MUTED)
    fig.update_xaxes(title="measurements taken", dtick=1)
    fig.update_yaxes(title="how sure we are", range=[0, 1.04])
    return _layout(fig, "HOW SURE ARE WE? · updated after every measurement", 300)


def scores_figure(frame):
    sc = frame.get("candidate_scores")
    if not sc:
        return None
    t = [s[0] for s in sc]
    v = [s[1] for s in sc]
    chosen = frame["experiment"]["time"]
    colors = [AMBER if abs(ti - chosen) < 1e-9 else "rgba(100,116,139,0.45)" for ti in t]
    fig = go.Figure(go.Bar(x=[f"{ti:.1f}" for ti in t], y=v, marker_color=colors))
    fig.update_xaxes(title="possible measurement date (years)", showgrid=False)
    fig.update_yaxes(title="how much our guesses disagree")
    return _layout(fig, "WHERE TO MEASURE NEXT · our guesses disagree most at the amber bar", 260)
