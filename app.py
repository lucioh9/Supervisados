"""
LYDH — Discover what you can't see   |   UI
Ejecutar:  streamlit run app.py
"""
import html
import json
import time
from pathlib import Path

import pandas as pd
import streamlit as st

import engine as eng
import viz

st.set_page_config(page_title="LYDH — Discover what you can't see", page_icon="🔭", layout="wide")

BACKENDS = {"Math engine only (no AI agents)": eng.run_discovery}
_omni = eng.load_omnigent_backend()
if _omni is not None:
    BACKENDS = {"Team of AI agents (Omnigent)": _omni, **BACKENDS}

AGENT_COLORS = {"observer": "#38bdf8", "hypothesis": "#a78bfa", "planner": "#fbbf24", "falsifier": "#fbbf24",
                "scientist": "#34d399", "scout": "#f472b6", "supervisor": "#e8ecf8"}
STAGES = ["Something's off", "Make guesses", "Measure", "Rule out", "Planet found", "Double-check"]
STAGE_OF = {"MONITORING": 0, "ANOMALY DETECTED": 1, "TESTING HYPOTHESIS": 2, "HYPOTHESIS FALSIFIED": 3,
            "HIDDEN CAUSE FOUND": 4, "REPLICATION PASSED": 5, "REPLICATION FAILED": 5, "INCONCLUSIVE": 3}

CSS = """
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Inter:wght@400;500;600&display=swap');
html, body, [class*="css"] { font-family: 'Inter', sans-serif; }
.stApp {
  background:
    radial-gradient(1100px 600px at 12% -10%, rgba(124,92,255,.28), transparent 60%),
    radial-gradient(900px 500px at 95% 0%, rgba(34,211,238,.18), transparent 60%),
    radial-gradient(800px 600px at 50% 110%, rgba(244,114,182,.12), transparent 60%),
    #070a14;
}
header[data-testid="stHeader"] { background: transparent; }
.block-container { padding-top: 1.4rem; max-width: 1500px; }
[data-testid="stSidebar"] { background: rgba(14,20,38,.75); backdrop-filter: blur(14px); border-right: 1px solid rgba(255,255,255,.06); }
.hero h1 { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 3.1rem; line-height: 1.05; margin: 0;
  background: linear-gradient(92deg, #ffffff 0%, #a5b4fc 35%, #22d3ee 70%, #f472b6 100%); -webkit-background-clip: text;
  background-clip: text; color: transparent; letter-spacing: -.02em; }
.hero p { color: #8b95b8; margin: .5rem 0 .9rem 0; font-size: 1.02rem; }
.chip { display: inline-block; padding: 4px 12px; margin: 0 8px 6px 0; border-radius: 999px; font-size: .74rem; font-weight: 600;
  letter-spacing: .04em; border: 1px solid rgba(255,255,255,.12); background: rgba(255,255,255,.04); color: #cbd5f5; }
.chip.live { border-color: rgba(52,211,153,.5); color: #6ee7b7; }
.chip.live:before { content: ""; display: inline-block; width: 7px; height: 7px; border-radius: 50%; background: #34d399;
  margin-right: 7px; box-shadow: 0 0 10px #34d399; animation: pulse 1.6s infinite; }
@keyframes pulse { 0% { opacity: 1; } 50% { opacity: .35; } 100% { opacity: 1; } }
.glass { background: linear-gradient(145deg, rgba(255,255,255,.07), rgba(255,255,255,.02)); border: 1px solid rgba(255,255,255,.09);
  border-radius: 18px; padding: 16px 18px; backdrop-filter: blur(10px); box-shadow: 0 10px 40px rgba(0,0,0,.35); }
.proof { display: flex; gap: 14px; flex-wrap: wrap; margin: 6px 0 18px 0; }
.proof .glass { flex: 1; min-width: 210px; }
.big { font-family: 'Space Grotesk', sans-serif; font-size: 2.3rem; font-weight: 700; line-height: 1.05; }
.lbl { font-size: .72rem; letter-spacing: .12em; text-transform: uppercase; color: #8b95b8; margin-bottom: 6px; }
.sub { font-size: .8rem; color: #8b95b8; margin-top: 4px; }
.stepper { display: flex; gap: 6px; margin: 4px 0 14px 0; }
.stg { flex: 1; text-align: center; padding: 9px 4px; border-radius: 12px; font-size: .74rem; font-weight: 600; letter-spacing: .05em;
  text-transform: uppercase; color: #566087; background: rgba(255,255,255,.03); border: 1px solid rgba(255,255,255,.05); }
.stg.done { color: #a5b4fc; background: rgba(124,92,255,.12); border-color: rgba(124,92,255,.35); }
.stg.now { color: #fff; background: linear-gradient(90deg, #7c5cff, #22d3ee); border-color: transparent; box-shadow: 0 0 24px rgba(124,92,255,.65); }
.pill { display: flex; align-items: center; gap: 16px; padding: 16px 22px; border-radius: 18px; margin-bottom: 12px; }
.pill .t { font-family: 'Space Grotesk', sans-serif; font-weight: 700; font-size: 1.7rem; letter-spacing: .04em; color: #fff; }
.pill .s { color: rgba(255,255,255,.82); font-size: .9rem; }
.dot { width: 14px; height: 14px; border-radius: 50%; background: #fff; animation: pulse 1.4s infinite; flex: none; }
.feed { max-height: 360px; overflow-y: auto; padding-right: 6px; }
.msg { border-left: 3px solid; padding: 8px 12px; margin: 0 0 8px 0; background: rgba(255,255,255,.03); border-radius: 0 12px 12px 0; font-size: .86rem; color: #dbe2fb; }
.msg b { text-transform: uppercase; font-size: .72rem; letter-spacing: .1em; }
.msg i { float: right; font-style: normal; font-size: .65rem; color: #8b95b8; border: 1px solid rgba(255,255,255,.12); border-radius: 999px; padding: 0 8px; }
.tl { position: relative; margin-left: 10px; border-left: 2px solid rgba(139,149,184,.25); padding-left: 22px; }
.tl .row { position: relative; margin-bottom: 14px; }
.tl .row:before { content: ""; position: absolute; left: -29px; top: 6px; width: 12px; height: 12px; border-radius: 50%; background: var(--c); box-shadow: 0 0 12px var(--c); }
.tl .st { font-weight: 700; font-size: .78rem; letter-spacing: .08em; color: var(--c); }
.tl .bd { color: #cbd5f5; font-size: .86rem; }
.tl .mt { color: #8b95b8; font-size: .76rem; }
div.stButton > button[kind="primary"] { background: linear-gradient(90deg, #7c5cff, #22d3ee); border: none; font-weight: 700; letter-spacing: .04em;
  box-shadow: 0 0 26px rgba(124,92,255,.55); }
div.stButton > button:hover { filter: brightness(1.12); }
[data-testid="stTabs"] button { font-weight: 600; }
</style>
"""
st.markdown(CSS, unsafe_allow_html=True)


@st.cache_resource(show_spinner=False)
def load_bank():
    return eng.get_bank()


def _uid():
    st.session_state["_uid"] = st.session_state.get("_uid", 0) + 1
    return st.session_state["_uid"]


def _bench():
    p = Path("runs/benchmark_summary.json")
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


# ------------------------------------------------------------------ componentes visuales
def hero(backend_label):
    st.markdown(
        "<div class='hero'><h1>LYDH</h1>"
        "<p>Discover what you can't see. LYDH looks for causes hidden from view: it decides <b>what to measure next</b>, "
        "tries to <b>prove its own ideas wrong</b>, and only trusts what <b>still holds on new data</b>.</p>"
        f"<span class='chip live'>{html.escape(backend_label)}</span>"
        "<span class='chip'>AI suggests ideas</span><span class='chip'>Code does the math</span>"
        "<span class='chip'>A simulator checks</span><span class='chip'>New data confirms</span></div>",
        unsafe_allow_html=True)


def proof_strip():
    b = _bench()
    if not b:
        return
    r = b["ratios"]["random_over_attn"]
    hard = max(b["by_scenario"].values(), key=lambda v: v["random_over_attn"])["random_over_attn"]
    n_runs = b["n_seeds_per_scenario"] * b["n_scenarios"] * 3
    st.markdown(
        "<div class='proof'>"
        f"<div class='glass'><div class='lbl'>fewer measurements than random picking</div><div class='big' style='color:#22d3ee'>{r['ratio']:.2f}×</div>"
        f"<div class='sub'>average over all tests (range {r['ci95'][0]:.2f}–{r['ci95'][1]:.2f})</div></div>"
        f"<div class='glass'><div class='lbl'>in the hardest test case</div><div class='big' style='color:#f472b6'>{hard:.2f}×</div>"
        f"<div class='sub'>the harder the problem, the bigger the gain</div></div>"
        f"<div class='glass'><div class='lbl'>test runs behind these numbers</div><div class='big' style='color:#34d399'>{n_runs}</div>"
        f"<div class='sub'>{b['n_scenarios']} cases × {b['n_seeds_per_scenario']} repeats × 3 methods</div></div></div>",
        unsafe_allow_html=True)


def stepper(status):
    cur = STAGE_OF.get(status, 0)
    cells = ""
    for i, name in enumerate(STAGES):
        cls = "now" if i == cur else ("done" if i < cur else "")
        cells += f"<div class='stg {cls}'>{name}</div>"
    st.markdown(f"<div class='stepper'>{cells}</div>", unsafe_allow_html=True)


def status_pill(status, subtitle=""):
    c = viz.STATUS_COLORS.get(status, "#475569")
    status = viz.STATUS_LABELS.get(status, status)
    st.markdown(
        f"<div class='pill' style='background:linear-gradient(90deg,{c},{c}99);box-shadow:0 0 40px {c}66'>"
        f"<div class='dot'></div><div><div class='t'>{status}</div><div class='s'>{html.escape(subtitle)}</div></div></div>",
        unsafe_allow_html=True)


def kpi(col, label, value, sub="", color="#e8ecf8"):
    col.markdown(f"<div class='glass'><div class='lbl'>{label}</div><div class='big' style='color:{color}'>{value}</div>"
                 f"<div class='sub'>{sub}</div></div>", unsafe_allow_html=True)


def observations_upto(record, frames, k):
    obs = [dict(o, kind="initial") for o in record["initial_observations"]]
    for f in frames[:k + 1]:
        if f["type"] == "step" and f.get("observation"):
            obs.append(dict(f["observation"], kind="selected"))
        elif f["type"] == "replication":
            obs += [{"time": p["time"], "x": p["obs"][0], "y": p["obs"][1], "kind": "replication"}
                    for p in f["points"]]
    return obs


def render_frame(bank, record, k, final_view=False):
    frames = eng.build_frames(record)
    f = frames[k]
    obs = observations_upto(record, frames, k)
    meta = record["meta"]
    uid = _uid()

    stepper(f["status"])
    status_pill(f["status"], f"Test case {meta['scenario_index']} · method: {viz.POLICY_NAMES.get(meta['policy'], meta['policy'])} · step {f['step']}")

    c1, c2, c3, c4 = st.columns(4)
    prev = frames[k - 1]["p_hidden"] if k > 0 else None
    delta = "" if prev is None or abs(f["p_hidden"] - prev) < 5e-4 else f"{(f['p_hidden'] - prev) * 100:+.0f} points since last step"
    kpi(c1, "Measurements taken", f["n_observations"], "dates picked by the method", "#fbbf24")
    kpi(c2, "Chance a hidden planet exists", f"{f['p_hidden']:.0%}", delta, "#a78bfa")
    kpi(c3, "Guesses ruled out", f"{f['n_falsified']}<span style='font-size:1.1rem;color:#8b95b8'> / {len(record['hypotheses'])}</span>",
        "proven wrong by the data", "#fb7185")
    if f["type"] == "replication":
        ok = f["passed"]
        kpi(c4, "Double-check on new data", "PASSED" if ok else "FAILED", "measurements never used to find the planet",
            "#34d399" if ok else "#fb7185")
    else:
        kpi(c4, "Double-check on new data", "not yet", "required before we accept the answer", "#8b95b8")
    st.write("")

    left, right = st.columns([3, 2])
    with left:
        st.plotly_chart(viz.orbit_figure(bank, f, obs), use_container_width=True, key=f"orb{uid}")
        st.plotly_chart(viz.residual_figure(bank, f, obs), use_container_width=True, key=f"res{uid}")
    with right:
        truth_cell = None
        if final_view and record.get("evaluation"):
            truth_cell = record["evaluation"]["truth_grid_index"]
        st.plotly_chart(viz.posterior_figure(f, truth_cell), use_container_width=True, key=f"post{uid}")
        _experiment_panel(f)
        sf = viz.scores_figure(f)
        if sf is not None:
            st.plotly_chart(sf, use_container_width=True, key=f"sc{uid}")

    st.plotly_chart(viz.belief_figure(frames[:k + 1]), use_container_width=True, key=f"bel{uid}")
    a, b = st.columns([1, 1])
    with a:
        _timeline(frames[:k + 1])
    with b:
        _agents(frames[:k + 1])
    if final_view and record.get("evaluation"):
        _evaluation(record)


def _experiment_panel(f):
    if f["type"] == "replication":
        ok = f["passed"]
        c = "#34d399" if ok else "#fb7185"
        verdict = ("Our answer explains the new data. \"No hidden planet\" does not." if ok
                   else "Our answer does NOT hold up on the new data, so we reject it.")
        st.markdown(
            f"<div class='glass' style='border-color:{c}88'><div class='lbl'>Double-check on new data</div>"
            f"<div style='font-size:.95rem'>We took {len(f['points'])} extra measurements that were kept hidden until now. {verdict}</div>"
            f"<div class='sub'>{html.escape(f['next_decision'])}</div></div>", unsafe_allow_html=True)
        return
    if f["experiment"] is None:
        a = f["anomaly"]
        msg = ("The picture with no hidden planet does not match our first measurements."
               if a["significant"] else "The picture with no hidden planet matches our first measurements so far.")
        st.markdown(
            f"<div class='glass'><div class='lbl'>Observer</div><div style='font-size:.95rem'>{msg}</div>"
            f"<div class='sub'>{html.escape(f['next_decision'])}</div></div>", unsafe_allow_html=True)
        return
    e, r = f["experiment"], f["obs_vs_models"]
    st.markdown(
        f"<div class='glass'><div class='lbl'>Next measurement</div>"
        f"<div style='font-size:1.25rem;font-weight:700;color:#fbbf24'>Measure at year {e['time']:.1f}</div>"
        f"<div style='font-size:.86rem;margin:6px 0;color:#dbe2fb'>{html.escape(e['reason'])}</div>"
        f"<div class='sub'>Result: with no hidden planet we were off by <b style='color:#fb7185'>{r['null_sigma']:.1f}×</b> the normal error; "
        f"our best guess was off by <b style='color:#34d399'>{r['map_sigma']:.1f}×</b>. "
        f"{len(f['newly_falsified'])} {'guess' if len(f['newly_falsified']) == 1 else 'guesses'} ruled out this step.</div></div>", unsafe_allow_html=True)


def _timeline(frames):
    rows = ""
    for f in frames:
        c = viz.STATUS_COLORS.get(f["status"], "#64748b")
        exp = f.get("experiment")
        t = f"measured at year {exp['time']:.1f}" if exp else "—"
        pb = f.get("p_hidden_before")
        pb = "—" if pb is None else f"{pb:.0%}"
        rows += (f"<div class='row' style='--c:{c}'><div class='st'>{f['step']} · {viz.STATUS_LABELS.get(f['status'], f['status'])}</div>"
                 f"<div class='bd'>{html.escape(f['next_decision'])}</div>"
                 f"<div class='mt'>{t} · chance of hidden planet {pb} → {f['p_hidden']:.0%}</div></div>")
    st.markdown("<div class='lbl'>Step-by-step story</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='glass'><div class='tl'>{rows}</div></div>", unsafe_allow_html=True)


def _agents(frames):
    msgs = [(f["step"], m) for f in frames for m in f.get("agent_messages", [])]
    items = ""
    for step, m in msgs:
        c = AGENT_COLORS.get(m["agent"], "#94a3b8")
        tag = "code" if m.get("source") == "code" else "AI agent"
        items += (f"<div class='msg' style='border-color:{c}'><i>{tag}</i><b style='color:{c}'>{html.escape(m['agent'])}</b>"
                  f" · step {step}<br>{html.escape(m['message'])}</div>")
    st.markdown(f"<div class='lbl'>What the agents tell each other · {len(msgs)} messages</div>", unsafe_allow_html=True)
    st.markdown(f"<div class='glass feed'>{items}</div>", unsafe_allow_html=True)


def _evaluation(record):
    ev, out = record["evaluation"], record["outcome"]
    t = ev["truth"]
    ok = ev["correct"]
    c = "#34d399" if ok else "#fb7185"
    st.markdown(
        f"<div class='glass' style='margin-top:14px;border-color:{c}88'><div class='lbl'>Grading · the real answer (the AI never saw it)</div>"
        f"<div style='font-size:1rem'>Real hidden planet: {t['hidden_a']:.2f} AU from the star, mass {t['hidden_mass']:.1e} Suns. "
        f"Our answer: {eng.hname(ev['map_id'])} · "
        f"<b style='color:{c}'>{'CORRECT' if ok else 'WRONG'}</b> · found with <b>{out['n_observations']}</b> measurements.</div>"
        f"<div class='sub'>One run proves nothing on its own. The numbers at the top come from hundreds of repeated runs.</div></div>",
        unsafe_allow_html=True)


# ------------------------------------------------------------------ sidebar
st.sidebar.markdown("### 🔭 LYDH")
st.sidebar.caption("Looks where others don't. Tests its own ideas hard. Trusts only what holds up on new data.")
mode = st.sidebar.radio("What do you want to do?", ["Run a discovery", "Replay a saved run"])

backend_name = list(BACKENDS)[0]
if mode == "Run a discovery":
    backend_name = st.sidebar.selectbox("Who does the work?", list(BACKENDS))
    scenario = st.sidebar.selectbox(
        "Test case (where the hidden planet really is)", list(range(len(eng.env.SCENARIOS))), index=1,
        format_func=lambda i: f"Case {i}: planet {eng.env.SCENARIOS[i]['hidden_a']} AU from the star",
        help="Each case hides a planet of a different size and distance. The AI is never told the answer.")
    policy = st.sidebar.selectbox("How to pick the next measurement", ["attn", "greedy", "random"],
                                  format_func=lambda p: viz.POLICY_NAMES[p])
    st.sidebar.caption(viz.POLICY_HELP[policy])
    st.sidebar.caption("AU = the Earth–Sun distance. Planet mass is shown in units of 0.00001 Suns.")
    seed_txt = st.sidebar.text_input("Repeat number (optional)", "",
                                     help="Each number gives a different set of random measurement errors. Same number, same result. Leave empty for the default.")
    max_obs = st.sidebar.slider("Max measurements allowed", 3, 24, 24)
    delay = st.sidebar.slider("Pause between steps (seconds)", 0.0, 3.0, 1.0, 0.25)
    run_clicked = st.sidebar.button("▶ Run Discovery", type="primary", use_container_width=True)
else:
    run_clicked = False
    runs = eng.list_runs()
    if not runs:
        st.sidebar.warning("No saved runs yet. Run: python make_backup_run.py")
    else:
        choice = st.sidebar.selectbox("Saved run", runs, format_func=lambda p: p.name)
        if st.sidebar.button("Load", use_container_width=True):
            st.session_state["record"] = eng.load_record(choice)
            st.session_state["loaded_from_file"] = True

# ------------------------------------------------------------------ cuerpo
hero(backend_name if mode == "Run a discovery" else "Replay of a real saved run")
proof_strip()

tab_main, tab_cmp = st.tabs(["Discovery", "Compare the 3 methods"])

with tab_main:
    with st.spinner("Getting ready (first time takes up to a minute)..."):
        bank = load_bank()

    if run_clicked:
        seed = int(seed_txt) if seed_txt.strip() else None
        backend = BACKENDS[backend_name]
        holder = st.empty()
        record = None
        try:
            for ev in backend(scenario_index=scenario, policy=policy, max_obs=max_obs, seed=seed, bank=bank):
                eng.validate_event(ev)
                if ev["type"] == "header":
                    record = eng.new_record(ev)
                    continue
                eng.add_event(record, ev)
                if ev["type"] in ("step", "replication"):
                    frames = eng.build_frames(record)
                    with holder.container():
                        render_frame(bank, record, len(frames) - 1)
                    time.sleep(delay)
        except Exception as exc:
            st.error(f"'{backend_name}' failed: {exc}. Switch to the math engine or replay the saved run.")
            raise
        if record is not None and record["steps"]:
            path = eng.save_record(record)
            st.session_state["record"] = record
            st.session_state["last_saved"] = str(path)
            st.rerun()

    record = st.session_state.get("record")
    if record is None:
        st.info("Press **Run Discovery** or replay a saved run from the sidebar.")
    else:
        frames = eng.build_frames(record)
        top = st.columns([1, 4])
        replay = top[0].button("⏯ Play it back")
        k = top[1].slider("Go to step", 0, len(frames) - 1, len(frames) - 1, key="slider_k")
        if replay:
            holder = st.empty()
            for i in range(len(frames)):
                with holder.container():
                    render_frame(bank, record, i, final_view=(i == len(frames) - 1))
                time.sleep(1.0)
        else:
            render_frame(bank, record, k, final_view=(k == len(frames) - 1))
        st.download_button("Download the full step-by-step record (JSON)", json.dumps(record, ensure_ascii=False, indent=1),
                           file_name="attn_research_record.json", mime="application/json")

with tab_cmp:
    st.markdown("Quick test of the three ways to pick the next measurement. The official numbers come from `python bench.py 100`.")
    cc = st.columns(3)
    sc_cmp = cc[0].selectbox("Test case", list(range(len(eng.env.SCENARIOS))), index=3, key="sc_cmp")
    n_seeds = cc[1].slider("Repeats", 5, 50, 20, help="How many times each method is tried, each with different random measurement errors.")
    go_cmp = cc[2].button("Compare Random / Biggest error / LYDH")
    if go_cmp:
        with st.spinner("Trying each method..."):
            rows = eng.quick_compare(sc_cmp, ["random", "greedy", "attn"], list(range(n_seeds)), load_bank())
        df = pd.DataFrame(rows)
        summ = df.groupby("policy").agg(
            mean_obs=("n_observations", "mean"), correct=("correct", "mean"),
            replicated=("replicated", "mean")).round(2)
        shown = summ.rename(index=viz.POLICY_NAMES, columns={
            "mean_obs": "average measurements needed", "correct": "found the right planet (share of runs)",
            "replicated": "passed the double-check (share of runs)"})
        st.dataframe(shown, use_container_width=True)
        if "random" in summ.index and "attn" in summ.index:
            st.write(f"LYDH needed **{summ.loc['random', 'mean_obs'] / summ.loc['attn', 'mean_obs']:.2f}×** fewer measurements "
                     f"than random picking (average of {n_seeds} repeats, test case {sc_cmp}).")
        st.bar_chart(shown["average measurements needed"])
