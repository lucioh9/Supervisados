"""
ATTN — Discover what you can't see   |   UI de Persona 4
Ejecutar:  streamlit run app.py
"""
import time

import pandas as pd
import streamlit as st

import engine as eng
import viz

st.set_page_config(page_title="ATTN — Discover what you can't see", page_icon="🔭", layout="wide")

BACKENDS = {"Motor local (matemático, sin LLM)": eng.run_discovery}
_omni = eng.load_omnigent_backend()
if _omni is not None:
    BACKENDS = {"Omnigent (agentes P2)": _omni, **BACKENDS}


@st.cache_resource(show_spinner=False)
def load_bank():
    return eng.get_bank()


def _uid():
    st.session_state["_uid"] = st.session_state.get("_uid", 0) + 1
    return st.session_state["_uid"]


# ------------------------------------------------------------------ helpers de render
def observations_upto(record, frames, k):
    obs = [dict(o, kind="initial") for o in record["initial_observations"]]
    for f in frames[:k + 1]:
        if f["type"] == "step" and f.get("observation"):
            obs.append(dict(f["observation"], kind="selected"))
        elif f["type"] == "replication":
            obs += [{"time": p["time"], "x": p["obs"][0], "y": p["obs"][1], "kind": "replication"}
                    for p in f["points"]]
    return obs


def banner(status, subtitle=""):
    color = viz.STATUS_COLORS.get(status, "#4b5563")
    st.markdown(
        f"<div style='background:{color};padding:14px 20px;border-radius:10px;'>"
        f"<span style='font-size:1.5rem;font-weight:700;letter-spacing:.04em;color:white'>{status}</span>"
        f"<span style='margin-left:16px;color:#f3f4f6'>{subtitle}</span></div>",
        unsafe_allow_html=True,
    )


def render_frame(bank, record, k, final_view=False):
    frames = eng.build_frames(record)
    f = frames[k]
    obs = observations_upto(record, frames, k)
    meta = record["meta"]
    uid = _uid()

    sub = f"escenario {meta['scenario_index']} · política {meta['policy']} · paso {f['step']}"
    banner(f["status"], sub)
    st.write("")

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Experimentos realizados", f["n_observations"])
    prev = frames[k - 1]["p_hidden"] if k > 0 else None
    c2.metric("P(cuerpo oculto)", f"{f['p_hidden']:.1%}",
              delta=None if prev is None else f"{(f['p_hidden'] - prev) * 100:+.1f} pp")
    c3.metric("Hipótesis falsificadas", f"{f['n_falsified']} / {len(record['hypotheses'])}")
    if f["type"] == "replication":
        c4.metric("Replicación", "PASADA ✅" if f["passed"] else "FALLIDA ❌")
    else:
        c4.metric("Replicación", "pendiente")

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
    _timeline(frames[:k + 1])
    _agents(frames[:k + 1])

    if final_view and record.get("evaluation"):
        _evaluation(record)


def _experiment_panel(f):
    st.markdown("#### Siguiente experimento y resultado")
    if f["type"] == "replication":
        st.success(f"Holdout ({len(f['points'])} puntos no usados en el descubrimiento): "
                   f"χ²={f['chi2_map']:.1f} (dof={f['dof']}), p={f['p_value']:.3g}, "
                   f"log-LR vs nulo = {f['llr_vs_null']:.1f}") if f["passed"] else \
            st.error(f"Holdout: χ²={f['chi2_map']:.1f} (dof={f['dof']}), p={f['p_value']:.3g}, "
                     f"log-LR vs nulo = {f['llr_vs_null']:.1f}")
        st.caption(f["next_decision"])
        return
    if f["experiment"] is None:
        a = f["anomaly"]
        st.info(f"Observer: χ²={a['chi2']:.1f} (dof={a['dof']}), p={a['p_value']:.3g} "
                f"para el modelo sin cuerpo oculto.")
        st.caption(f["next_decision"])
        return
    e, r = f["experiment"], f["obs_vs_models"]
    st.markdown(f"**Observar t = {e['time']:.2f} años** (política `{e['policy']}`)")
    st.write(e["reason"])
    st.markdown(
        f"**Resultado:** el modelo sin oculto falla por **{r['null_sigma']:.1f}σ**; "
        f"la hipótesis MAP por **{r['map_sigma']:.1f}σ**."
    )
    if f["newly_falsified"]:
        st.caption(f"Falsificadas en este paso: {len(f['newly_falsified'])} "
                   f"(p < {eng.FALSIFY_THRESHOLD:g}).")
    st.caption(f["next_decision"])


def _timeline(frames):
    rows = []
    for f in frames:
        exp = f.get("experiment")
        rows.append({
            "paso": f["step"],
            "estado": f["status"],
            "t observado": None if not exp else round(exp["time"], 2),
            "P(oculto) antes": None if f.get("p_hidden_before") is None else round(f["p_hidden_before"], 3),
            "P(oculto) después": round(f["p_hidden"], 3),
            "MAP": f["map_id"],
            "decisión siguiente": f["next_decision"],
        })
    st.markdown("#### Research timeline")
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)


def _agents(frames):
    msgs = [(f["step"], m) for f in frames for m in f.get("agent_messages", [])]
    with st.expander(f"Handoffs entre agentes ({len(msgs)} mensajes)"):
        for step, m in msgs:
            tag = "código" if m.get("source") == "code" else "LLM"
            st.markdown(f"**[{step}] {m['agent']}** · _{tag}_ — {m['message']}")


def _evaluation(record):
    ev, out = record["evaluation"], record["outcome"]
    st.markdown("#### Evaluación (ground truth, solo para medir)")
    t = ev["truth"]
    ok = "✅ correcta" if ev["correct"] else "❌ incorrecta"
    st.write(f"Verdad: m={t['hidden_mass']:.0e} M☉, a={t['hidden_a']:.2f} AU · "
             f"hipótesis MAP: `{ev['map_id']}` · identificación {ok} · "
             f"observaciones hasta identificar: **{out['n_observations']}**")
    st.caption("Una sola corrida NO es evidencia: las cifras oficiales salen del benchmark de P3 "
               "(múltiples semillas y escenarios).")


# ------------------------------------------------------------------ sidebar
st.sidebar.title("🔭 ATTN")
st.sidebar.caption("ATTN looks differently, tests ruthlessly, and trusts only what reproduces.")
mode = st.sidebar.radio("Modo", ["Ejecutar descubrimiento", "Cargar corrida guardada (replay)"])

record_loaded = None
if mode == "Ejecutar descubrimiento":
    backend_name = st.sidebar.selectbox("Backend", list(BACKENDS))
    scenario = st.sidebar.selectbox(
        "Escenario", list(range(len(eng.env.SCENARIOS))), index=1,
        format_func=lambda i: f"{i}: m={eng.env.SCENARIOS[i]['hidden_mass']:.0e}, "
                              f"a={eng.env.SCENARIOS[i]['hidden_a']}")
    policy = st.sidebar.selectbox("Política de experimentos", ["attn", "greedy", "random"])
    seed_txt = st.sidebar.text_input("Semilla de ruido (vacío = ruido de P1)", "")
    max_obs = st.sidebar.slider("Presupuesto máx. de experimentos", 3, 24, 24)
    delay = st.sidebar.slider("Pausa entre pasos (s)", 0.0, 3.0, 1.0, 0.25)
    run_clicked = st.sidebar.button("▶ Run Discovery", type="primary", use_container_width=True)
else:
    run_clicked = False
    runs = eng.list_runs()
    if not runs:
        st.sidebar.warning("No hay corridas en runs/. Ejecuta: python make_backup_run.py")
    else:
        choice = st.sidebar.selectbox("Corrida", runs, format_func=lambda p: p.name)
        if st.sidebar.button("Cargar", use_container_width=True):
            st.session_state["record"] = eng.load_record(choice)
            st.session_state["loaded_from_file"] = True

# ------------------------------------------------------------------ cuerpo
st.title("ATTN — Discover what you can't see")
st.caption("Question → Evidence → Hypothesis → Experiment → Result → Updated decision. "
           "LLM propone · código calcula · simulador juzga · estadística verifica.")

tab_main, tab_cmp = st.tabs(["Discovery", "Comparación rápida de políticas"])

with tab_main:
    with st.spinner("Cargando banco de hipótesis (la primera vez tarda ~10–60 s)..."):
        bank = load_bank()

    if run_clicked:
        seed = int(seed_txt) if seed_txt.strip() else None
        backend = BACKENDS[backend_name]
        holder = st.empty()
        record = None
        try:
            for ev in backend(scenario_index=scenario, policy=policy, max_obs=max_obs,
                              seed=seed, bank=bank):
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
        except Exception as exc:  # fallback visible para la demo
            st.error(f"El backend '{backend_name}' falló: {exc}. "
                     f"Usa el motor local o carga la corrida de respaldo.")
            raise
        if record is not None and record["steps"]:
            path = eng.save_record(record)
            st.session_state["record"] = record
            st.session_state["last_saved"] = str(path)
            st.rerun()

    record = st.session_state.get("record")
    if record is None:
        st.info("Pulsa **Run Discovery** o carga una corrida guardada desde la barra lateral.")
    else:
        frames = eng.build_frames(record)
        if st.session_state.get("last_saved"):
            st.caption(f"Corrida guardada en {st.session_state['last_saved']}")
        top = st.columns([1, 4])
        replay = top[0].button("⏯ Replay animado")
        k = top[1].slider("Paso del research record", 0, len(frames) - 1, len(frames) - 1,
                          key="slider_k")
        if replay:
            holder = st.empty()
            for i in range(len(frames)):
                with holder.container():
                    render_frame(bank, record, i, final_view=(i == len(frames) - 1))
                time.sleep(1.0)
        else:
            render_frame(bank, record, k, final_view=(k == len(frames) - 1))
        import json
        st.download_button("Descargar research record (JSON)",
                           json.dumps(record, ensure_ascii=False, indent=1),
                           file_name="attn_research_record.json", mime="application/json")

with tab_cmp:
    st.markdown("Vista previa rápida con el motor local. **La evaluación oficial es la de P3** "
                "(más semillas/escenarios, métricas finales).")
    cc = st.columns(3)
    sc_cmp = cc[0].selectbox("Escenario", list(range(len(eng.env.SCENARIOS))), index=3, key="sc_cmp")
    n_seeds = cc[1].slider("Semillas", 5, 50, 20)
    go_cmp = cc[2].button("Comparar random / greedy / attn")
    if go_cmp:
        with st.spinner("Corriendo políticas..."):
            rows = eng.quick_compare(sc_cmp, ["random", "greedy", "attn"], list(range(n_seeds)), load_bank())
        df = pd.DataFrame(rows)
        summ = df.groupby("policy").agg(
            obs_media=("n_observations", "mean"), obs_mediana=("n_observations", "median"),
            identificadas=("identified", "mean"), correctas=("correct", "mean"),
            replicadas=("replicated", "mean")).round(2)
        st.dataframe(summ, use_container_width=True)
        if "random" in summ.index and "attn" in summ.index:
            st.write(f"ATTN usó **{summ.loc['random', 'obs_media'] / summ.loc['attn', 'obs_media']:.2f}×** "
                     f"menos observaciones que random (media, {n_seeds} semillas, escenario {sc_cmp}).")
        st.bar_chart(summ["obs_media"])
