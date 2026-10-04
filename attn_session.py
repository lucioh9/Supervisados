"""
ATTN — sesión de descubrimiento para Omnigent (Persona 2).

Esta es la capa de HERRAMIENTAS que usan los agentes de Omnigent. Reutiliza el motor matemático de P4 (engine.py)
y el simulador de P1 (orbital_env.py), pero expone el loop en pasos que los AGENTES llaman uno a uno:

  observer_analyze -> hypothesis_register -> [ plan_experiment -> run_experiment -> assess_step ]* -> run_replication -> finalize

Principios:
  * LLM propone / explica (campos 'commentary', 'rationale', 'interpretation'); el CÓDIGO calcula todo.
  * Máquina de estados: una herramienta fuera de orden devuelve error (los agentes no pueden saltarse pasos,
    p.ej. no se puede replicar sin haber identificado, ni observar tiempos fuera de lo permitido).
  * La verdad (ground truth) solo se usa dentro de finalize() y NO se devuelve a los agentes.
  * Cada paso emite un evento con EXACTAMENTE el contrato de engine.validate_event -> la UI de P4 funciona igual.
"""
import json
import os
import sys
from pathlib import Path

import numpy as np

import engine as eng
import orbital_env as env

RUNS = Path(os.environ.get("ATTN_RUNS_DIR", "runs"))
STATE = RUNS / "_omni_state.json"
EVENTS = RUNS / "_omni_events.jsonl"
SCOUT = RUNS / "scout_report.json"
_BANK = None

HINT = {
    "need_observer": "observer_analyze", "need_hypothesis": "hypothesis_register",
    "planning": "plan_experiment", "planned": "run_experiment", "executed": "assess_step",
    "replicate": "run_replication", "need_finalize": "finalize", "done": "(finished)",
}


# ------------------------------------------------------------------ utilidades
def _bank():
    global _BANK
    if _BANK is None:
        _BANK = eng.get_bank()
    return _BANK


def _conv(o):
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(type(o))


def _emit(ev):
    eng.validate_event(ev)
    RUNS.mkdir(exist_ok=True)
    with open(EVENTS, "a", encoding="utf-8") as f:
        f.write(json.dumps(ev, default=_conv, ensure_ascii=False) + "\n")


def _load():
    if not STATE.exists():
        raise RuntimeError("No active session (the human/pipeline creates it with start_session).")
    return json.loads(STATE.read_text(encoding="utf-8"))


def _save(s):
    STATE.write_text(json.dumps(s, default=_conv, ensure_ascii=False), encoding="utf-8")


def _guard(s, *phases):
    if s["phase"] not in phases:
        return {"error": f"Current phase '{s['phase']}'. Expected tool now: {HINT[s['phase']]}"}
    return None


def _msg(agent, source, message):
    return {"agent": agent, "source": source, "message": message}


def _llm(agent, text):
    return [_msg(agent, "llm", text.strip())] if text and text.strip() else []


def _posterior(s, bank):
    K = len(bank.hyps)
    prior = np.full(K, (1.0 - eng.NULL_PRIOR) / (K - 1))
    prior[0] = eng.NULL_PRIOR
    logp = np.log(prior)
    for o in s["observations"]:
        logp += eng._loglik(bank, o)
    return eng._posterior(logp)


def _world(s, bank):
    return eng.World(s["scenario_index"], bank, s["seed"])


# ------------------------------------------------------------------ sesión (la inicia el humano, NO los agentes)
def start_session(scenario_index=1, policy="attn", seed=None, max_obs=None):
    bank = _bank()
    world = eng.World(scenario_index, bank, seed)
    cand = [float(t) for t in env.candidate_observation_times()]
    max_obs = len(cand) if max_obs is None else min(int(max_obs), len(cand))
    obs = [world.observe(t) for t in eng.INITIAL_TIMES]
    RUNS.mkdir(exist_ok=True)
    EVENTS.write_text("")
    if SCOUT.exists():
        SCOUT.unlink()
    s = {"scenario_index": int(scenario_index), "policy": policy, "seed": seed, "max_obs": max_obs,
         "observations": obs, "cand": cand, "n": 0, "phase": "need_observer", "falsified": [],
         "prev_sig": False, "msgs": [], "planned": None, "pending": None, "replication_passed": None}
    post = _posterior(s, bank)
    s["falsified"] = [h for h, p in zip(bank.ids, post) if p < eng.FALSIFY_THRESHOLD]
    _save(s)
    from datetime import datetime
    return {
        "type": "header",
        "meta": {"project": "LYDH", "backend": "omnigent", "scenario_index": scenario_index, "policy": policy,
                 "seed": seed, "noise_std": eng.SIGMA, "max_obs": max_obs,
                 "created": datetime.now().isoformat(timespec="seconds"),
                 "criteria": {"p_hidden_threshold": eng.P_HIDDEN_THRESHOLD, "cluster_threshold": eng.CLUSTER_THRESHOLD,
                              "min_obs_for_decision": eng.MIN_OBS_FOR_DECISION, "falsify_threshold": eng.FALSIFY_THRESHOLD,
                              "replication_pvalue": eng.REPLICATION_PVALUE, "replication_llr": eng.REPLICATION_LLR}},
        "hypotheses": bank.hyps, "initial_observations": obs,
    }


# ------------------------------------------------------------------ 1. OBSERVER
def observer_analyze(commentary=""):
    s = _load()
    if (e := _guard(s, "need_observer")):
        return e
    bank = _bank()
    a = eng.observer_report(bank, s["observations"])
    s["anomaly"], s["prev_sig"] = a, a["significant"]
    s["msgs"] = [_msg("observer", "code",
                      (f"Does 'no hidden planet' explain our first {len(s['observations'])} measurements? "
                       f"No clear fit: {eng.luck_text(a['p_value'])}." if a["significant"] else
                       f"'No hidden planet' explains our first {len(s['observations'])} measurements so far."))] + _llm("observer", commentary)
    s["phase"] = "need_hypothesis"
    _save(s)
    return {"anomaly_detected": a["significant"], "chi2": a["chi2"], "dof": a["dof"], "p_value": a["p_value"],
            "worst_time": a["worst_time"], "worst_sigma": a["worst_sigma"], "next": HINT[s["phase"]]}


# ------------------------------------------------------------------ 2. HYPOTHESIS AGENT
def hypothesis_register(rationale=""):
    s = _load()
    if (e := _guard(s, "need_hypothesis")):
        return e
    bank = _bank()
    post = _posterior(s, bank)
    state = eng._belief_state(bank, post, set(s["falsified"]))
    K = len(bank.hyps)
    msgs = s["msgs"] + [_msg("hypothesis", "code",
                             f"{K - 1} guesses for a hidden planet (each a different mass and distance), plus one guess "
                             f"that there is no planet and the gap is just measurement noise. Each guess predicts a different orbit.")] + _llm("hypothesis", rationale)
    _emit({"type": "step", "step": 0, "n_observations": 0,
           "status": "ANOMALY DETECTED" if s["prev_sig"] else "MONITORING",
           "anomaly": s["anomaly"], "experiment": None, "observation": None, "obs_vs_models": None,
           "newly_falsified": sorted(s["falsified"]), "candidate_scores": [], "p_hidden_before": None,
           "next_decision": "Pick the first most useful measurement.", "agent_messages": msgs, **state})
    s["msgs"], s["phase"] = [], "planning"
    _save(s)
    return {"n_hypotheses": K, "p_hidden": state["p_hidden"], "map_id": state["map_id"], "next": HINT[s["phase"]]}


# ------------------------------------------------------------------ 3. FALSIFIER / PLANNER
def plan_experiment(rationale=""):
    s = _load()
    if (e := _guard(s, "planning")):
        return e
    bank, cand, pol = _bank(), s["cand"], s["policy"]
    post = _posterior(s, bank)
    rng = np.random.default_rng((0 if s["seed"] is None else int(s["seed"])) * 1000 + s["n"] + 1)
    j, dis = eng.choose_experiment(pol, bank, post, cand, rng)
    rec = cand[j]
    order = [int(k) for k in np.argsort(-dis)[:3]]
    allowed = sorted({cand[k] for k in order} | {rec}) if pol == "attn" else [rec]
    reason = eng._explain_choice(pol, bank, post, rec, float(dis[j]))
    s["planned"] = {"time": rec, "reason": reason, "allowed": allowed,
                    "scores": [[float(t), float(v)] for t, v in zip(cand, dis)]}
    s["msgs"] += [_msg("planner", "code", reason)] + _llm("planner", rationale)
    s["phase"] = "planned"
    _save(s)
    return {"policy": pol, "recommended_time": rec, "allowed_times": allowed,
            "top3": [{"time": cand[k], "disagreement_sigma2": float(dis[k])} for k in order],
            "reason": reason, "p_hidden": float(1 - post[0]), "next": HINT[s["phase"]]}


def run_experiment(time=-1.0, rationale=""):
    s = _load()
    if (e := _guard(s, "planned")):
        return e
    bank, pl, pol = _bank(), s["planned"], s["policy"]
    t = pl["time"] if time is None or time < 0 else next((a for a in pl["allowed"] if abs(a - time) < 1e-6), None)
    if t is None:
        return {"error": f"t={time} not allowed. Valid options: {pl['allowed']}"}
    post0 = _posterior(s, bank)
    score = next(v for tt, v in pl["scores"] if abs(tt - t) < 1e-9)
    reason = pl["reason"] if abs(t - pl["time"]) < 1e-9 else (
        eng._explain_choice(pol, bank, post0, t, score) + " (the agent chose one of the top 3 alternatives)")
    obs = _world(s, bank).observe(t)
    s["observations"].append(obs)
    s["cand"] = [c for c in s["cand"] if abs(c - t) > 1e-9]
    s["n"] += 1
    post = _posterior(s, bank)
    new_f = [h for h, p in zip(bank.ids, post) if p < eng.FALSIFY_THRESHOLD and h not in set(s["falsified"])]
    s["falsified"] += new_f
    state = eng._belief_state(bank, post, set(s["falsified"]))
    anomaly = eng.observer_report(bank, s["observations"])
    ident = bool(s["n"] >= eng.MIN_OBS_FOR_DECISION and state["p_hidden"] >= eng.P_HIDDEN_THRESHOLD
                 and state["cluster_mass"] >= eng.CLUSTER_THRESHOLD and state["map_id"] != eng.NULL_ID)
    ti, k_map = bank.idx(t), bank.ids.index(state["map_id"])
    xy = np.array([obs["x"], obs["y"]])
    pn, pm = bank.V1[0, ti], bank.V1[k_map, ti]
    s["pending"] = {
        "experiment": {"time": float(t), "policy": pol, "reason": reason, "score": float(score)},
        "observation": obs, "newly_falsified": new_f, "candidate_scores": pl["scores"],
        "p_hidden_before": float(1 - post0[0]), "anomaly": anomaly, "state": state, "identified": ident,
        "obs_vs_models": {"obs": [obs["x"], obs["y"]], "null": pn.tolist(), "map": pm.tolist(),
                          "null_sigma": float(np.linalg.norm(xy - pn) / eng.SIGMA),
                          "map_sigma": float(np.linalg.norm(xy - pm) / eng.SIGMA)}}
    s["msgs"] += _llm("planner", rationale)
    s["phase"] = "executed"
    _save(s)
    o = s["pending"]["obs_vs_models"]
    return {"observed_time": float(t), "null_model_misses_by_sigma": o["null_sigma"], "map_model_misses_by_sigma": o["map_sigma"],
            "falsified_now": len(new_f), "p_hidden_before": s["pending"]["p_hidden_before"],
            "p_hidden_after": state["p_hidden"], "next": HINT[s["phase"]]}


# ------------------------------------------------------------------ 4. SCIENTIST / REPLICATOR
def assess_step(interpretation=""):
    s = _load()
    if (e := _guard(s, "executed")):
        return e
    pd, st, n = s["pending"], s["pending"]["state"], s["n"]
    anomaly, ident, new_f = pd["anomaly"], pd["identified"], pd["newly_falsified"]
    if ident:
        status = "HIDDEN CAUSE FOUND"
    elif anomaly["significant"] and not s["prev_sig"]:
        status = "ANOMALY DETECTED"
    elif new_f:
        status = "HYPOTHESIS FALSIFIED"
    elif anomaly["significant"]:
        status = "TESTING HYPOTHESIS"
    else:
        status = "MONITORING"
    s["prev_sig"] = anomaly["significant"]
    if ident:
        nxt, phase = "Strong candidate found: now double-check it on new measurements.", "replicate"
    elif n >= s["max_obs"]:
        nxt, phase = "Out of measurements without a clear answer.", "need_finalize"
    else:
        nxt, phase = "Still unsure: pick the next most useful measurement.", "planning"
    msgs = s["msgs"] + [_msg("scientist", "code",
                             f"Chance of a hidden planet: {pd['p_hidden_before']:.0%} → {st['p_hidden']:.0%}. {len(new_f)} {'guess' if len(new_f) == 1 else 'guesses'} "
                             f"ruled out this step ({st['n_falsified']} in total). {nxt}")] + _llm("scientist", interpretation)
    _emit({"type": "step", "step": n, "n_observations": n, "status": status, "anomaly": anomaly,
           "experiment": pd["experiment"], "observation": pd["observation"], "obs_vs_models": pd["obs_vs_models"],
           "newly_falsified": new_f, "candidate_scores": pd["candidate_scores"],
           "p_hidden_before": pd["p_hidden_before"], "next_decision": nxt, "agent_messages": msgs, **st})
    s["msgs"], s["phase"], s["anomaly"] = [], phase, anomaly
    _save(s)
    return {"status": status, "identified": ident, "p_hidden": st["p_hidden"], "cluster_mass": st["cluster_mass"],
            "map_id": st["map_id"], "n_observations": n, "next": HINT[phase]}


def run_replication(interpretation=""):
    from scipy.stats import chi2 as chi2_dist
    s = _load()
    if (e := _guard(s, "replicate")):
        return e
    bank = _bank()
    post = _posterior(s, bank)
    st = eng._belief_state(bank, post, set(s["falsified"]))
    world, k_map = _world(s, bank), bank.ids.index(st["map_id"])
    pts, cm, cn = [], 0.0, 0.0
    for t in env.replication_observation_times():
        o = world.observe(t)
        i = bank.idx(t)
        xy = np.array([o["x"], o["y"]])
        dm, dn = xy - bank.V1[k_map, i], xy - bank.V1[0, i]
        cm += float((dm ** 2).sum()) / eng.SIGMA ** 2
        cn += float((dn ** 2).sum()) / eng.SIGMA ** 2
        pts.append({"time": float(t), "obs": [o["x"], o["y"]], "map_pred": bank.V1[k_map, i].tolist(),
                    "null_pred": bank.V1[0, i].tolist(), "map_sigma": float(np.linalg.norm(dm) / eng.SIGMA),
                    "null_sigma": float(np.linalg.norm(dn) / eng.SIGMA)})
    dof = 2 * len(pts)
    pval = float(chi2_dist.sf(cm, dof))
    llr = 0.5 * (cn - cm)
    passed = bool(pval > eng.REPLICATION_PVALUE and llr >= eng.REPLICATION_LLR)
    msgs = [_msg("scientist", "code", f"Double-check on {len(pts)} new measurements kept hidden until now: "
                                      + (f"{eng.hname(st['map_id'])} explains them, and 'no hidden planet' does not."
                                         if passed else "our best guess does not hold up."))] + _llm("scientist", interpretation)
    _emit({"type": "replication", "step": s["n"] + 1, "n_observations": s["n"],
           "status": "REPLICATION PASSED" if passed else "REPLICATION FAILED", "points": pts, "chi2_map": cm,
           "chi2_null": cn, "dof": dof, "p_value": pval, "llr_vs_null": float(llr), "passed": passed,
           "anomaly": s["anomaly"], "experiment": None, "observation": None, "obs_vs_models": None,
           "newly_falsified": [], "candidate_scores": [], "p_hidden_before": st["p_hidden"],
           "next_decision": "Conclusion accepted." if passed else "The idea fails on new data: conclusion rejected.",
           "agent_messages": msgs, **st})
    s["replication_passed"], s["phase"] = passed, "need_finalize"
    _save(s)
    return {"replication_passed": passed, "p_value": pval, "log_lr_vs_null": float(llr), "next": HINT[s["phase"]]}


def finalize(summary=""):
    s = _load()
    if (e := _guard(s, "need_finalize")):
        return e
    bank = _bank()
    post = _posterior(s, bank)
    st = eng._belief_state(bank, post, set(s["falsified"]))
    identified = s["replication_passed"] is not None
    truth = env.evaluation_truth(s["scenario_index"])      # la verdad SOLO se usa aquí, para evaluar
    ti = int(np.argmin([abs(m - truth["hidden_mass"]) for m in eng.MASSES]))
    tj = int(np.argmin([abs(a - truth["hidden_a"]) for a in eng.SEMIAXES]))
    h = bank.hyps[bank.ids.index(st["map_id"])]
    ok = bool(h["kind"] == "hidden_body" and abs(h["gi"] - ti) <= 1 and abs(h["gj"] - tj) <= 1)
    rp = s["replication_passed"]
    final = "INCONCLUSIVE" if not identified else ("REPLICATION PASSED" if rp else "REPLICATION FAILED")
    _emit({"type": "final",
           "outcome": {"identified": identified, "n_observations": s["n"], "replication_passed": rp, "final_status": final},
           "evaluation": {"note": "The real answer is used only to grade the result. The agents never see it.", "truth": truth,
                          "map_id": st["map_id"], "map_within_one_cell_of_truth": ok,
                          "correct": bool(identified and ok), "truth_grid_index": [ti, tj]}})
    s["phase"] = "done"
    _save(s)
    return {"final_status": final, "n_observations": s["n"], "replication_passed": rp,
            "note": "Session closed. The grading against the real answer is saved in the research record (not visible to agents)."}


# ------------------------------------------------------------------ extras
def get_status():
    s = _load()
    bank = _bank()
    post = _posterior(s, bank)
    st = eng._belief_state(bank, post, set(s["falsified"]))
    return {"phase": s["phase"], "next_tool": HINT[s["phase"]], "policy": s["policy"], "n_observations": s["n"],
            "max_obs": s["max_obs"], "p_hidden": st["p_hidden"], "map_id": st["map_id"], "cluster_mass": st["cluster_mass"]}


def scout_note(open_problems="", related_work="", who_else="", biases_and_risks=""):
    RUNS.mkdir(exist_ok=True)
    rep = {"open_problems": open_problems, "related_work": related_work, "who_else": who_else,
           "biases_and_risks": biases_and_risks, "note": "Written by an AI agent: verify citations before using them."}
    SCOUT.write_text(json.dumps(rep, ensure_ascii=False, indent=1), encoding="utf-8")
    return {"saved": str(SCOUT)}


def autopilot():
    """Fallback 100 % código: completa la sesión actual sin LLM (mismas herramientas, mismo contrato)."""
    table = {"need_observer": observer_analyze, "need_hypothesis": hypothesis_register, "planning": plan_experiment,
             "planned": run_experiment, "executed": assess_step, "replicate": run_replication, "need_finalize": finalize}
    for _ in range(200):
        ph = _load()["phase"]
        if ph == "done":
            return True
        out = table[ph]()
        if isinstance(out, dict) and "error" in out:
            raise RuntimeError(out["error"])
    raise RuntimeError("autopilot no terminó")


TOOLS = {f.__name__: f for f in (observer_analyze, hypothesis_register, plan_experiment, run_experiment,
                                 assess_step, run_replication, finalize, get_status, scout_note)}

if __name__ == "__main__":      # uso:  python attn_session.py <herramienta> '<json con argumentos>'
    name = sys.argv[1]
    kw = json.loads(sys.argv[2]) if len(sys.argv) > 2 else {}
    try:
        out = TOOLS[name](**kw)
    except Exception as exc:
        out = {"error": f"{type(exc).__name__}: {exc}"}
    print(json.dumps(out, default=_conv, ensure_ascii=False))
