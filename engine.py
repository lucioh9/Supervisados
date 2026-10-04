"""
ATTN engine — capa de integración de Persona 4.

Motor matemático LOCAL (sin LLM) que ejecuta el loop científico sobre el simulador de P1:
    observaciones -> anomalía -> hipótesis -> mejor experimento -> resultado -> creencia -> replicación

Sirve para 3 cosas:
  1. La UI funciona end-to-end desde ya, sin esperar a Omnigent (P2).
  2. Define el CONTRATO de eventos que el backend de Omnigent debe emitir (ver validate_event).
  3. Es el fallback si Omnigent/LLM falla durante la demo.

Principio: LLM propone, código calcula, simulador juzga. Aquí todo el cálculo es código.
"""
import hashlib
import json
import pickle
from datetime import datetime
from pathlib import Path

import numpy as np
from scipy.stats import chi2 as chi2_dist

import orbital_env as env

CACHE_DIR = Path(".cache")
RUNS_DIR = Path("runs")

SIGMA = env.NOISE_STD
NULL_ID = "H0_no_hidden"

# Rejilla de hipótesis "cuerpo oculto" (contiene los 4 escenarios de P1 como puntos de la rejilla).
MASSES = [3e-5, 4e-5, 5e-5, 6e-5, 7e-5]
SEMIAXES = [1.30, 1.45, 1.60, 1.75, 1.90]

# Debe coincidir con orbital_env.initial_observations (lo verifica check_integration.py)
INITIAL_TIMES = np.linspace(0.1, 2.0, 8)

# Criterios (documentarlos en la presentación)
NULL_PRIOR = 0.25
P_HIDDEN_THRESHOLD = 0.99    # P(existe cuerpo oculto)
CLUSTER_THRESHOLD = 0.90     # masa posterior en la vecindad (±1 celda) de la hipótesis MAP
CLUSTER_RADIUS = 1           # vecindad en celdas; scale_curve.py lo escala con el tamaño de la rejilla
MIN_OBS_FOR_DECISION = 3     # no se declara causa con menos de 3 experimentos (evita falsos positivos por 1 dato)
FALSIFY_THRESHOLD = 1e-3     # hipótesis con p < 1e-3 se consideran falsificadas
REPLICATION_PVALUE = 0.01    # chi² de la hipótesis MAP en holdout
REPLICATION_LLR = float(np.log(100.0))  # evidencia mínima vs. modelo nulo en holdout

STATUSES = [
    "MONITORING", "ANOMALY DETECTED", "TESTING HYPOTHESIS", "HYPOTHESIS FALSIFIED",
    "HIDDEN CAUSE FOUND", "REPLICATION PASSED", "REPLICATION FAILED", "INCONCLUSIVE",
]


# ----------------------------------------------------------------------------------------------
# Hipótesis y banco de predicciones
# ----------------------------------------------------------------------------------------------
def build_hypotheses():
    hyps = [{
        "id": NULL_ID, "kind": "null",
        "label": "Sin cuerpo oculto (solo ruido)",
        "mass": None, "a": None, "gi": None, "gj": None,
    }]
    for i, m in enumerate(MASSES):
        for j, a in enumerate(SEMIAXES):
            hyps.append({
                "id": f"H_m{m * 1e5:.0f}_a{a:.2f}", "kind": "hidden_body",
                "label": f"Cuerpo oculto m={m:.0e} M☉, a={a:.2f} AU",
                "mass": m, "a": a, "gi": i, "gj": j,
            })
    return hyps


def hname(hid):
    """Nombre legible de una hipótesis: 'H_m5_a1.60' -> 'mass 5 / distance 1.60 AU'."""
    if hid == NULL_ID:
        return "no hidden planet"
    m_s, a_s = hid[3:].split("_a")
    return f"a planet {float(a_s):.2f} AU out (mass {float(m_s):.0f})"


def luck_text(p):
    """p-valor -> frase simple."""
    if p < 0.001:
        return "less than 0.1% chance this is just luck"
    return f"{p:.1%} chance this is just luck"


def _key(t):
    return round(float(t), 9)


def time_grid():
    ts = np.concatenate([
        INITIAL_TIMES,
        env.candidate_observation_times(),
        env.replication_observation_times(),
        np.linspace(0.0, 10.0, 201),
    ])
    return np.unique(np.round(ts, 9))


class HypothesisBank:
    """Predicciones precalculadas (1 simulación por hipótesis, cacheadas en disco)."""

    def __init__(self, progress=None, use_cache=True):
        self.hyps = build_hypotheses()
        self.ids = [h["id"] for h in self.hyps]
        self.times = time_grid()
        self.index = {_key(t): i for i, t in enumerate(self.times)}
        self.traj = self._load_or_build(progress, use_cache)
        self.V1 = np.stack([self.traj[i]["v1"] for i in self.ids])  # (K, T, 2)

    def idx(self, t):
        return self.index[_key(t)]

    def _cache_path(self):
        h = hashlib.sha256()
        h.update(Path(env.__file__).read_bytes())
        h.update(json.dumps([MASSES, SEMIAXES]).encode())
        h.update(self.times.tobytes())
        return CACHE_DIR / f"bank_{h.hexdigest()[:16]}.pkl"

    def _simulate(self, h):
        if h["kind"] == "null":
            data = env.simulate_system(self.times, include_hidden=False)
        else:
            data = env.simulate_system(
                self.times, include_hidden=True, hidden_mass=h["mass"], hidden_a=h["a"]
            )
        return {
            "v1": data[:, 1, :2].copy(),
            "v2": data[:, 2, :2].copy(),
            "hid": data[:, 3, :2].copy() if data.shape[1] > 3 else None,
        }

    def _load_or_build(self, progress, use_cache):
        path = self._cache_path()
        if use_cache and path.exists():
            with open(path, "rb") as f:
                return pickle.load(f)
        traj = {}
        n = len(self.hyps)
        for i, h in enumerate(self.hyps):
            traj[h["id"]] = self._simulate(h)
            if progress:
                progress(i + 1, n)
        if use_cache:
            CACHE_DIR.mkdir(exist_ok=True)
            with open(path, "wb") as f:
                pickle.dump(traj, f)
        return traj


_TRUTH_CACHE = {}


class World:
    """
    El 'mundo' que responde a observaciones. Usa simulate_system de P1 (una sola llamada batch
    por escenario) y añade ruido gaussiano NOISE_STD, igual que orbital_env.observe.
    seed=None -> mismo ruido que observe() de P1 (determinista por tiempo).
    seed=int  -> ruido independiente por semilla (para repetir corridas).
    El agente/motor SOLO accede a observe(); la verdad se usa únicamente en evaluate().
    """

    def __init__(self, scenario_index, bank, seed=None):
        self.scenario_index = scenario_index
        self.seed = seed
        key = (scenario_index, bank.times.tobytes())
        if key not in _TRUTH_CACHE:
            s = env.SCENARIOS[scenario_index]
            data = env.simulate_system(
                bank.times, include_hidden=True,
                hidden_mass=s["hidden_mass"], hidden_a=s["hidden_a"],
            )
            _TRUTH_CACHE[key] = data[:, 1, :2].copy()
        self._v1 = _TRUTH_CACHE[key]
        self._bank = bank

    def observe(self, t):
        xy = self._v1[self._bank.idx(t)].copy()
        t_int = int(round(float(t) * 10000))
        if self.seed is None:
            rng = np.random.default_rng(env.BASE_SEED + t_int)
        else:
            rng = np.random.default_rng(int(self.seed) * 1_000_003 + t_int)
        xy += rng.normal(0, SIGMA, size=2)
        return {"time": float(t), "x": float(xy[0]), "y": float(xy[1])}


# ----------------------------------------------------------------------------------------------
# Matemática: verosimilitud, posterior, observer, políticas
# ----------------------------------------------------------------------------------------------
def _loglik(bank, obs):
    d = bank.V1[:, bank.idx(obs["time"]), :] - np.array([obs["x"], obs["y"]])
    return -0.5 * (d ** 2).sum(axis=1) / SIGMA ** 2


def _posterior(logp):
    z = np.exp(logp - logp.max())
    return z / z.sum()


def observer_report(bank, observations):
    """Agente Observer (código): ¿el modelo SIN cuerpo oculto explica las observaciones?"""
    res = []
    for o in observations:
        d = np.array([o["x"], o["y"]]) - bank.V1[0, bank.idx(o["time"])]
        res.append((o["time"], float(np.linalg.norm(d) / SIGMA)))
    chi2 = float(sum(r[1] ** 2 for r in res))
    dof = 2 * len(res)
    p = float(chi2_dist.sf(chi2, dof))
    worst = max(res, key=lambda r: r[1])
    return {
        "chi2": chi2, "dof": dof, "p_value": p, "significant": p < 0.01,
        "worst_time": worst[0], "worst_sigma": worst[1],
    }


def choose_experiment(policy, bank, post, cand_times, rng):
    """
    Devuelve (índice elegido, disagreement por candidato).
    disagreement(t) = Var_posterior[predicción(t)] / σ²   (proxy de ganancia esperada de información)
    - attn  : t que maximiza el desacuerdo entre hipótesis (ponderado por su probabilidad)
    - greedy: t donde el modelo actual (media posterior) más se aleja del modelo nulo ('donde el error crece')
    - random: al azar
    NOTA: la política oficial del benchmark la define P3; este mismo interfaz permite sustituirla.
    """
    idxs = np.array([bank.idx(t) for t in cand_times])
    mu = bank.V1[:, idxs, :]                        # (K, C, 2)
    mbar = np.tensordot(post, mu, axes=1)           # (C, 2)
    disagree = (post[:, None] * ((mu - mbar) ** 2).sum(-1)).sum(0) / SIGMA ** 2
    if policy == "attn":
        j = int(np.argmax(disagree))
    elif policy == "greedy":
        j = int(np.argmax(np.linalg.norm(mbar - mu[0], axis=-1)))
    elif policy == "random":
        j = int(rng.integers(len(cand_times)))
    else:
        raise ValueError(f"policy desconocida: {policy}")
    return j, disagree


def _explain_choice(policy, bank, post, t, score):
    if policy == "random":
        return f"Random pick: year {t:.1f} was chosen by chance among the dates not measured yet."
    if policy == "greedy":
        return (f"Biggest-error pick: year {t:.1f} is where our current best guess differs most "
                f"from the 'no hidden planet' picture.")
    mu = bank.V1[:, bank.idx(t), :]
    top = np.argsort(post)[::-1][:6]
    best = None
    for a in top:
        for b in top:
            if a < b:
                d = float(np.linalg.norm(mu[a] - mu[b]) / SIGMA)
                w = post[a] * post[b] * d * d
                if best is None or w > best[0]:
                    best = (w, a, b, d)
    if best is None:
        return f"Year {t:.1f} is where our guesses disagree the most."
    _, a, b, d = best
    return (f"At year {t:.1f}, our two leading guesses disagree the most: {hname(bank.ids[a])} "
            f"({post[a]:.0%} likely) vs {hname(bank.ids[b])} ({post[b]:.0%} likely). "
            f"They predict positions {d:.1f}× the normal measurement error apart, "
            f"so measuring here should rule one of them out.")


# ----------------------------------------------------------------------------------------------
# Estado de creencia
# ----------------------------------------------------------------------------------------------
def _belief_state(bank, post, falsified):
    k = int(np.argmax(post))
    if k == 0:
        cluster = float(post[0])
    else:
        h0 = bank.hyps[k]
        cluster = float(sum(
            post[i] for i, h in enumerate(bank.hyps)
            if h["kind"] == "hidden_body"
            and abs(h["gi"] - h0["gi"]) <= CLUSTER_RADIUS and abs(h["gj"] - h0["gj"]) <= CLUSTER_RADIUS
        ))
    entropy = float(-(post * np.log2(post + 1e-300)).sum())
    return {
        "posterior": {hid: float(p) for hid, p in zip(bank.ids, post)},
        "p_hidden": float(1.0 - post[0]),
        "map_id": bank.ids[k],
        "map_p": float(post[k]),
        "cluster_mass": cluster,
        "entropy_bits": entropy,
        "n_falsified": len(falsified),
    }


# ----------------------------------------------------------------------------------------------
# Loop principal (generador de eventos)
# ----------------------------------------------------------------------------------------------
def get_bank(progress=None):
    return HypothesisBank(progress=progress)


def run_discovery(scenario_index=1, policy="attn", max_obs=None, seed=None,
                  bank=None, progress=None):
    """
    Generador de eventos (CONTRATO con la UI; Omnigent debe emitir lo mismo):
      {"type":"header", ...}  ->  {"type":"step", ...} * N  ->  [{"type":"replication", ...}]
      -> {"type":"final", ...}
    """
    bank = bank or get_bank(progress)
    world = World(scenario_index, bank, seed)
    rng = np.random.default_rng(0 if seed is None else seed)
    cand = list(env.candidate_observation_times())
    max_obs = len(cand) if max_obs is None else min(int(max_obs), len(cand))

    K = len(bank.hyps)
    prior = np.full(K, (1.0 - NULL_PRIOR) / (K - 1))
    prior[0] = NULL_PRIOR
    logp = np.log(prior)

    observations = [world.observe(t) for t in INITIAL_TIMES]
    for o in observations:
        logp += _loglik(bank, o)
    post = _posterior(logp)
    falsified = {h for h, p in zip(bank.ids, post) if p < FALSIFY_THRESHOLD}

    yield {
        "type": "header",
        "meta": {
            "project": "LYDH", "backend": "local-math-engine", "scenario_index": scenario_index,
            "policy": policy, "seed": seed, "noise_std": SIGMA, "max_obs": max_obs,
            "created": datetime.now().isoformat(timespec="seconds"),
            "criteria": {
                "p_hidden_threshold": P_HIDDEN_THRESHOLD, "cluster_threshold": CLUSTER_THRESHOLD,
                "min_obs_for_decision": MIN_OBS_FOR_DECISION, "falsify_threshold": FALSIFY_THRESHOLD, "replication_pvalue": REPLICATION_PVALUE,
                "replication_llr": REPLICATION_LLR,
            },
        },
        "hypotheses": bank.hyps,
        "initial_observations": observations,
    }

    # ---- Paso 0: Observer sobre las observaciones iniciales
    anomaly = observer_report(bank, observations)
    prev_sig = anomaly["significant"]
    state = _belief_state(bank, post, falsified)
    yield {
        "type": "step", "step": 0, "n_observations": 0,
        "status": "ANOMALY DETECTED" if prev_sig else "MONITORING",
        "anomaly": anomaly, "experiment": None, "observation": None, "obs_vs_models": None,
        "newly_falsified": sorted(falsified), "candidate_scores": [], "p_hidden_before": None,
        "next_decision": "Pick the first most useful measurement.",
        "agent_messages": [
            {"agent": "observer", "source": "code",
             "message": (f"Does 'no hidden planet' explain our first {len(observations)} measurements? "
                         f"No clear fit: {luck_text(anomaly['p_value'])}." if anomaly["significant"] else
                         f"'No hidden planet' explains our first {len(observations)} measurements so far.")},
            {"agent": "hypothesis", "source": "code",
             "message": (f"{K - 1} guesses for a hidden planet (each a different mass and distance), plus one guess "
                         f"that there is no planet and the gap is just measurement noise. Each guess predicts a different orbit.")},
        ],
        **state,
    }

    identified = False
    n = 0
    for n in range(1, max_obs + 1):
        p_before = float(1.0 - post[0])
        j, disagree = choose_experiment(policy, bank, post, cand, rng)
        scores = [[float(t), float(s)] for t, s in zip(cand, disagree)]
        t = cand.pop(j)
        reason = _explain_choice(policy, bank, post, t, float(disagree[j]))

        obs = world.observe(t)
        observations.append(obs)
        logp += _loglik(bank, obs)
        post = _posterior(logp)

        new_f = [h for h, p in zip(bank.ids, post) if p < FALSIFY_THRESHOLD and h not in falsified]
        falsified.update(new_f)
        anomaly = observer_report(bank, observations)
        state = _belief_state(bank, post, falsified)
        identified = (n >= MIN_OBS_FOR_DECISION
                      and state["p_hidden"] >= P_HIDDEN_THRESHOLD
                      and state["cluster_mass"] >= CLUSTER_THRESHOLD
                      and state["map_id"] != NULL_ID)

        if identified:
            status = "HIDDEN CAUSE FOUND"
        elif anomaly["significant"] and not prev_sig:
            status = "ANOMALY DETECTED"
        elif new_f:
            status = "HYPOTHESIS FALSIFIED"
        elif anomaly["significant"]:
            status = "TESTING HYPOTHESIS"
        else:
            status = "MONITORING"
        prev_sig = anomaly["significant"]

        xy = np.array([obs["x"], obs["y"]])
        k_map = bank.ids.index(state["map_id"])
        ti = bank.idx(t)
        pred_null = bank.V1[0, ti]
        pred_map = bank.V1[k_map, ti]
        if identified:
            nxt = "Strong candidate found: now double-check it on new measurements."
        elif n == max_obs:
            nxt = "Out of measurements without a clear answer."
        else:
            nxt = "Still unsure: pick the next most useful measurement."

        yield {
            "type": "step", "step": n, "n_observations": n, "status": status,
            "anomaly": anomaly,
            "experiment": {"time": float(t), "policy": policy, "reason": reason,
                           "score": float(disagree[j])},
            "observation": obs,
            "obs_vs_models": {
                "obs": [obs["x"], obs["y"]], "null": pred_null.tolist(), "map": pred_map.tolist(),
                "null_sigma": float(np.linalg.norm(xy - pred_null) / SIGMA),
                "map_sigma": float(np.linalg.norm(xy - pred_map) / SIGMA),
            },
            "newly_falsified": new_f, "candidate_scores": scores, "p_hidden_before": p_before,
            "next_decision": nxt,
            "agent_messages": [
                {"agent": "planner", "source": "code", "message": reason},
                {"agent": "scientist", "source": "code",
                 "message": (f"Chance of a hidden planet: {p_before:.0%} → {state['p_hidden']:.0%}. "
                             f"{len(new_f)} {'guess' if len(new_f) == 1 else 'guesses'} ruled out this step ({state['n_falsified']} in total). {nxt}")},
            ],
            **state,
        }
        if identified:
            break

    # ---- Replicación sobre datos holdout (no usados en el descubrimiento)
    replication_passed = None
    if identified:
        k_map = bank.ids.index(state["map_id"])
        pts, chi_map, chi_null = [], 0.0, 0.0
        for t in env.replication_observation_times():
            o = world.observe(t)
            i = bank.idx(t)
            d_map = np.array([o["x"], o["y"]]) - bank.V1[k_map, i]
            d_null = np.array([o["x"], o["y"]]) - bank.V1[0, i]
            chi_map += float((d_map ** 2).sum()) / SIGMA ** 2
            chi_null += float((d_null ** 2).sum()) / SIGMA ** 2
            pts.append({
                "time": float(t), "obs": [o["x"], o["y"]],
                "map_pred": bank.V1[k_map, i].tolist(), "null_pred": bank.V1[0, i].tolist(),
                "map_sigma": float(np.linalg.norm(d_map) / SIGMA),
                "null_sigma": float(np.linalg.norm(d_null) / SIGMA),
            })
        dof = 2 * len(pts)
        pval = float(chi2_dist.sf(chi_map, dof))
        llr = 0.5 * (chi_null - chi_map)
        replication_passed = bool(pval > REPLICATION_PVALUE and llr >= REPLICATION_LLR)
        yield {
            "type": "replication", "step": n + 1, "n_observations": n,
            "status": "REPLICATION PASSED" if replication_passed else "REPLICATION FAILED",
            "points": pts, "chi2_map": chi_map, "chi2_null": chi_null, "dof": dof,
            "p_value": pval, "llr_vs_null": float(llr), "passed": replication_passed,
            "anomaly": anomaly, "experiment": None, "observation": None, "obs_vs_models": None,
            "newly_falsified": [], "candidate_scores": [], "p_hidden_before": state["p_hidden"],
            "next_decision": ("Conclusion accepted." if replication_passed
                              else "The idea fails on new data: conclusion rejected."),
            "agent_messages": [{
                "agent": "scientist", "source": "code",
                "message": (f"Double-check on {len(pts)} new measurements kept hidden until now: "
                            + (f"{hname(state['map_id'])} explains them, and 'no hidden planet' does not."
                               if replication_passed else "our best guess does not hold up."))}],
            **state,
        }

    # ---- Evaluación (usa verdad solo aquí)
    truth = env.evaluation_truth(scenario_index)
    ti = int(np.argmin([abs(m - truth["hidden_mass"]) for m in MASSES]))
    tj = int(np.argmin([abs(a - truth["hidden_a"]) for a in SEMIAXES]))
    map_h = bank.hyps[bank.ids.index(state["map_id"])]
    map_ok = (map_h["kind"] == "hidden_body"
              and abs(map_h["gi"] - ti) <= CLUSTER_RADIUS and abs(map_h["gj"] - tj) <= CLUSTER_RADIUS)
    if not identified:
        final_status = "INCONCLUSIVE"
    else:
        final_status = "REPLICATION PASSED" if replication_passed else "REPLICATION FAILED"
    yield {
        "type": "final",
        "outcome": {"identified": identified, "n_observations": n,
                    "replication_passed": replication_passed, "final_status": final_status},
        "evaluation": {
            "note": "The real answer is used only to grade the result. The agents never see it.",
            "truth": truth, "map_id": state["map_id"],
            "map_within_one_cell_of_truth": bool(map_ok),
            "correct": bool(identified and map_ok),
            "truth_grid_index": [ti, tj],
        },
    }


# ----------------------------------------------------------------------------------------------
# Record, contrato, persistencia
# ----------------------------------------------------------------------------------------------
_REQUIRED = {
    "header": {"meta", "hypotheses", "initial_observations"},
    "step": {"step", "n_observations", "status", "posterior", "p_hidden", "map_id", "map_p",
             "cluster_mass", "entropy_bits", "n_falsified", "anomaly", "experiment", "observation",
             "newly_falsified", "candidate_scores", "next_decision", "agent_messages"},
    "replication": {"status", "points", "p_value", "llr_vs_null", "passed", "posterior",
                    "p_hidden", "map_id", "n_falsified", "n_observations"},
    "final": {"outcome", "evaluation"},
}


def validate_event(ev):
    """Úsalo para comprobar que el backend de Omnigent (P2) respeta el contrato."""
    t = ev.get("type")
    if t not in _REQUIRED:
        raise ValueError(f"type desconocido: {t}")
    missing = _REQUIRED[t] - set(ev)
    if missing:
        raise ValueError(f"evento '{t}' sin campos: {sorted(missing)}")
    if "status" in ev and ev["status"] not in STATUSES:
        raise ValueError(f"status inválido: {ev['status']}")
    return True


def new_record(header):
    rec = {k: v for k, v in header.items() if k != "type"}
    rec.update({"steps": [], "replication": None, "outcome": None, "evaluation": None})
    return rec


def add_event(record, ev):
    t = ev["type"]
    if t == "step":
        record["steps"].append(ev)
    elif t == "replication":
        record["replication"] = ev
    elif t == "final":
        record["outcome"] = ev["outcome"]
        record["evaluation"] = ev["evaluation"]
    return record


def run_to_record(**kwargs):
    record = None
    for ev in run_discovery(**kwargs):
        if ev["type"] == "header":
            record = new_record(ev)
        else:
            add_event(record, ev)
    return record


def build_frames(record):
    return record["steps"] + ([record["replication"]] if record.get("replication") else [])


def save_record(record, path=None):
    RUNS_DIR.mkdir(exist_ok=True)
    if path is None:
        m = record["meta"]
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = RUNS_DIR / f"run_{m['policy']}_sc{m['scenario_index']}_{stamp}.json"
    with open(path, "w", encoding="utf-8") as f:
        json.dump(record, f, ensure_ascii=False, indent=1)
    return Path(path)


def load_record(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def list_runs():
    RUNS_DIR.mkdir(exist_ok=True)
    return sorted(RUNS_DIR.glob("*.json"), key=lambda p: p.stat().st_mtime, reverse=True)


def load_omnigent_backend():
    """Si P2 deja un módulo omnigent_pipeline.py con run_discovery(...) (mismo contrato), se usa."""
    try:
        import omnigent_pipeline  # noqa: F401
        return omnigent_pipeline.run_discovery
    except Exception:
        return None


def quick_compare(scenario_index, policies, seeds, bank):
    """Vista previa rápida (la evaluación oficial es de P3)."""
    rows = []
    for pol in policies:
        for s in seeds:
            rec = run_to_record(scenario_index=scenario_index, policy=pol, seed=s, bank=bank)
            o, e = rec["outcome"], rec["evaluation"]
            rows.append({"policy": pol, "seed": s, "n_observations": o["n_observations"],
                         "identified": o["identified"], "correct": e["correct"],
                         "replicated": bool(o["replication_passed"])})
    return rows
