"""
Backend Omnigent para la UI de P4.  app.py lo detecta solo (engine.load_omnigent_backend importa este módulo).

run_discovery(...) es un GENERADOR con el mismo contrato que engine.run_discovery:
  header -> step* -> [replication] -> final
Flujo: crea la sesión (el humano fija escenario/política/semilla) -> lanza `omnigent run attn_lab/config.yaml -p ...` ->
los agentes llaman a las herramientas, que escriben eventos en runs/_omni_events.jsonl -> aquí se leen EN VIVO y se emiten a la UI.
Si Omnigent/LLM falla, `attn_session.autopilot()` completa la sesión con código (los mensajes quedan marcados source='code').
"""
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

import engine as eng
import attn_session as ses

HERE = Path(__file__).resolve().parent
CONFIG = HERE / "attn_lab" / "attn_lab.yaml"
PROMPT = ("Ejecuta el ciclo completo de descubrimiento para la sesión activa siguiendo tu protocolo: scout, observer, "
          "hypothesis, bucle falsifier/scientist hasta identificar o agotar presupuesto, replicación y finalize. No uses terminal ni comandos de shell: solo las herramientas de los sub-agentes.")
IDLE_TIMEOUT = float(os.environ.get("ATTN_IDLE_TIMEOUT", "240"))   # s sin eventos nuevos antes de caer al fallback
TOTAL_TIMEOUT = float(os.environ.get("ATTN_TOTAL_TIMEOUT", "1200"))


def _read_new(offset):
    if not ses.EVENTS.exists():
        return [], offset
    lines = ses.EVENTS.read_text(encoding="utf-8").splitlines()
    return [json.loads(l) for l in lines[offset:] if l.strip()], len(lines)


def run_discovery(scenario_index=1, policy="attn", max_obs=None, seed=None, bank=None, progress=None):
    header = ses.start_session(scenario_index, policy, seed, max_obs)
    yield header

    exe = shutil.which("omnigent") or shutil.which("omni")
    env = dict(os.environ, PYTHONPATH=str(HERE) + os.pathsep + os.environ.get("PYTHONPATH", ""),
               ATTN_PYTHON=sys.executable, ATTN_WORKDIR=str(HERE), ATTN_RUNS_DIR=str(ses.RUNS.resolve()))
    proc, log = None, None
    if exe:
        ses.RUNS.mkdir(exist_ok=True)
        log = open(ses.RUNS / "_omni_stdout.log", "w", encoding="utf-8")
        proc = subprocess.Popen([exe, "run", str(CONFIG), "-p", PROMPT], cwd=str(HERE), env=env,
                                stdout=log, stderr=subprocess.STDOUT, text=True)

    offset, t0, last, done = 0, time.time(), time.time(), False
    while proc is not None and not done:
        evs, offset = _read_new(offset)
        for ev in evs:
            last = time.time()
            eng.validate_event(ev)
            yield ev
            done = done or ev["type"] == "final"
        if done:
            break
        if proc.poll() is not None and not evs:
            break                                   # Omnigent terminó sin cerrar la sesión
        if time.time() - last > IDLE_TIMEOUT or time.time() - t0 > TOTAL_TIMEOUT:
            break
        time.sleep(0.4)

    if proc is not None and proc.poll() is None:
        proc.terminate()
    if log:
        log.close()
    if not done:                                    # FALLBACK determinista: nunca dejar la demo colgada
        os.environ["ATTN_RUNS_DIR"] = str(ses.RUNS)
        ses.autopilot()
        evs, offset = _read_new(offset)
        for ev in evs:
            eng.validate_event(ev)
            yield ev
