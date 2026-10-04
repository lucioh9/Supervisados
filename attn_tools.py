"""
Function tools that Omnigent exposes to the agents. stdlib only: each call runs attn_session.py with the project
interpreter (ATTN_PYTHON). The docstrings are what the LLM reads to know when to use each tool.
"""
import json
import os
import subprocess
import sys

_HERE = os.path.dirname(os.path.abspath(__file__))


def _call(fn: str, **kw) -> dict:
    py = os.environ.get("ATTN_PYTHON") or sys.executable
    r = subprocess.run([py, os.path.join(_HERE, "attn_session.py"), fn, json.dumps(kw)],
                       capture_output=True, text=True, cwd=os.environ.get("ATTN_WORKDIR", _HERE), timeout=600)
    lines = [ln for ln in r.stdout.strip().splitlines() if ln.strip()]
    if r.returncode != 0 or not lines:
        return {"error": (r.stderr or "no output")[-600:]}
    try:
        return json.loads(lines[-1])
    except Exception:
        return {"error": "output was not JSON: " + lines[-1][:300]}


def observer_analyze(commentary: str = "") -> dict:
    """OBSERVER. Step 1. Code checks whether the model WITHOUT a hidden planet explains the first measurements.
    `commentary`: 1-2 plain-English sentences (high-school level, no jargon) saying what does not add up."""
    return _call("observer_analyze", commentary=commentary)


def hypothesis_register(rationale: str = "") -> dict:
    """HYPOTHESIS AGENT. Step 2. Registers the list of guesses (no planet, or a hidden planet of some mass and distance;
    each predicts an orbit). `rationale`: plain English, why these alternative causes are plausible."""
    return _call("hypothesis_register", rationale=rationale)


def plan_experiment(rationale: str = "") -> dict:
    """FALSIFIER/PLANNER. Code computes which measurement date makes the guesses disagree the most. Returns the
    recommended date and the 3 best alternatives. `rationale`: plain English, why this date could rule a guess out."""
    return _call("plan_experiment", rationale=rationale)


def run_experiment(time: float = -1.0, rationale: str = "") -> dict:
    """FALSIFIER/PLANNER. Takes the measurement. time=-1 uses the recommended date; only dates returned by
    plan_experiment (allowed_times) are accepted. The simulator returns the data and code updates the probabilities."""
    return _call("run_experiment", time=time, rationale=rationale)


def assess_step(interpretation: str = "") -> dict:
    """SCIENTIST. Reads the result: status, guesses ruled out, whether the cause is identified, and what to do next
    (`next`). `interpretation`: 1-2 plain-English sentences citing numbers returned by the tools."""
    return _call("assess_step", interpretation=interpretation)


def run_replication(interpretation: str = "") -> dict:
    """REPLICATOR. Only available if assess_step returned identified=true. Tests the leading guess on new
    measurements that were never used in the discovery."""
    return _call("run_replication", interpretation=interpretation)


def finalize(summary: str = "") -> dict:
    """SCIENTIST. Closes the session and saves the result. Call after the double-check on new data, or when the
    measurement budget is used up."""
    return _call("finalize", summary=summary)


def get_status() -> dict:
    """Current session status (phase, next expected tool, chance of a hidden planet, leading guess)."""
    return _call("get_status")


def scout_note(open_problems: str = "", related_work: str = "", who_else: str = "", biases_and_risks: str = "") -> dict:
    """SCOUT. Saves the background report: open problems in the literature, related work, who else works on this,
    and the biases and risks of the idea. Plain English; mark anything unverified as 'unverified'."""
    return _call("scout_note", open_problems=open_problems, related_work=related_work,
                 who_else=who_else, biases_and_risks=biases_and_risks)
