"""
Benchmark ATTN vs random vs greedy (Persona 3). 100 % codigo: NO usa LLM, NO gasta creditos.

Uso:   python bench.py            (30 semillas por escenario, ~10 s)
       python bench.py 100        (100 semillas por escenario)

Salidas (carpeta runs/):
  benchmark.csv            una fila por corrida
  benchmark_summary.json   numeros finales (los que van al pitch)
  benchmark.png            grafica

Metrica principal: numero de observaciones (experimentos) hasta declarar la causa oculta.
Una corrida cuenta como EXITO solo si la hipotesis es correcta (ground truth, usada SOLO aqui) Y la replicacion holdout pasa.
Las corridas pares comparten semilla (mismo ruido) para las tres politicas; el IC95% es bootstrap pareado.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

import engine as eng
import orbital_env as env

POLICIES = ["random", "greedy", "attn"]
NAMES = {"random": "Random", "greedy": "Biggest error", "attn": "LYDH"}
N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 30
OUT = Path("runs")
OUT.mkdir(exist_ok=True)


def run_all(n_seeds):
    bank = eng.get_bank()
    rows = []
    for sc in range(len(env.SCENARIOS)):
        for pol in POLICIES:
            for r in eng.quick_compare(sc, [pol], range(n_seeds), bank):
                rows.append(dict(r, scenario=sc))
    df = pd.DataFrame(rows)
    df["success"] = df["correct"] & df["replicated"]
    return df


def boot_ratio(piv, num, den, n=4000, seed=0):
    """Razon de medias num/den con bootstrap pareado sobre (escenario, semilla)."""
    rng = np.random.default_rng(seed)
    a, b = piv[num].to_numpy(float), piv[den].to_numpy(float)
    idx = rng.integers(0, len(a), size=(n, len(a)))
    r = a[idx].mean(axis=1) / b[idx].mean(axis=1)
    return float(a.mean() / b.mean()), float(np.percentile(r, 2.5)), float(np.percentile(r, 97.5))


def summarize(df):
    out = {"n_seeds_per_scenario": N_SEEDS, "n_scenarios": int(df.scenario.nunique()), "policies": {}, "ratios": {}, "by_scenario": {}}
    for pol, g in df.groupby("policy"):
        out["policies"][pol] = {
            "runs": int(len(g)), "mean_obs": float(g.n_observations.mean()), "median_obs": float(g.n_observations.median()),
            "success_rate": float(g.success.mean()), "correct_rate": float(g.correct.mean()),
            "replication_pass_rate": float(g.replicated.mean()),
        }
    piv = df.pivot_table(index=["scenario", "seed"], columns="policy", values="n_observations")
    for base in ("random", "greedy"):
        m, lo, hi = boot_ratio(piv, base, "attn")
        out["ratios"][f"{base}_over_attn"] = {"ratio": m, "ci95": [lo, hi]}
    for sc, g in df.groupby("scenario"):
        p = g.pivot_table(index="seed", columns="policy", values="n_observations")
        out["by_scenario"][int(sc)] = {
            "truth": env.SCENARIOS[int(sc)],
            **{pol: float(p[pol].mean()) for pol in POLICIES},
            "random_over_attn": float(p["random"].mean() / p["attn"].mean()),
        }
    return out


def plot(df, summ):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = {"random": "#9aa0a6", "greedy": "#e8a33d", "attn": "#2f7fd1"}
    # 1) media de observaciones por politica con IC bootstrap simple
    rng = np.random.default_rng(1)
    for i, pol in enumerate(POLICIES):
        v = df[df.policy == pol].n_observations.to_numpy(float)
        bs = [rng.choice(v, len(v)).mean() for _ in range(2000)]
        ax[0].bar(i, v.mean(), color=cols[pol], yerr=[[v.mean() - np.percentile(bs, 2.5)], [np.percentile(bs, 97.5) - v.mean()]], capsize=4)
    ax[0].set_xticks(range(3)); ax[0].set_xticklabels([NAMES[p] for p in POLICIES])
    ax[0].set_ylabel("measurements needed to find the planet (average, 95% range)"); ax[0].set_title("How many measurements?\n(lower is better)")
    # 2) por escenario
    w = 0.27
    for k, pol in enumerate(POLICIES):
        vals = [summ["by_scenario"][s][pol] for s in sorted(summ["by_scenario"])]
        ax[1].bar(np.arange(len(vals)) + (k - 1) * w, vals, w, color=cols[pol], label=NAMES[pol])
    ax[1].set_xticks(range(len(vals))); ax[1].set_xticklabels([f"case {s}" for s in sorted(summ["by_scenario"])])
    ax[1].set_title("By test case\n(harder to the right)"); ax[1].legend()
    # 3) tasa de exito (correcta + replicada)
    sr = [summ["policies"][p]["success_rate"] * 100 for p in POLICIES]
    ax[2].bar(range(3), sr, color=[cols[p] for p in POLICIES]); ax[2].set_ylim(0, 105)
    ax[2].set_xticks(range(3)); ax[2].set_xticklabels([NAMES[p] for p in POLICIES])
    ax[2].set_ylabel("% of runs that found the right planet\nand passed the double-check"); ax[2].set_title("Success rate")
    for i, v in enumerate(sr):
        ax[2].text(i, v + 1.5, f"{v:.0f}%", ha="center")
    fig.tight_layout(); fig.savefig(OUT / "benchmark.png", dpi=150)


if __name__ == "__main__":
    df = run_all(N_SEEDS)
    df.to_csv(OUT / "benchmark.csv", index=False)
    summ = summarize(df)
    (OUT / "benchmark_summary.json").write_text(json.dumps(summ, indent=2), encoding="utf-8")
    plot(df, summ)
    print(f"\n{summ['n_scenarios']} escenarios x {N_SEEDS} semillas x 3 politicas = {len(df)} corridas\n")
    print(f"{'politica':8} {'obs media':>9} {'mediana':>8} {'exito':>7} {'replica':>8}")
    for p in POLICIES:
        s = summ["policies"][p]
        print(f"{p:8} {s['mean_obs']:9.2f} {s['median_obs']:8.1f} {s['success_rate']*100:6.0f}% {s['replication_pass_rate']*100:7.0f}%")
    for b in ("random", "greedy"):
        r = summ["ratios"][f"{b}_over_attn"]
        print(f"\n{b} / attn = {r['ratio']:.2f}x  (IC95% {r['ci95'][0]:.2f} - {r['ci95'][1]:.2f})")
    print("\nPor escenario (random / attn):",
          {s: round(v["random_over_attn"], 2) for s, v in summ["by_scenario"].items()})
    print("\nGuardado: runs/benchmark.csv, runs/benchmark_summary.json, runs/benchmark.png")
