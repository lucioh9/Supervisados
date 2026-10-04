"""
Curva de escala (para el pitch): ¿crece la ventaja de ATTN al crecer el espacio de hipotesis?
Uso:  python scale_curve.py            (rejillas 5,8,12,16  ~ 5-8 min la 1a vez, luego usa cache)
      python scale_curve.py 10         (semillas por escenario; por defecto 10)
Salida: runs/scale_curve.png  y  runs/scale_curve.json
"""
import json, sys
from pathlib import Path
import numpy as np
import engine as eng
import orbital_env as env

N_SEEDS = int(sys.argv[1]) if len(sys.argv) > 1 else 10
GRIDS = [5, 8, 12, 16]
POLS = ["random", "attn"]
OUT = Path("runs"); OUT.mkdir(exist_ok=True)

res = {}
for n in GRIDS:
    eng.MASSES = [float(x) for x in np.linspace(3e-5, 7e-5, n)]
    eng.SEMIAXES = [float(x) for x in np.linspace(1.30, 1.90, n)]
    eng.CLUSTER_RADIUS = max(1, round((n - 1) / 4))      # misma vecindad FISICA en todas las rejillas
    print(f"rejilla {n}x{n} = {n*n+1} hipotesis (construyendo/leyendo banco)...", flush=True)
    bank = eng.get_bank()
    row = {}
    for pol in POLS:
        obs, ok = [], []
        for sc in range(len(env.SCENARIOS)):
            for r in eng.quick_compare(sc, [pol], range(N_SEEDS), bank):
                obs.append(r["n_observations"]); ok.append(r["correct"] and r["replicated"])
        row[pol] = {"mean_obs": float(np.mean(obs)), "success": float(np.mean(ok))}
    row["ratio"] = row["random"]["mean_obs"] / row["attn"]["mean_obs"]
    res[n * n + 1] = row
    print(f"   random {row['random']['mean_obs']:.2f} | attn {row['attn']['mean_obs']:.2f} | random/attn = {row['ratio']:.2f}x", flush=True)

(OUT / "scale_curve.json").write_text(json.dumps(res, indent=2))
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
K = list(res)
fig, ax = plt.subplots(1, 2, figsize=(11, 4))
for pol, c in (("random", "#9aa0a6"), ("attn", "#2f7fd1")):
    ax[0].plot(K, [res[k][pol]["mean_obs"] for k in K], "o-", color=c, label={"random":"Random","attn":"LYDH"}[pol])
ax[0].set_xlabel("number of possible answers (guesses)"); ax[0].set_ylabel("measurements needed"); ax[0].legend(); ax[0].set_title("More possible answers, more measurements")
ax[1].plot(K, [res[k]["ratio"] for k in K], "o-", color="#2f7fd1"); ax[1].axhline(1, ls="--", color="gray")
ax[1].set_xlabel("number of possible answers (guesses)"); ax[1].set_ylabel("Random / LYDH (times fewer)"); ax[1].set_title("LYDH advantage as the problem grows")
fig.tight_layout(); fig.savefig(OUT / "scale_curve.png", dpi=150)
print("Guardado runs/scale_curve.png y runs/scale_curve.json")
