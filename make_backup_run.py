"""
Genera la corrida de respaldo para la demo (replay).
Es una ejecución REAL del motor; se elige la corrida MEDIANA (no la mejor) entre N semillas
para no hacer cherry-picking. Calienta también la caché de hipótesis (.cache/).

Uso:  python make_backup_run.py [--scenario 1] [--policy attn] [--seeds 20]
"""
import argparse
import numpy as np
import engine as eng

ap = argparse.ArgumentParser()
ap.add_argument("--scenario", type=int, default=1)
ap.add_argument("--policy", default="attn")
ap.add_argument("--seeds", type=int, default=20)
args = ap.parse_args()

bank = eng.get_bank(progress=lambda i, n: print(f"\rprecalculando hipótesis {i}/{n}", end=""))
print()
runs = []
for s in range(args.seeds):
    rec = eng.run_to_record(scenario_index=args.scenario, policy=args.policy, seed=s, bank=bank)
    ok = rec["outcome"]["identified"] and rec["evaluation"]["correct"] and rec["outcome"]["replication_passed"]
    runs.append((s, rec, ok))
    print(f"seed={s:2d} n_obs={rec['outcome']['n_observations']:2d} "
          f"identified={rec['outcome']['identified']} correct={rec['evaluation']['correct']} "
          f"replicated={rec['outcome']['replication_passed']}")

good = sorted([r for r in runs if r[2]], key=lambda r: r[1]["outcome"]["n_observations"])
if not good:
    raise SystemExit("Ninguna corrida identificó+replicó correctamente: revisar calibración (P1/P3).")
seed, rec, _ = good[len(good) // 2]
rec["meta"]["selection"] = (f"mediana de n_observations entre {len(good)} corridas exitosas "
                            f"de {args.seeds} semillas (seed elegida={seed})")
path = eng.save_record(rec, eng.RUNS_DIR / "backup_run.json")
print(f"\nGuardado {path}  |  éxito {len(good)}/{args.seeds}  |  {rec['meta']['selection']}")
