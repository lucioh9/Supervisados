"""Comprueba que mi capa (P4) es consistente con orbital_env.py de P1. Ejecutar antes de la demo."""
import numpy as np
import engine as eng
import orbital_env as env

# 1) tiempos iniciales idénticos a P1
t_p1 = [o["time"] for o in env.initial_observations(1)]
assert np.allclose(t_p1, eng.INITIAL_TIMES), "INITIAL_TIMES difiere de initial_observations()"

bank = eng.get_bank()
# 2) modelo nulo del banco == predict_without_hidden
t = env.candidate_observation_times()[10]
p = env.predict_without_hidden(t)
assert np.allclose(bank.V1[0, bank.idx(t)], [p["x"], p["y"]], atol=1e-6)

# 3) mundo batch == observe() de P1 (con el mismo ruido)
for sc in range(len(env.SCENARIOS)):
    s = env.SCENARIOS[sc]
    o1 = env.observe(t, hidden_mass=s["hidden_mass"], hidden_a=s["hidden_a"])
    o2 = eng.World(sc, bank, seed=None).observe(t)
    assert abs(o1["x"] - o2["x"]) < 1e-6 and abs(o1["y"] - o2["y"]) < 1e-6, f"scenario {sc} difiere"

# 4) contrato de eventos
for ev in eng.run_discovery(1, "attn", bank=bank):
    eng.validate_event(ev)
print("OK: P1 ↔ P4 consistentes y eventos válidos")
