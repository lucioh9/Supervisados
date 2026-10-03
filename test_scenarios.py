from orbital_env import (
    SCENARIOS,
    observe,
    predict_without_hidden,
    candidate_observation_times,
    NOISE_STD,
)
import numpy as np

for i, s in enumerate(SCENARIOS):
    max_ratio = 0

    for t in candidate_observation_times():
        obs = observe(
            t,
            add_noise=False,
            hidden_mass=s["hidden_mass"],
            hidden_a=s["hidden_a"]
        )

        pred = predict_without_hidden(t)

        error = np.sqrt(
            (obs["x"] - pred["x"])**2 +
            (obs["y"] - pred["y"])**2
        )

        ratio = error / NOISE_STD
        max_ratio = max(max_ratio, ratio)

    print(
        f"Scenario {i} | "
        f"mass={s['hidden_mass']} | "
        f"a={s['hidden_a']} | "
        f"max signal/noise={max_ratio:.2f}"
    )