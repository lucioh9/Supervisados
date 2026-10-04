import numpy as np
from orbital_env import (
    observe,
    predict_without_hidden,
    candidate_observation_times,
    NOISE_STD
)

results = []

for t in candidate_observation_times():
    obs = observe(t, add_noise=False)
    pred = predict_without_hidden(t)

    error = np.sqrt(
        (obs["x"] - pred["x"])**2 +
        (obs["y"] - pred["y"])**2
    )

    results.append((t, error))

results.sort(key=lambda x: x[1], reverse=True)

print("\nLargest discrepancies:")
for t, error in results[:10]:
    print(
        f"time={t:.2f} yr | "
        f"error={error:.6f} AU | "
        f"signal/noise={error/NOISE_STD:.1f} sigma"
    )

print("\nMAX SIGNAL/NOISE:", results[0][1] / NOISE_STD)