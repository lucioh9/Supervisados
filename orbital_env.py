import numpy as np
from scipy.integrate import solve_ivp

# Units:
# distance = AU
# mass = solar masses
# time = years
G = 4 * np.pi**2

NOISE_STD = 1e-3
BASE_SEED = 42

SCENARIOS = [
    {"hidden_mass": 4e-5, "hidden_a": 1.45},
    {"hidden_mass": 5e-5, "hidden_a": 1.60},
    {"hidden_mass": 6e-5, "hidden_a": 1.75},
    {"hidden_mass": 7e-5, "hidden_a": 1.90},
]

def simulate_system(
    times,
    include_hidden=True,
    hidden_mass=5e-5,
    hidden_a=1.6,
):
    """
    Simulates star + 2 visible planets + optional hidden planet.
    Returns positions of all bodies at requested times.
    """

    masses = [1.0, 1e-5, 2e-5]
    radii = [0.0, 1.0, 2.3]

    if include_hidden:
        masses.append(hidden_mass)
        radii.append(hidden_a)

    n = len(masses)

    # Initial state: [x,y,vx,vy] for each body
    state = []

    for i, r in enumerate(radii):
        if i == 0:
            # star
            state.extend([0.0, 0.0, 0.0, 0.0])
        else:
            # approximately circular orbit
            v = np.sqrt(G * masses[0] / r)

            # Give each planet a different phase
            angle = i * 0.7

            x = r * np.cos(angle)
            y = r * np.sin(angle)

            vx = -v * np.sin(angle)
            vy = v * np.cos(angle)

            state.extend([x, y, vx, vy])

    state = np.array(state, dtype=float)
    masses = np.array(masses)

    def derivatives(t, y):
        dydt = np.zeros_like(y)

        positions = y.reshape(n, 4)[:, :2]
        velocities = y.reshape(n, 4)[:, 2:]

        accelerations = np.zeros((n, 2))

        for i in range(n):
            for j in range(n):
                if i == j:
                    continue

                diff = positions[j] - positions[i]
                dist = np.linalg.norm(diff) + 1e-12

                accelerations[i] += (
                    G * masses[j] * diff / dist**3
                )

        reshaped = dydt.reshape(n, 4)

        reshaped[:, :2] = velocities
        reshaped[:, 2:] = accelerations

        return dydt

    times = np.asarray(times)

    sol = solve_ivp(
        derivatives,
        (0, float(np.max(times))),
        state,
        t_eval=times,
        rtol=1e-9,
        atol=1e-11,
    )

    if not sol.success:
        raise RuntimeError(sol.message)

    # shape:
    # [time, body, (x,y,vx,vy)]
    return sol.y.T.reshape(len(times), n, 4)


def true_position(
    time,
    target="visible_1",
    hidden_mass=5e-5,
    hidden_a=1.6
):
    body_index = {
        "visible_1": 1,
        "visible_2": 2,
    }[target]

    data = simulate_system(
        [0.0, float(time)],
        include_hidden=True,
        hidden_mass=hidden_mass,
        hidden_a=hidden_a
    )

    return data[-1, body_index, :2]


def observe(
    time,
    target="visible_1",
    add_noise=True,
    hidden_mass=5e-5,
    hidden_a=1.6
):
    xy = true_position(
        time,
        target,
        hidden_mass=hidden_mass,
        hidden_a=hidden_a
    ).copy()

    if add_noise:
        seed = BASE_SEED + int(round(time * 10000))
        rng = np.random.default_rng(seed)
        xy += rng.normal(0, NOISE_STD, size=2)

    return {
        "time": float(time),
        "target": target,
        "x": float(xy[0]),
        "y": float(xy[1]),
        "noise_std": NOISE_STD,
    }


def predict_without_hidden(time, target="visible_1"):
    body_index = {
        "visible_1": 1,
        "visible_2": 2,
    }[target]

    data = simulate_system(
        [0.0, float(time)],
        include_hidden=False
    )

    xy = data[-1, body_index, :2]

    return {
        "time": float(time),
        "target": target,
        "x": float(xy[0]),
        "y": float(xy[1]),
    }


def predict_with_hidden_candidate(
    time,
    hidden_mass,
    hidden_a,
    target="visible_1",
):
    body_index = {
        "visible_1": 1,
        "visible_2": 2,
    }[target]

    data = simulate_system(
        [0.0, float(time)],
        include_hidden=True,
        hidden_mass=hidden_mass,
        hidden_a=hidden_a,
    )

    xy = data[-1, body_index, :2]

    return {
        "time": float(time),
        "target": target,
        "x": float(xy[0]),
        "y": float(xy[1]),
    }


def initial_observations(scenario_index=1):
    s = SCENARIOS[scenario_index]
    times = np.linspace(0.1, 2.0, 8)

    return [
        observe(
            float(t),
            hidden_mass=s["hidden_mass"],
            hidden_a=s["hidden_a"]
        )
        for t in times
    ]


def candidate_observation_times():
    return [
        float(t)
        for t in np.linspace(2.25, 8.0, 24)
    ]


def evaluation_truth(scenario_index=1):
    s = SCENARIOS[scenario_index]

    return {
        "hidden_body_exists": True,
        "hidden_mass": s["hidden_mass"],
        "hidden_a": s["hidden_a"],
    }


if __name__ == "__main__":
    print(initial_observations()[:2])


def replication_observation_times():
    """
    Holdout observations.
    These times should NOT be used during discovery.
    Only use them after a hypothesis has been selected.
    """
    return [
        float(t)
        for t in np.linspace(8.25, 10.0, 8)
    ]