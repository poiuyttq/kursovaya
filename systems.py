import numpy as np


AXES = {
    "henon": {
        "x_label": "b",
        "y_label": "alfa",
        "title": "Henon-like map",
        "dimension": 2
    },
    "logistic": {
        "x_label": "r",
        "y_label": "x0",
        "title": "Logistic map",
        "dimension": 1
    },
    "power_abs": {
        "x_label": "mu",
        "y_label": "A",
        "title": "Power absolute map",
        "dimension": 1
    }
}


def step_scalar(system_name, state, params):
    if system_name == "henon":
        x, y = state
        return np.array([
            1 - params["alfa"] * x**2 - params["b"] * y,
            x
        ])

    if system_name == "logistic":
        x = state[0]
        return np.array([params["r"] * x * (1 - x)])

    if system_name == "power_abs":
        x = state[0]
        return np.array([
            -params["mu"]
            + params["A"] * abs(x) ** params["nu"]
            + params["C"] * abs(x) ** (2 * params["nu"])
        ])

    raise ValueError(f"Unknown map '{system_name}'.")


def step_vector(system_name, state, x_grid, y_grid, params):
    if system_name == "henon":
        x, y = state
        b = x_grid
        alfa = y_grid
        return (
            1 - alfa * x**2 - b * y,
            x,
        )

    if system_name == "logistic":
        x = state[0]
        r = x_grid
        return (r * x * (1 - x),)

    if system_name == "power_abs":
        x = state[0]
        mu = x_grid
        a_value = y_grid
        nu = params["nu"]
        c_value = params["C"]
        return (
            -mu + a_value * np.abs(x) ** nu + c_value * np.abs(x) ** (2 * nu),
        )

    raise ValueError(f"Unknown map '{system_name}'.")


def scalar_params_from_axes(system_name, x_value, y_value, base_params):
    params = dict(base_params)

    if system_name == "henon":
        params["b"] = x_value
        params["alfa"] = y_value
    elif system_name == "logistic":
        params["r"] = x_value
    elif system_name == "power_abs":
        params["mu"] = x_value
        params["A"] = y_value
    else:
        raise ValueError(f"Unknown map '{system_name}'.")

    return params


def scalar_initial_state_from_axes(system_name, y_value, initial_state):
    if system_name == "henon":
        return np.array([initial_state["x"], initial_state["y"]], dtype=float)

    if system_name == "logistic":
        return np.array([y_value], dtype=float)

    if system_name == "power_abs":
        return np.array([initial_state["x"]], dtype=float)

    raise ValueError(f"Unknown map '{system_name}'.")


def vector_initial_state_from_axes(system_name, y_grid, initial_state):
    shape = y_grid.shape

    if system_name == "henon":
        return (
            np.full(shape, initial_state["x"], dtype=float),
            np.full(shape, initial_state["y"], dtype=float),
        )

    if system_name == "logistic":
        return (y_grid.astype(float).copy(),)

    if system_name == "power_abs":
        return (np.full(shape, initial_state["x"], dtype=float),)

    raise ValueError(f"Unknown map '{system_name}'.")


def trajectory_initial_state(system_name, initial_state):
    if system_name == "henon":
        return np.array([initial_state["x"], initial_state["y"]], dtype=float)

    if system_name in {"logistic", "power_abs"}:
        return np.array([initial_state["x"]], dtype=float)

    raise ValueError(f"Unknown map '{system_name}'.")


def generate_trajectory(system_name, params, initial_state, steps):
    state = trajectory_initial_state(system_name, initial_state)
    trajectory = np.zeros((steps, len(state)), dtype=float)
    trajectory[0] = state

    for index in range(1, steps):
        trajectory[index] = step_scalar(system_name, trajectory[index - 1], params)

        if not np.all(np.isfinite(trajectory[index])):
            trajectory[index] = 0.0

    return trajectory
