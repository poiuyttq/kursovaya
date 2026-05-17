from concurrent.futures import ProcessPoolExecutor
import os

import numpy as np

from systems import (
    scalar_initial_state_from_axes,
    scalar_params_from_axes,
    step_scalar,
    step_vector,
    vector_initial_state_from_axes,
)


DIVERGENCE_CODE = -1
CHAOS_CODE = 0


def classify_dynamic_regime_for_cell(task):
    (
        system_name,
        x_value,
        y_value,
        params,
        initial_state,
        transient_steps,
        sample_steps,
        divergence_check_steps,
        max_period,
        divergence_limit,
        tolerance,
    ) = task

    cell_params = scalar_params_from_axes(system_name, x_value, y_value, params)
    state = scalar_initial_state_from_axes(system_name, y_value, initial_state)
    points = []
    total_steps = transient_steps + sample_steps + divergence_check_steps

    for step in range(total_steps):
        state = step_scalar(system_name, state, cell_params)

        if (
            not np.all(np.isfinite(state))
            or np.any(np.abs(state) >= divergence_limit)
        ):
            return DIVERGENCE_CODE

        if transient_steps <= step < transient_steps + sample_steps:
            points.append(state.copy())

    points = np.array(points)

    for period in range(1, min(max_period, sample_steps - 1) + 1):
        distances = np.linalg.norm(points[period:] - points[:-period], axis=1)
        if np.max(distances) < tolerance:
            return period

    return CHAOS_CODE


def classify_dynamic_regime_for_row(task):
    (
        row_index,
        system_name,
        x_values,
        y_value,
        params,
        initial_state,
        transient_steps,
        sample_steps,
        divergence_check_steps,
        max_period,
        divergence_limit,
        tolerance,
    ) = task

    row = [
        classify_dynamic_regime_for_cell(
            (
                system_name,
                x_value,
                y_value,
                params,
                initial_state,
                transient_steps,
                sample_steps,
                divergence_check_steps,
                max_period,
                divergence_limit,
                tolerance,
            )
        )
        for x_value in x_values
    ]

    return row_index, row


def calculate_dynamic_regime_map(
    system_name,
    params,
    initial_state,
    map_config,
    compute_method,
):
    x_values = np.linspace(
        map_config["x_min"],
        map_config["x_max"],
        map_config["x_steps"],
    )
    y_values = np.linspace(
        map_config["y_min"],
        map_config["y_max"],
        map_config["y_steps"],
    )

    if compute_method == "vector":
        regimes = calculate_dynamic_regime_map_vector(
            system_name,
            params,
            initial_state,
            map_config,
            x_values,
            y_values,
        )
    elif compute_method == "parallel":
        regimes = calculate_dynamic_regime_map_parallel(
            system_name,
            params,
            initial_state,
            map_config,
            x_values,
            y_values,
        )
    else:
        raise ValueError(f"Unknown compute method '{compute_method}'.")

    return x_values, y_values, regimes


def calculate_dynamic_regime_map_vector(
    system_name,
    params,
    initial_state,
    map_config,
    x_values,
    y_values,
):
    x_grid, y_grid = np.meshgrid(x_values, y_values)
    state = vector_initial_state_from_axes(system_name, y_grid, initial_state)
    shape = x_grid.shape
    dimension = len(state)

    transient_steps = map_config["transient_steps"]
    sample_steps = map_config["sample_steps"]
    divergence_check_steps = map_config["divergence_check_steps"]
    max_period = map_config["max_period"]
    divergence_limit = map_config["divergence_limit"]
    tolerance = map_config["tolerance"]
    show_progress = map_config.get("show_progress", True)

    diverged = np.zeros(shape, dtype=bool)
    history = np.zeros((max_period, dimension, *shape), dtype=float)
    max_distances = np.zeros((max_period, *shape), dtype=float)

    total_steps = transient_steps + sample_steps + divergence_check_steps
    progress_step = max(1, total_steps // 10)

    for step in range(total_steps):
        with np.errstate(over="ignore", invalid="ignore"):
            state = step_vector(system_name, state, x_grid, y_grid, params)

        bad_points = np.zeros(shape, dtype=bool)
        for component in state:
            bad_points |= ~np.isfinite(component)
            bad_points |= np.abs(component) >= divergence_limit

        diverged |= bad_points
        state = tuple(np.where(diverged, 0.0, component) for component in state)

        sample_index = step - transient_steps
        if 0 <= sample_index < sample_steps:
            periods_to_check = min(max_period, sample_index)

            for period in range(1, periods_to_check + 1):
                history_index = (sample_index - period) % max_period
                squared_distance = np.zeros(shape, dtype=float)

                for dimension_index, component in enumerate(state):
                    difference = component - history[history_index, dimension_index]
                    squared_distance += difference * difference

                distances = np.sqrt(squared_distance)
                max_distances[period - 1] = np.maximum(
                    max_distances[period - 1],
                    np.where(diverged, 0.0, distances),
                )

            history_index = sample_index % max_period
            for dimension_index, component in enumerate(state):
                history[history_index, dimension_index] = component

        if show_progress and (
            step == 0
            or (step + 1) % progress_step == 0
            or step == total_steps - 1
        ):
            progress = 100 * (step + 1) / total_steps
            print(f"Vector regime map progress: {progress:.0f}%")

    regimes = np.zeros(shape, dtype=int)
    regimes[diverged] = DIVERGENCE_CODE

    for period in range(1, min(max_period, sample_steps - 1) + 1):
        has_period = (
            ~diverged
            & (regimes == CHAOS_CODE)
            & (max_distances[period - 1] < tolerance)
        )
        regimes[has_period] = period

    return regimes


def calculate_dynamic_regime_map_parallel(
    system_name,
    params,
    initial_state,
    map_config,
    x_values,
    y_values,
):
    workers = map_config.get("workers") or os.cpu_count() or 1
    tasks = [
        (
            row_index,
            system_name,
            x_values,
            y_value,
            params,
            initial_state,
            map_config["transient_steps"],
            map_config["sample_steps"],
            map_config["divergence_check_steps"],
            map_config["max_period"],
            map_config["divergence_limit"],
            map_config["tolerance"],
        )
        for row_index, y_value in enumerate(y_values)
    ]

    show_progress = map_config.get("show_progress", True)
    regimes = np.zeros((len(y_values), len(x_values)), dtype=int)
    completed_rows = 0
    progress_step = max(1, len(y_values) // 10)

    print(
        "Parallel regime map calculation: "
        f"workers={workers}, rows={len(y_values)}, columns={len(x_values)}"
    )

    with ProcessPoolExecutor(max_workers=workers) as executor:
        for row_index, row in executor.map(
            classify_dynamic_regime_for_row,
            tasks,
            chunksize=1,
        ):
            regimes[row_index] = row
            completed_rows += 1

            if show_progress and (
                completed_rows == 1
                or completed_rows % progress_step == 0
                or completed_rows == len(y_values)
            ):
                progress = 100 * completed_rows / len(y_values)
                print(f"Parallel regime map progress: {progress:.0f}%")

    return regimes
