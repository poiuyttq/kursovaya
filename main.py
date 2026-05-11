import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch


DIVERGENCE_CODE = -1
CHAOS_CODE = 0


def format_number_for_filename(value):
    return str(value).replace("-", "m").replace(".", "p")


def next_point(x, y, b, alfa):
    """One iteration of the map."""
    return 1 - alfa * x**2 - b * y, x


def generate_trajectory(b, alfa, n_points=100000, x0=0.0, y0=0.0):
    x = np.zeros(n_points)
    y = np.zeros(n_points)
    x[0] = x0
    y[0] = y0

    for i in range(1, n_points):
        x[i], y[i] = next_point(x[i - 1], y[i - 1], b, alfa)

        if not np.isfinite(x[i]) or not np.isfinite(y[i]):
            print(f"Overflow on step {i}: x={x[i]}, y={y[i]}")
            x[i] = 0.0
            y[i] = 0.0

    return x, y


def plot_trajectory(b, alfa, n_points=100000, skip_points=100):
    x, y = generate_trajectory(b, alfa, n_points=n_points)

    x_clean = np.nan_to_num(x, nan=0.0, posinf=10.0, neginf=-10.0)
    y_clean = np.nan_to_num(y, nan=0.0, posinf=10.0, neginf=-10.0)

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.scatter(
        x_clean[skip_points:],
        y_clean[skip_points:],
        c="black",
        s=10,
        alpha=0.7,
        edgecolors="none",
    )

    x_min, x_max = x_clean.min(), x_clean.max()
    y_min, y_max = y_clean.min(), y_clean.max()

    x_range = x_max - x_min
    y_range = y_max - y_min
    if x_range == 0:
        x_range = 2.0
    if y_range == 0:
        y_range = 2.0

    ax.set_xlim(x_min - 0.1 * x_range, x_max + 0.1 * x_range)
    ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)

    ax.set_xlabel("x", fontsize=12)
    ax.set_ylabel("y", fontsize=12)
    ax.set_title(
        f"Dynamic system: b={b}, alfa={alfa}",
        fontsize=14,
        pad=20,
    )
    ax.grid(True, alpha=0.3, linestyle="--")

    plt.tight_layout()

    filename = f"scatter_b{b}_a{alfa}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Trajectory plot saved as '{filename}'")

    return fig, ax


def classify_dynamic_regime(
    b,
    alfa,
    x0=0.0,
    y0=0.0,
    transient_steps=500,
    sample_steps=300,
    max_period=12,
    divergence_limit=10.0,
    tolerance=1e-4,
):
    """Return period number, CHAOS_CODE, or DIVERGENCE_CODE for one (b, alfa)."""
    x = x0
    y = y0
    points = []
    total_steps = transient_steps + sample_steps

    for step in range(total_steps):
        x, y = next_point(x, y, b, alfa)

        if (
            not np.isfinite(x)
            or not np.isfinite(y)
            or abs(x) >= divergence_limit
            or abs(y) >= divergence_limit
        ):
            return DIVERGENCE_CODE

        if step >= transient_steps:
            points.append((x, y))

    points = np.array(points)

    for period in range(1, max_period + 1):
        if len(points) <= period:
            break

        distances = np.linalg.norm(points[period:] - points[:-period], axis=1)
        if np.max(distances) < tolerance:
            return period

    return CHAOS_CODE


def calculate_dynamic_regime_map(
    b_min=-0.6,
    b_max=0.6,
    alfa_min=0.0,
    alfa_max=2.0,
    b_steps=220,
    alfa_steps=220,
    show_progress=True,
    x0=0.0,
    y0=0.0,
    transient_steps=500,
    sample_steps=300,
    max_period=12,
    divergence_limit=10.0,
    tolerance=1e-4,
):
    b_values = np.linspace(b_min, b_max, b_steps)
    alfa_values = np.linspace(alfa_min, alfa_max, alfa_steps)

    b_grid, alfa_grid = np.meshgrid(b_values, alfa_values)
    x = np.full((alfa_steps, b_steps), x0, dtype=float)
    y = np.full((alfa_steps, b_steps), y0, dtype=float)

    diverged = np.zeros((alfa_steps, b_steps), dtype=bool)

    history_x = np.zeros((max_period, alfa_steps, b_steps), dtype=float)
    history_y = np.zeros((max_period, alfa_steps, b_steps), dtype=float)
    max_distances = np.zeros((max_period, alfa_steps, b_steps), dtype=float)

    total_steps = transient_steps + sample_steps
    progress_step = max(1, total_steps // 10)

    for step in range(total_steps):
        previous_x = x

        with np.errstate(over="ignore", invalid="ignore"):
            x = 1 - alfa_grid * x**2 - b_grid * y
        y = previous_x

        bad_points = (
            ~np.isfinite(x)
            | ~np.isfinite(y)
            | (np.abs(x) >= divergence_limit)
            | (np.abs(y) >= divergence_limit)
        )
        diverged |= bad_points

        # Keep already-diverged cells numerically harmless for the next vector step.
        x = np.where(diverged, 0.0, x)
        y = np.where(diverged, 0.0, y)

        sample_index = step - transient_steps
        if sample_index >= 0:
            periods_to_check = min(max_period, sample_index)

            for period in range(1, periods_to_check + 1):
                history_index = (sample_index - period) % max_period
                dx = x - history_x[history_index]
                dy = y - history_y[history_index]
                distances = np.sqrt(dx * dx + dy * dy)
                max_distances[period - 1] = np.maximum(
                    max_distances[period - 1],
                    np.where(diverged, 0.0, distances),
                )

            history_index = sample_index % max_period
            history_x[history_index] = x
            history_y[history_index] = y

        if show_progress and (
            step == 0
            or (step + 1) % progress_step == 0
            or step == total_steps - 1
        ):
            progress = 100 * (step + 1) / total_steps
            print(f"Regime map progress: {progress:.0f}%")

    regimes = np.zeros((alfa_steps, b_steps), dtype=int)
    regimes[diverged] = DIVERGENCE_CODE

    for period in range(1, min(max_period, sample_steps - 1) + 1):
        has_period = (
            ~diverged
            & (regimes == CHAOS_CODE)
            & (max_distances[period - 1] < tolerance)
        )
        regimes[has_period] = period

    return b_values, alfa_values, regimes


def plot_dynamic_regime_map(
    b_min=-0.6,
    b_max=0.6,
    alfa_min=0.0,
    alfa_max=2.0,
    b_steps=220,
    alfa_steps=220,
    max_period=4,
    filename="dynamic_regime_map.png",
    show_progress=True,
    **classification_kwargs,
):
    b_values, alfa_values, regimes = calculate_dynamic_regime_map(
        b_min=b_min,
        b_max=b_max,
        alfa_min=alfa_min,
        alfa_max=alfa_max,
        b_steps=b_steps,
        alfa_steps=alfa_steps,
        show_progress=show_progress,
        max_period=max_period,
        **classification_kwargs,
    )

    # 0 is divergence, 1 is chaos/aperiodic, 2.. are periods 1..max_period.
    color_indexes = np.where(
        regimes == DIVERGENCE_CODE,
        0,
        np.where(regimes == CHAOS_CODE, 1, regimes + 1),
    )

    period_color_map = {
        1: "#0047b3",
        2: "#17a51f",
        3: "#ffb000",
        4: "#d40000",
        5: "#cc33cc",
        6: "#7a4a32",
        7: "#00a6b2",
        8: "#ff8c33",
        9: "#8f8f8f",
        10: "#9acd32",
        11: "#a6c8ff",
        12: "#1f77b4",
    }
    colors = ["#b8b8b8", "#ffffff"]
    colors.extend(
        period_color_map.get(period, plt.cm.tab20(period % 20))
        for period in range(1, max_period + 1)
    )
    labels = ["divergence", "chaos / no period"]
    labels.extend([f"period {period}" for period in range(1, max_period + 1)])

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(
        color_indexes,
        origin="lower",
        extent=[b_values[0], b_values[-1], alfa_values[0], alfa_values[-1]],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    ax.set_xlabel("b", fontsize=12)
    ax.set_ylabel("alfa", fontsize=12)
    ax.set_title("Dynamic regime map", fontsize=14, pad=16)
    ax.grid(False)

    present_indexes = sorted(np.unique(color_indexes))
    legend_items = [
        Patch(facecolor=colors[index], edgecolor="black", label=labels[index])
        for index in present_indexes
    ]
    ax.legend(
        handles=legend_items,
        loc="center left",
        bbox_to_anchor=(1.02, 0.5),
        frameon=True,
    )

    plt.tight_layout()
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Dynamic regime map saved as '{filename}'")

    return fig, ax, regimes


def ask_yes_no(prompt, default=True):
    default_text = "Y/n" if default else "y/N"

    try:
        answer = input(f"{prompt} [{default_text}]: ").strip().lower()
    except EOFError:
        return default

    if not answer:
        return default

    return answer in ("y", "yes", "d", "da", "д", "да")


def main():
    b = float(input("Enter parameter b, for example 0.2: "))
    alfa = float(input("Enter parameter alfa, for example 1.4: "))

    plot_trajectory(b, alfa)

    if ask_yes_no("Build colored dynamic regime map", default=True):
        map_filename = (
            "dynamic_regime_map_"
            f"b{format_number_for_filename(b)}_"
            f"a{format_number_for_filename(alfa)}.png"
        )
        plot_dynamic_regime_map(
            max_period=4,
            filename=map_filename,
        )

    plt.show()


if __name__ == "__main__":
    main()
