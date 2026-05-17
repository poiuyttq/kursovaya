from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.patches import Patch

from regimes import CHAOS_CODE, DIVERGENCE_CODE
from systems import AXES


def format_number_for_filename(value):
    return str(value).replace("-", "m").replace(".", "p")


def safe_filename(name):
    return name.replace(" ", "_").replace("/", "_")


def plot_trajectory(system_name, trajectory, skip_points, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    clean_trajectory = np.nan_to_num(
        trajectory,
        nan=0.0,
        posinf=10.0,
        neginf=-10.0,
    )

    fig, ax = plt.subplots(figsize=(7, 7))

    if clean_trajectory.shape[1] == 1:
        ax.plot(
            np.arange(skip_points, len(clean_trajectory)),
            clean_trajectory[skip_points:, 0],
            color="black",
            linewidth=0.8,
        )
        ax.set_xlabel("n", fontsize=12)
        ax.set_ylabel("x", fontsize=12)
    else:
        ax.scatter(
            clean_trajectory[skip_points:, 0],
            clean_trajectory[skip_points:, 1],
            c="black",
            s=10,
            alpha=0.7,
            edgecolors="none",
        )
        ax.set_xlabel("x", fontsize=12)
        ax.set_ylabel("y", fontsize=12)

    title = AXES[system_name]["title"]
    ax.set_title(f"{title}: trajectory", fontsize=14, pad=16)
    ax.grid(True, alpha=0.3, linestyle="--")
    plt.tight_layout()

    filename = output_path / f"trajectory_{safe_filename(system_name)}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Trajectory plot saved as '{filename}'")

    return fig, ax


def period_colors(max_period):
    preferred = {
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
    for period in range(1, max_period + 1):
        colors.append(preferred.get(period, plt.cm.hsv(period / max_period)))

    return colors


def plot_regime_map(system_name, x_values, y_values, regimes, max_period, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    color_indexes = np.where(
        regimes == DIVERGENCE_CODE,
        0,
        np.where(regimes == CHAOS_CODE, 1, regimes + 1),
    )

    colors = period_colors(max_period)
    labels = ["divergence", "chaos / no period"]
    labels.extend([f"period {period}" for period in range(1, max_period + 1)])

    cmap = ListedColormap(colors)
    norm = BoundaryNorm(np.arange(-0.5, len(colors) + 0.5, 1), cmap.N)

    fig, ax = plt.subplots(figsize=(8, 7))
    ax.imshow(
        color_indexes,
        origin="lower",
        extent=[x_values[0], x_values[-1], y_values[0], y_values[-1]],
        aspect="auto",
        cmap=cmap,
        norm=norm,
        interpolation="nearest",
    )

    axes = AXES[system_name]
    ax.set_xlabel(axes["x_label"], fontsize=12)
    ax.set_ylabel(axes["y_label"], fontsize=12)
    ax.set_title(f"{axes['title']}: dynamic regime map", fontsize=14, pad=16)
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
    filename = output_path / f"regime_map_{safe_filename(system_name)}.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Regime map saved as '{filename}'")

    return fig, ax


def plot_logistic_bifurcation(config, output_dir):
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    bifurcation_config = config["bifurcation_diagram"]
    r_values = np.linspace(
        bifurcation_config["r_min"],
        bifurcation_config["r_max"],
        bifurcation_config["r_steps"],
    )
    x = np.full_like(r_values, bifurcation_config["x0"], dtype=float)

    saved_r = []
    saved_x = []
    total_steps = (
        bifurcation_config["transient_steps"]
        + bifurcation_config["saved_iterations"]
    )

    for step in range(total_steps):
        x = r_values * x * (1 - x)

        if step >= bifurcation_config["transient_steps"]:
            saved_r.append(r_values.copy())
            saved_x.append(x.copy())

    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(
        np.concatenate(saved_r),
        np.concatenate(saved_x),
        ",",
        color="black",
        alpha=0.35,
    )

    ax.set_xlabel("r", fontsize=12)
    ax.set_ylabel("x", fontsize=12)
    ax.set_title("Logistic map: bifurcation diagram", fontsize=14, pad=16)
    ax.set_xlim(bifurcation_config["r_min"], bifurcation_config["r_max"])
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.25, linestyle="--")

    plt.tight_layout()
    filename = output_path / "bifurcation_logistic.png"
    plt.savefig(filename, dpi=150, bbox_inches="tight")
    print(f"Bifurcation diagram saved as '{filename}'")

    return fig, ax
