from dataclasses import dataclass, field

from plotting import plot_logistic_bifurcation, plot_regime_map, plot_trajectory
from regimes import calculate_dynamic_regime_map
from systems import generate_trajectory


@dataclass
class ProjectRunResult:
    generated_items: list[str] = field(default_factory=list)


class DynamicSystemProject:
    def __init__(self, config):
        self.config = config
        self.system_name = config["map"]
        self.output_dir = config.get("output_dir", ".")

    def run_all(self):
        result = ProjectRunResult()

        if self.config.get("draw_trajectory", True):
            self.run_trajectory()
            result.generated_items.append("trajectory")

        if self.config.get("draw_regime_map", True):
            self.run_regime_map()
            result.generated_items.append("regime map")

        if (
            self.system_name == "logistic"
            and self.config.get("draw_bifurcation_diagram", False)
        ):
            self.run_bifurcation_diagram()
            result.generated_items.append("bifurcation diagram")

        return result

    def run_trajectory(self):
        trajectory = generate_trajectory(
            system_name=self.system_name,
            params=self.config["parameters"][self.system_name],
            initial_state=self.config["initial_state"],
            steps=self.config["trajectory"]["steps"],
        )
        return plot_trajectory(
            system_name=self.system_name,
            trajectory=trajectory,
            skip_points=self.config["trajectory"]["skip_points"],
            output_dir=self.output_dir,
        )

    def run_regime_map(self):
        x_values, y_values, regimes = calculate_dynamic_regime_map(
            system_name=self.system_name,
            params=self.config["parameters"][self.system_name],
            initial_state=self.config["initial_state"],
            map_config=self.config["regime_map"],
            compute_method=self.config["compute_method"],
        )
        return plot_regime_map(
            system_name=self.system_name,
            x_values=x_values,
            y_values=y_values,
            regimes=regimes,
            max_period=self.config["regime_map"]["max_period"],
            output_dir=self.output_dir,
        )

    def run_bifurcation_diagram(self):
        return plot_logistic_bifurcation(self.config, self.output_dir)
