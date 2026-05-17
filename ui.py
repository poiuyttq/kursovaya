import copy
import math
import threading
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

from config import DEFAULT_CONFIG, load_config, save_config, validate_config
from core import DynamicSystemProject


class DynamicSystemsUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Dynamic Systems Laboratory")
        self.root.geometry("860x720")
        self.config_path = tk.StringVar(value="config.json")
        self.status = tk.StringVar(value="Ready")
        self.variables = {}
        self.preview_image = None

        self._build_layout()
        self.load_config("config.json")

    def _build_layout(self):
        self.root.columnconfigure(0, weight=1)
        self.root.rowconfigure(1, weight=1)

        top = ttk.Frame(self.root, padding=12)
        top.grid(row=0, column=0, sticky="ew")
        top.columnconfigure(1, weight=1)

        ttk.Label(top, text="Config").grid(row=0, column=0, sticky="w")
        ttk.Entry(top, textvariable=self.config_path).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(top, text="Load", command=self._load_from_entry).grid(row=0, column=2, padx=4)
        ttk.Button(top, text="Browse", command=self._browse_config).grid(row=0, column=3, padx=4)
        ttk.Button(top, text="Save", command=self._save_current_config).grid(row=0, column=4, padx=4)

        notebook = ttk.Notebook(self.root)
        notebook.grid(row=1, column=0, sticky="nsew", padx=12, pady=4)

        self.general_frame = ttk.Frame(notebook, padding=12)
        self.trajectory_frame = ttk.Frame(notebook, padding=12)
        self.regime_frame = ttk.Frame(notebook, padding=12)
        self.parameters_frame = ttk.Frame(notebook, padding=12)
        self.preview_frame = ttk.Frame(notebook, padding=12)

        notebook.add(self.general_frame, text="General")
        notebook.add(self.trajectory_frame, text="Trajectory")
        notebook.add(self.regime_frame, text="Regime Map")
        notebook.add(self.parameters_frame, text="Parameters")
        notebook.add(self.preview_frame, text="Preview")

        self._build_general_tab()
        self._build_trajectory_tab()
        self._build_regime_tab()
        self._build_parameters_tab()
        self._build_preview_tab()

        bottom = ttk.Frame(self.root, padding=12)
        bottom.grid(row=2, column=0, sticky="ew")
        bottom.columnconfigure(0, weight=1)

        ttk.Label(bottom, textvariable=self.status).grid(row=0, column=0, sticky="w")
        ttk.Button(bottom, text="Run Project", command=self.run_project).grid(row=0, column=1, padx=6)
        ttk.Button(bottom, text="Quit", command=self.root.destroy).grid(row=0, column=2)

    def _build_general_tab(self):
        self._combo(self.general_frame, "map", "Map", ["henon", "logistic", "power_abs"], 0)
        self._combo(self.general_frame, "compute_method", "Compute method", ["vector", "parallel"], 1)
        self._entry(self.general_frame, "output_dir", "Output directory", 2)
        self._check(self.general_frame, "draw_trajectory", "Draw trajectory", 3)
        self._check(self.general_frame, "draw_regime_map", "Draw regime map", 4)
        self._check(
            self.general_frame,
            "draw_bifurcation_diagram",
            "Draw logistic bifurcation diagram",
            5,
        )

    def _build_trajectory_tab(self):
        self._entry(self.trajectory_frame, "trajectory.steps", "Steps", 0)
        self._entry(self.trajectory_frame, "trajectory.skip_points", "Skip points", 1)
        self._entry(self.trajectory_frame, "initial_state.x", "Initial x", 2)
        self._entry(self.trajectory_frame, "initial_state.y", "Initial y", 3)

    def _build_regime_tab(self):
        fields = [
            ("regime_map.x_min", "X min"),
            ("regime_map.x_max", "X max"),
            ("regime_map.y_min", "Y min"),
            ("regime_map.y_max", "Y max"),
            ("regime_map.x_steps", "X steps"),
            ("regime_map.y_steps", "Y steps"),
            ("regime_map.transient_steps", "Transient steps"),
            ("regime_map.sample_steps", "Sample steps"),
            ("regime_map.divergence_check_steps", "Divergence check steps"),
            ("regime_map.max_period", "Max period"),
            ("regime_map.divergence_limit", "Divergence limit"),
            ("regime_map.tolerance", "Tolerance"),
            ("regime_map.workers", "Parallel workers"),
        ]

        for row, (key, label) in enumerate(fields):
            self._entry(self.regime_frame, key, label, row)

    def _build_parameters_tab(self):
        fields = [
            ("parameters.henon.b", "Henon b"),
            ("parameters.henon.alfa", "Henon alfa"),
            ("parameters.logistic.r", "Logistic r"),
            ("parameters.power_abs.mu", "Power mu"),
            ("parameters.power_abs.A", "Power A"),
            ("parameters.power_abs.nu", "Power nu"),
            ("parameters.power_abs.C", "Power C"),
            ("bifurcation_diagram.r_min", "Bifurcation r min"),
            ("bifurcation_diagram.r_max", "Bifurcation r max"),
            ("bifurcation_diagram.r_steps", "Bifurcation r steps"),
            ("bifurcation_diagram.x0", "Bifurcation x0"),
            ("bifurcation_diagram.transient_steps", "Bifurcation transient"),
            ("bifurcation_diagram.saved_iterations", "Bifurcation saved iterations"),
        ]

        for row, (key, label) in enumerate(fields):
            self._entry(self.parameters_frame, key, label, row)

    def _build_preview_tab(self):
        self.preview_frame.columnconfigure(0, weight=1)
        self.preview_frame.rowconfigure(1, weight=1)

        ttk.Label(
            self.preview_frame,
            text="The latest regime map will appear here after calculation.",
        ).grid(row=0, column=0, sticky="w", pady=(0, 8))

        self.preview_label = ttk.Label(
            self.preview_frame,
            text="Regime map has not been generated yet.",
            anchor="center",
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew")

    def _entry(self, parent, key, label, row):
        variable = tk.StringVar()
        self.variables[key] = variable
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Entry(parent, textvariable=variable).grid(row=row, column=1, sticky="ew", pady=4)
        parent.columnconfigure(1, weight=1)

    def _combo(self, parent, key, label, values, row):
        variable = tk.StringVar()
        self.variables[key] = variable
        ttk.Label(parent, text=label).grid(row=row, column=0, sticky="w", pady=4)
        ttk.Combobox(parent, textvariable=variable, values=values, state="readonly").grid(
            row=row,
            column=1,
            sticky="ew",
            pady=4,
        )
        parent.columnconfigure(1, weight=1)

    def _check(self, parent, key, label, row):
        variable = tk.BooleanVar()
        self.variables[key] = variable
        ttk.Checkbutton(parent, text=label, variable=variable).grid(
            row=row,
            column=0,
            columnspan=2,
            sticky="w",
            pady=4,
        )

    def _browse_config(self):
        path = filedialog.askopenfilename(
            filetypes=[("JSON config", "*.json"), ("All files", "*.*")]
        )
        if path:
            self.config_path.set(path)
            self.load_config(path)

    def _load_from_entry(self):
        self.load_config(self.config_path.get())

    def load_config(self, path):
        try:
            config = load_config(path)
            self._apply_config(config)
            self.status.set(f"Loaded {path}")
        except Exception as error:
            messagebox.showerror("Config error", str(error))

    def _save_current_config(self):
        try:
            config = self._collect_config()
            validate_config(config)
            save_config(config, self.config_path.get())
            self.status.set(f"Saved {self.config_path.get()}")
        except Exception as error:
            messagebox.showerror("Save error", str(error))

    def run_project(self):
        try:
            config = self._collect_config()
            validate_config(config)
            save_config(config, "config_ui.json")
        except Exception as error:
            messagebox.showerror("Run error", str(error))
            return

        self.status.set("Running... output config saved to config_ui.json")
        thread = threading.Thread(target=self._run_project_thread, args=(config,), daemon=True)
        thread.start()

    def _run_project_thread(self, config):
        try:
            project = DynamicSystemProject(config)
            result = project.run_all()
            message = "Finished: " + ", ".join(result.generated_items)
            self.root.after(0, lambda: self._finish_project_run(config, message))
        except Exception as error:
            error_message = str(error)
            self.root.after(
                0,
                lambda: messagebox.showerror("Runtime error", error_message),
            )
            self.root.after(0, lambda: self.status.set("Failed"))

    def _finish_project_run(self, config, message):
        self.status.set(message)

        if config.get("draw_regime_map", True):
            self._show_regime_map_preview(config)
        else:
            self.preview_label.configure(
                image="",
                text="Regime map was not generated: draw_regime_map = false.",
            )
            self.preview_image = None

    def _show_regime_map_preview(self, config):
        system_name = config["map"]
        output_dir = Path(config.get("output_dir", "."))
        image_path = output_dir / f"regime_map_{self._safe_filename(system_name)}.png"

        if not image_path.exists():
            self.preview_label.configure(
                image="",
                text=f"Regime map file was not found:\n{image_path}",
            )
            self.preview_image = None
            return

        image = tk.PhotoImage(file=str(image_path))
        width = image.width()
        height = image.height()
        max_width = 760
        max_height = 520
        scale = max(1, math.ceil(max(width / max_width, height / max_height)))

        if scale > 1:
            image = image.subsample(scale, scale)

        self.preview_image = image
        self.preview_label.configure(image=self.preview_image, text="")

    def _safe_filename(self, name):
        return name.replace(" ", "_").replace("/", "_")

    def _apply_config(self, config):
        merged = self._with_bifurcation_defaults(config)

        for key, variable in self.variables.items():
            value = self._get_nested(merged, key)
            if isinstance(variable, tk.BooleanVar):
                variable.set(bool(value))
            else:
                variable.set("" if value is None else str(value))

    def _collect_config(self):
        config = self._with_bifurcation_defaults(copy.deepcopy(DEFAULT_CONFIG))

        for key, variable in self.variables.items():
            old_value = self._get_nested(config, key)
            raw_value = variable.get()

            if isinstance(variable, tk.BooleanVar):
                value = bool(raw_value)
            else:
                value = self._parse_value(raw_value, old_value)

            self._set_nested(config, key, value)

        return config

    def _with_bifurcation_defaults(self, config):
        config = copy.deepcopy(config)
        config.setdefault("draw_bifurcation_diagram", False)
        config.setdefault(
            "bifurcation_diagram",
            {
                "r_min": 0.0,
                "r_max": 4.0,
                "r_steps": 3000,
                "x0": 0.2,
                "transient_steps": 1000,
                "saved_iterations": 250,
            },
        )
        return config

    def _parse_value(self, raw_value, old_value):
        if isinstance(old_value, bool):
            return raw_value.lower() in {"true", "1", "yes"}
        if isinstance(old_value, int) and not isinstance(old_value, bool):
            return int(raw_value)
        if isinstance(old_value, float):
            return float(raw_value)
        if old_value is None:
            return None if raw_value.strip().lower() == "none" else raw_value
        return raw_value

    def _get_nested(self, config, key):
        value = config
        for part in key.split("."):
            value = value[part]
        return value

    def _set_nested(self, config, key, value):
        target = config
        parts = key.split(".")
        for part in parts[:-1]:
            target = target.setdefault(part, {})
        target[parts[-1]] = value


def run_ui():
    root = tk.Tk()
    app = DynamicSystemsUI(root)
    root.mainloop()
    return app


if __name__ == "__main__":
    run_ui()
