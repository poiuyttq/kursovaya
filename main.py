import sys

from config import load_config
from core import DynamicSystemProject


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "--ui":
        from ui import run_ui

        run_ui()
        return

    config_path = sys.argv[1] if len(sys.argv) > 1 else "config.json"
    config = load_config(config_path)
    project = DynamicSystemProject(config)
    project.run_all()


if __name__ == "__main__":
    main()
