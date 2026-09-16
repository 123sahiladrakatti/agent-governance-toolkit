"""Launch the shared manager-facing AML governance UI."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

workspace_root = Path(__file__).resolve().parents[5]
scenario_root = Path(__file__).resolve().parent
sys.path.insert(0, str(workspace_root))
sys.path.insert(0, str(scenario_root))
runpy.run_path(str(workspace_root / "app.py"), run_name="__main__")
