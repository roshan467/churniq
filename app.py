"""ChurnIQ Streamlit entry point.

This root launcher keeps the application compatible with Streamlit Cloud while
leaving the main application code in src/app.py.
"""
from pathlib import Path
import runpy

ROOT = Path(__file__).resolve().parent
runpy.run_path(str(ROOT / "src" / "app.py"), run_name="__main__")
