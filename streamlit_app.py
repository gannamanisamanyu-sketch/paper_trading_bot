"""Streamlit Community Cloud entry point. See dashboard/app.py."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dashboard.app import main  # noqa: E402

main()
