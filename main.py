"""Entry point for Smart Vision Studio desktop application."""

from __future__ import annotations

import os
import sys
import tempfile
from pathlib import Path


def _configure_env() -> None:
    root_dir = str(Path(__file__).resolve().parent)
    if root_dir not in sys.path:
        sys.path.insert(0, root_dir)
    mpl_dir = Path(tempfile.gettempdir()) / "smart_vision_mpl"
    mpl_dir.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("MPLCONFIGDIR", str(mpl_dir))


def main() -> None:
    _configure_env()
    from src.app_ui import SmartVisionStudioApp

    app = SmartVisionStudioApp()
    app.mainloop()


if __name__ == "__main__":
    main()
