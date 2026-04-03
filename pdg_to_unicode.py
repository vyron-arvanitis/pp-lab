from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pp_lab.pdg_to_unicode import pdg_to_unicode  # noqa: F401
