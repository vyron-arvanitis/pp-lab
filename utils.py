from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from pp_lab.data import *  # noqa: F401,F403
from pp_lab.training import *  # noqa: F401,F403
