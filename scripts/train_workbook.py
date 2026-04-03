from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from pp_lab.cli.train_workbook import main


if __name__ == "__main__":
    raise SystemExit(main())
