import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from scripts.structural_ssaw_runner_common import main_for_runner


if __name__ == "__main__":
    raise SystemExit(main_for_runner("simplified_no_confidence"))
