"""Compatibility entry point for the formal Full/no-SSAW horizon queue."""

from scripts.run_full_no_ssaw_horizon_queue import *  # noqa: F401,F403
from scripts.run_full_no_ssaw_horizon_queue import main


if __name__ == "__main__":
    raise SystemExit(main())
