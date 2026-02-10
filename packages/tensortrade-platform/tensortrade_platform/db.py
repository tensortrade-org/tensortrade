"""Shared database path constant for all platform stores."""

import os

DEFAULT_DB_DIR = os.path.expanduser("~/.tensortrade")
DEFAULT_DB_PATH = os.path.join(DEFAULT_DB_DIR, "experiments.db")
