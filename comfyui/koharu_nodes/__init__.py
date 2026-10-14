"""Koharu inference extension; install this directory as a ComfyUI custom node."""
from pathlib import Path
import sys
# Installation is a symlink into this checkout. Reuse the tested model adapters.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS
from .loaders import register_model_paths
register_model_paths()
from . import routes

WEB_DIRECTORY = './js'
