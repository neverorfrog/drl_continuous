import os

from .misc import parse_map_grid

IMG_DIR = os.path.join(os.path.dirname(__file__), "img")
Q_DIR = os.path.join(os.path.dirname(__file__), "qtables")

__all__ = ["parse_map_grid", "IMG_DIR", "Q_DIR"]
