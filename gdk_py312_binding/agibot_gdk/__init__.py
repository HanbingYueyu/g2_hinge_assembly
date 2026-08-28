"""Project-local CPython 3.12 facade for the official GDK extension."""

import importlib


_native = importlib.import_module("agibot_gdk.agibot_gdk")
globals().update({name: getattr(_native, name) for name in dir(_native) if not name.startswith("_")})

# Copied during build from the matching official GDK runtime.
from . import dds
