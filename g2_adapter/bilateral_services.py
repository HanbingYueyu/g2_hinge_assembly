#!/usr/bin/env python3
"""Injection contract shared by the staged bilateral entry point and the demos.

``StageServices`` lets the single-process orchestrator hand an already-created
GDK session, controller, gate, SpaceMouse worker, cameras and preloaded models
to a demo stage.  A stage called with ``services=None`` keeps the original
standalone behaviour: it creates and owns everything itself.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any


@dataclass
class StageServices:
    """Objects owned by the orchestrator and only borrowed by one stage."""

    gdk: Any = None
    robot: Any = None
    controller: Any = None
    gate: Any = None
    worker: Any = None
    dianpi_runner: Any = None
    tf: Any = None

    # Preloaded, resident models.
    head_yolo: Any = None
    left_yolo: Any = None
    right_detector: Any = None

    # Cameras created once by the orchestrator.
    head_camera: Any = None
    left_camera: Any = None
    right_camera: Any = None

    # Shared web surface (one page, three panes).
    web: Any = None

    # Cross-stage state produced by an earlier stage.
    extra: dict = field(default_factory=dict)

    def put(self, key: str, value: Any) -> Any:
        self.extra[key] = value
        return value

    def get(self, key: str, default: Any = None) -> Any:
        return self.extra.get(key, default)
