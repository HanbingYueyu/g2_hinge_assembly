"""Head preview ownership: a preview must never close a camera it borrowed.

The single-process orchestrator injects one head camera for the whole run.
The preview thread used to close it unconditionally, so the orchestrator's
teardown closed the same native camera again and the interpreter aborted with
``malloc(): mismatching next->prev_size`` inside the GDK binding.
"""

import pathlib
import unittest
from types import SimpleNamespace
from unittest.mock import Mock

import left_grasp_gdk


def start_preview(*, camera, model, **overrides):
    arguments = dict(
        gdk=SimpleNamespace(CameraType=SimpleNamespace(kHeadColor="head")),
        head_model_path=pathlib.Path("model.pt"),
        args=SimpleNamespace(head_yolo_confidence=0.5, yolo_device="cpu"),
        shared=Mock(),
        camera=camera,
        model=model,
    )
    arguments.update(overrides)
    stop, worker = left_grasp_gdk.start_head_camera_web_preview(**arguments)
    left_grasp_gdk.stop_head_camera_web_preview(stop, worker)
    return stop, worker


class HeadPreviewOwnershipTests(unittest.TestCase):
    def test_borrowed_camera_is_not_closed(self):
        camera, model = Mock(), Mock()
        start_preview(camera=camera, model=model, owns_camera=False, owns_model=False)
        camera.close_camera.assert_not_called()

    def test_owned_camera_is_closed_once(self):
        camera, model = Mock(), Mock()
        start_preview(camera=camera, model=model)
        camera.close_camera.assert_called_once()

    def test_preview_without_a_shared_page_starts_no_thread(self):
        stop, worker = left_grasp_gdk.start_head_camera_web_preview(
            gdk=Mock(), head_model_path=pathlib.Path("model.pt"), args=Mock(), shared=None,
        )
        self.assertIsNone(stop)
        self.assertIsNone(worker)


if __name__ == "__main__":
    unittest.main()
