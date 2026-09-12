import unittest

import cv2
import numpy as np

from live_left_d405_sam3_web import CompositeSharedView


class CompositeSharedViewTests(unittest.TestCase):
    def test_hiding_head_switches_to_left_only_view(self):
        view = CompositeSharedView()
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        ok, encoded = cv2.imencode(".jpg", image)
        self.assertTrue(ok)
        jpeg = encoded.tobytes()

        view.pane("head").update(jpeg, {"state": "head_camera_live", "source": "head"})
        view.pane("left").update(jpeg, {"state": "left_camera_live", "source": "left"})
        view.hide_pane("head")

        overlay, status = view.snapshot()
        self.assertIsNotNone(overlay)
        decoded = cv2.imdecode(np.frombuffer(overlay, dtype=np.uint8), cv2.IMREAD_COLOR)
        self.assertEqual(tuple(decoded.shape[:2]), (24, 32))
        self.assertTrue(status["camera_views"]["head"]["hidden"])
        self.assertEqual(status["active_source"], "left")


if __name__ == "__main__":
    unittest.main()
