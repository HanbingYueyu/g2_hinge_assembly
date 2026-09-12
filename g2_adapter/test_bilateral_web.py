"""Read-only three-camera page: pane publishing and HTTP endpoints."""

import json
import urllib.error
import urllib.request
import unittest

from bilateral_web import PANE_NAMES, BilateralWeb, RawPaneFeed, _Pane


class BilateralWebTests(unittest.TestCase):
    def setUp(self) -> None:
        self.web = BilateralWeb(host="127.0.0.1", port=0, title="单元测试页面")
        self.base = self.web.start()
        self.addCleanup(self.web.stop)

    def get(self, path: str):
        return urllib.request.urlopen(f"{self.base}{path}", timeout=5)

    def test_page_shows_head_on_the_first_row_and_hands_side_by_side(self):
        body = self.get("/").read().decode("utf-8")
        self.assertIn("单元测试页面", body)
        self.assertNotIn("__TITLE__", body)
        self.assertNotIn("__NAMES__", body)
        for name in PANE_NAMES:
            self.assertIn(f'"{name}"', body.replace("'", '"'))
            self.assertIn(f"id='img-{name}'", body)
            self.assertIn(f"id='st-{name}'", body)
        # Head first, then the two hand panes inside one grid row.
        head_at = body.index("id='pane-head'")
        row_at = body.index("class='row'")
        left_at = body.index("id='pane-left'")
        right_at = body.index("id='pane-right'")
        self.assertLess(head_at, row_at)
        self.assertLess(row_at, left_at)
        self.assertLess(left_at, right_at)
        self.assertIn("grid-template-columns:1fr 1fr", body)

    def test_page_has_no_hide_control(self):
        body = self.get("/").read().decode("utf-8")
        self.assertNotIn("type='checkbox'", body)
        self.assertNotIn("显示", body)
        self.assertNotIn("collapsed", body)

    def test_each_pane_publishes_its_own_frame_and_status(self):
        for name in PANE_NAMES:
            self.web.pane(name).update(f"jpeg-{name}".encode(), {"state": f"{name}_ready"})
        payload = json.loads(self.get("/status.json").read().decode("utf-8"))
        self.assertEqual(set(payload["panes"]), set(PANE_NAMES))
        for name in PANE_NAMES:
            self.assertEqual(payload["panes"][name]["status"]["state"], f"{name}_ready")
            self.assertGreaterEqual(payload["panes"][name]["sequence"], 1)
            frame = self.get(f"/frame/{name}.jpg").read()
            self.assertEqual(frame, f"jpeg-{name}".encode())

    def test_a_pane_without_a_frame_reports_service_unavailable(self):
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/frame/head.jpg")
        self.assertEqual(caught.exception.code, 503)

    def test_unknown_pane_is_rejected(self):
        with self.assertRaises(ValueError):
            self.web.pane("middle")
        with self.assertRaises(urllib.error.HTTPError) as caught:
            self.get("/frame/middle.jpg")
        self.assertEqual(caught.exception.code, 404)

    def test_hide_and_show_are_reported_to_the_page(self):
        self.web.hide_pane("head")
        payload = json.loads(self.get("/status.json").read().decode("utf-8"))
        self.assertEqual(payload["hidden"], ["head"])
        self.web.show_pane("head")
        payload = json.loads(self.get("/status.json").read().decode("utf-8"))
        self.assertEqual(payload["hidden"], [])

    def test_unqualified_update_targets_the_left_pane(self):
        self.web.update(b"left-jpeg", {"state": "left_only"})
        image, status = self.web.snapshot()
        self.assertEqual(image, b"left-jpeg")
        self.assertEqual(status["state"], "left_only")

    def test_legacy_overlay_endpoint_serves_the_left_pane(self):
        self.web.pane("left").update(b"legacy-jpeg", {"state": "left_ready"})
        self.assertEqual(self.get("/overlay.jpg").read(), b"legacy-jpeg")

    def test_starting_twice_is_rejected_and_running_is_reported(self):
        self.assertTrue(self.web.running)
        with self.assertRaises(RuntimeError):
            self.web.start()


class RawFallbackTests(unittest.TestCase):
    """Every pane stays live: raw frames whenever no stage overlay is active."""

    def setUp(self) -> None:
        self.web = BilateralWeb(host="127.0.0.1", port=0)
        self.addCleanup(self.web.stop)

    def test_a_raw_frame_is_shown_when_no_overlay_has_been_published(self):
        self.assertTrue(self.web.publish_raw_pane("head", b"raw", {"state": "head_raw_camera"}))
        image, status = self.web.snapshot_pane("head")
        self.assertEqual(image, b"raw")
        self.assertEqual(status["state"], "head_raw_camera")

    def test_a_raw_frame_yields_while_a_stage_overlay_is_fresh(self):
        self.web.pane("left").update(b"overlay", {"state": "left_yolo"})
        self.assertFalse(self.web.publish_raw_pane("left", b"raw", {"state": "left_raw"}))
        image, status = self.web.snapshot_pane("left")
        self.assertEqual(image, b"overlay")
        self.assertEqual(status["state"], "left_yolo")

    def test_the_raw_frame_returns_after_the_overlay_goes_stale(self):
        self.web.pane("left").update(b"overlay", {"state": "left_yolo"})
        self.assertFalse(self.web.publish_raw_pane("left", b"raw", {"state": "left_raw"}))
        # Simulate the hold window elapsing by moving the overlay timestamp back.
        with self.web._lock:  # noqa: SLF001 - test-only clock control
            self.web._overlay_at["left"] -= 5.0
        self.assertTrue(self.web.publish_raw_pane("left", b"raw-2", {"state": "left_raw"}))
        self.assertEqual(self.web.snapshot_pane("left")[0], b"raw-2")

    def test_overlay_age_is_infinite_before_any_overlay(self):
        self.assertEqual(self.web.overlay_age("right"), float("inf"))
        self.web.pane("right").update(b"overlay", {})
        self.assertLess(self.web.overlay_age("right"), 1.0)

    def test_an_empty_raw_read_publishes_nothing(self):
        self.assertFalse(self.web.publish_raw_pane("head", None, {"state": "head_raw"}))


class RawPaneFeedTests(unittest.TestCase):
    def setUp(self) -> None:
        self.web = BilateralWeb(host="127.0.0.1", port=0)
        self.addCleanup(self.web.stop)

    def make_feed(self, read_jpeg, *, hold_s=1.0, now=None):
        return RawPaneFeed(
            "head", read_jpeg, self.web, hold_s=hold_s, fps=5.0,
            clock=(lambda: 100.0 if now is None else now),
        )

    def test_feed_publishes_the_raw_frame_when_idle(self):
        feed = self.make_feed(lambda: b"raw")
        self.assertEqual(feed.step(), "published")
        self.assertEqual(self.web.snapshot_pane("head")[0], b"raw")
        self.assertEqual(feed.published, 1)

    def test_feed_backs_off_while_an_overlay_is_fresh(self):
        self.web.pane("head").update(b"overlay", {"state": "head_yolo"})
        called = []
        feed = self.make_feed(lambda: called.append(True) or b"raw")
        self.assertEqual(feed.step(), "overlay")
        self.assertEqual(called, [])
        self.assertEqual(feed.skipped, 1)

    def test_feed_does_not_touch_a_camera_that_a_stage_owns(self):
        called = []
        feed = self.make_feed(lambda: called.append(True) or b"raw")
        feed.pause()
        self.assertEqual(feed.step(), "paused")
        self.assertEqual(called, [])
        feed.resume()
        self.assertEqual(feed.step(), "published")

    def test_feed_reports_read_errors_without_raising(self):
        def boom():
            raise RuntimeError("camera busy")

        feed = self.make_feed(boom)
        self.assertEqual(feed.step(), "error")
        self.assertIsInstance(feed.last_error, RuntimeError)
        # The failure must be visible on the page, not a blank pane.
        status = self.web.snapshot_pane("head")[1]
        self.assertEqual(status["state"], "head_raw_error")
        self.assertIn("camera busy", status["error"])

    def test_feed_ignores_an_empty_read(self):
        feed = self.make_feed(lambda: None)
        self.assertEqual(feed.step(), "empty")
        self.assertEqual(feed.published, 0)

    def test_feed_start_and_stop_are_idempotent_enough_for_teardown(self):
        feed = self.make_feed(lambda: b"raw", hold_s=0.0)
        feed.start()
        self.assertTrue(feed.running)
        with self.assertRaises(RuntimeError):
            feed.start()
        feed.stop()
        feed.stop()
        self.assertFalse(feed.running)


class SharedViewDuckTypeTests(unittest.TestCase):
    """An injected pane must offer every method the stage-owned views expose.

    The right-hand demo calls ``shared.update_status(...)``; a missing method
    aborted the whole right_grasp stage with AttributeError.
    """

    @staticmethod
    def _public_methods(cls) -> set[str]:
        return {
            name for name in dir(cls)
            if not name.startswith("_") and callable(getattr(cls, name, None))
        }

    def test_bilateral_web_covers_the_composite_view_api(self):
        import live_left_d405_sam3_web

        required = self._public_methods(live_left_d405_sam3_web.CompositeSharedView)
        missing = sorted(name for name in required if not hasattr(BilateralWeb, name))
        self.assertEqual(missing, [], f"BilateralWeb 缺少 CompositeSharedView 的方法：{missing}")

    def test_a_pane_covers_the_right_view_api(self):
        import right_apriltag_grasp_gdk

        required = self._public_methods(right_apriltag_grasp_gdk.SharedView)
        missing = sorted(name for name in required if not hasattr(_Pane, name))
        self.assertEqual(missing, [], f"pane 缺少右手 SharedView 的方法：{missing}")

    def test_a_pane_covers_the_left_camera_view_api(self):
        import live_left_d405_sam3_web

        required = self._public_methods(live_left_d405_sam3_web._CameraPane)
        missing = sorted(name for name in required if not hasattr(_Pane, name))
        self.assertEqual(missing, [], f"pane 缺少 _CameraPane 的方法：{missing}")

    def test_update_status_merges_status_without_touching_the_image(self):
        web = BilateralWeb(host="127.0.0.1", port=0)
        self.addCleanup(web.stop)
        web.pane("right").update(b"overlay", {"state": "yolo"})
        web.pane("right").update_status({"motion": "hold_current_pose"})
        image, status = web.snapshot_pane("right")
        self.assertEqual(image, b"overlay")
        self.assertEqual(status["state"], "yolo")
        self.assertEqual(status["motion"], "hold_current_pose")

    def test_update_status_does_not_count_as_an_overlay(self):
        web = BilateralWeb(host="127.0.0.1", port=0)
        self.addCleanup(web.stop)
        web.pane("right").update_status({"state": "premove_pr1"})
        self.assertEqual(web.overlay_age("right"), float("inf"))
        self.assertIsNone(web.snapshot_pane("right")[0])

    def test_update_status_does_not_block_the_raw_fallback(self):
        web = BilateralWeb(host="127.0.0.1", port=0)
        self.addCleanup(web.stop)
        web.pane("right").update_status({"state": "premove_pr1"})
        self.assertTrue(web.publish_raw_pane("right", b"raw", {"state": "right_raw_camera"}))


if __name__ == "__main__":
    unittest.main()
