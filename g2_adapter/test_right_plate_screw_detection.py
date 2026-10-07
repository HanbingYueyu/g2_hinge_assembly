"""Check ROI coordinate conversion and live-preview failure reporting."""
import io
import json
import re
import shutil
import subprocess
import threading
from types import SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np

from .right_plate_screw_detection import detect, read_camera
from .live_right_plate_screws_web import PAGE, Latest, make_handler, run_detector


class ScrewDetectionTests(unittest.TestCase):
    @unittest.skipUnless(shutil.which('node'), 'Node.js unavailable')
    def test_browser_executes_and_updates_preview(self):
        script = re.search(r'<script>(.*?)</script>', PAGE, re.S).group(1)
        for ok in (True, False):
            frame = dict(ok=ok, count=2 if ok else None, sequence=1,
                         updated_at=0, inference_ms=6, detections=[],
                         image='data:image/jpeg;base64,ABC', error='camera disconnected')
            harness = '''
const vm = require('node:vm');
const assert = require('node:assert/strict');
const elements = {count:{},view:{dataset:{},hidden:true},status:{}};
const frame = FRAME;
frame.updated_at = Date.now()/1000;
vm.runInNewContext(SCRIPT, {
  fetch:async()=>({ok:true,json:async()=>frame}),
  document:{getElementById:id=>elements[id]},
  Date, String, Error, setTimeout:()=>{}
});
setImmediate(()=>{
 assert.equal(elements.count.textContent, frame.ok?'螺丝：2 颗':'画面暂不可用');
 assert.equal(elements.view.hidden, !frame.ok);
 if(frame.ok) assert.equal(elements.view.src, frame.image);
 else assert.equal(elements.status.textContent, 'camera disconnected');
});
'''.replace('FRAME', json.dumps(frame)).replace('SCRIPT', json.dumps(script))
            result = subprocess.run(['node'], input=harness, text=True, capture_output=True)
            self.assertEqual(result.returncode, 0, result.stderr)

    def test_crop_and_original_coordinates(self):
        image = np.zeros((1056, 1280, 3), dtype=np.uint8)
        image[435:815, 525:780] = 123
        box = SimpleNamespace(cls=np.array(0), conf=np.array(.9),
                              xyxy=np.array([[10., 20., 30., 40.]]))
        prediction = SimpleNamespace(boxes=[box], plot=Mock(return_value=image[435:815, 525:780]))
        model = SimpleNamespace(names={0: 'plate_screw'}, predict=Mock(return_value=[prediction]))
        crop, _, data = detect(model, image)
        self.assertEqual(crop.shape, (380, 255, 3))
        self.assertTrue(np.all(crop == 123))
        self.assertEqual(data['count'], 1)
        self.assertEqual(data['detections'][0]['bbox_crop_xyxy'], [10, 20, 30, 40])
        self.assertEqual(data['detections'][0]['bbox_xyxy'], [535, 455, 555, 475])
        self.assertEqual(model.predict.call_args.kwargs['imgsz'], 384)

    def test_reject_already_cropped_image(self):
        with self.assertRaisesRegex(ValueError, 'full 1280x1056'):
            detect(None, np.zeros((380, 255, 3), dtype=np.uint8))

    def test_reject_stale_camera_frame(self):
        payload = dict(images=[dict(topic='/camera/hand_right_color', available=True, age_ms=501)])
        with patch('g2_adapter.right_plate_screw_detection.urlopen',
                   return_value=io.BytesIO(json.dumps(payload).encode())):
            with self.assertRaisesRegex(RuntimeError, 'fresh'):
                read_camera('http://unused')

    def test_camera_failure_is_not_zero_screws(self):
        latest, stop = Latest(), threading.Event()
        def disconnected(_url):
            stop.set()
            raise RuntimeError('camera disconnected')
        args = SimpleNamespace(camera_url='http://unused', interval=.01, confidence=.5)
        with patch('g2_adapter.live_right_plate_screws_web.read_camera', side_effect=disconnected):
            run_detector(None, args, latest, stop)
        status = latest.get()
        self.assertFalse(status['ok'])
        self.assertIsNone(status['count'])
        self.assertEqual(status['error'], 'camera disconnected')

    def test_frame_and_status_share_snapshot_without_mutation(self):
        latest = Latest()
        latest.put(dict(ok=True, count=2, sequence=1, image='data:image/jpeg;base64,ABC'))
        handler_type = make_handler(latest)
        for path in ('/status', '/frame'):
            handler = object.__new__(handler_type)
            handler.path = path
            handler.wfile = io.BytesIO()
            handler.send_response = Mock()
            handler.send_header = Mock()
            handler.end_headers = Mock()
            handler.do_GET()
            payload = json.loads(handler.wfile.getvalue())
            self.assertEqual(payload['count'], 2)
            self.assertEqual('image' in payload, path == '/frame')
        self.assertIn('image', latest.get())


if __name__ == '__main__':
    unittest.main()
