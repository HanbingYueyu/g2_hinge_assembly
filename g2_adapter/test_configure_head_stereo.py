import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import configure_hand_depth as config


class StereoConfigTests(unittest.TestCase):
    def test_stereo_profile_enables_stereo_and_existing_workflow_streams(self):
        _, document = config.load_and_validate(Path('config/thor_camera_conf_head_stereo.json'))
        enabled = {item['name'] for item in document.values() if item['publish']}
        self.assertEqual(enabled, {'head_stereo_left', 'head_stereo_right', 'head_color', 'head_depth', 'hand_left_color', 'hand_right_color', 'hand_left_depth', 'hand_right_depth'})

    def test_missing_stereo_stream_is_rejected(self):
        source = config.PROJECT_ROOT / 'config/thor_camera_conf_head_stereo.json'
        document = json.loads(source.read_text())
        document['cam12']['publish'] = False
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'profile.json'
            path.write_text(json.dumps(document))
            with patch.dict(config.PROFILE_REQUIREMENTS, {path.resolve(): config.PROFILE_REQUIREMENTS[source.resolve()]}):
                with self.assertRaises(ValueError):
                    config.load_and_validate(path)
