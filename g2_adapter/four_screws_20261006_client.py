"""Standalone 20261006 screw-count candidate; original crops and two references."""
import base64
import json
from pathlib import Path
import time
from vlm_checks.src.vlm_three_checks import VlmClient, api_key_from_environment, response_text

ASSETS = Path(__file__).with_name('vlm_four_screws_20261006')
LABELS = ['参考图：实际0颗螺丝。', '参考图：实际4颗螺丝。',
          '待测当前图。数清实际螺丝，只输出JSON {"count":0}，看不清返回{"count":"unknown"}。']


def decode_result(raw):
    try:
        count = json.loads(raw.strip().removeprefix('```json').removesuffix('```').strip()).get('count')
    except (ValueError, AttributeError):
        count = None
    if str(count) not in {'0', '1', '2', '3', '4'}:
        count = 'unknown'
    complete = 'unknown' if count == 'unknown' else 'yes' if str(count) == '4' else 'no'
    return {'model_count': count, 'complete': complete, 'latest_decision': complete, 'raw': raw}


class FourScrews20261006Client(VlmClient):
    input_labels = LABELS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session.trust_env = False
        self.prompt = (ASSETS/'prompt.txt').read_text(encoding='utf-8').strip()
        self.references = ((ASSETS/'reference_zero.jpg').read_bytes(),
                           (ASSETS/'reference_four.jpg').read_bytes())

    def prepare_images(self, frames):
        if not frames:
            raise ValueError('需要当前相机裁剪图')
        return (*self.references, frames[-1])

    def run_bytes(self, task, images):
        if task != 'four-screws' or len(images) != 3:
            raise ValueError('输入必须为零颗参考图、四颗参考图、当前裁剪图')
        content = [{'type': 'text', 'text': self.prompt}]
        for label, data in zip(LABELS, images, strict=True):
            url = 'data:image/jpeg;base64,'+base64.b64encode(data).decode('ascii')
            content.extend([{'type': 'text', 'text': label},
                            {'type': 'image_url', 'image_url': {'url': url}}])
        key = api_key_from_environment(self.mode)
        headers = {'Content-Type': 'application/json'}
        if key:
            headers['Authorization'] = 'Bearer '+key
        started = time.monotonic()
        response = self.session.post(
            self.api_url, headers=headers,
            json={'model': self.model, 'messages': [{'role': 'user', 'content': content}],
                  'temperature': 0, 'max_tokens': 100}, timeout=self.timeout_s,
        )
        response.raise_for_status()
        result = decode_result(response_text(response.json()))
        return {**result, 'seconds': round(time.monotonic()-started, 3)}
