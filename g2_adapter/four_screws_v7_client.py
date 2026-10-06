"""V7 two-reference matching, scoped to the standalone live review."""
import base64
from pathlib import Path

from vlm_checks.src.vlm_three_checks import (
    VlmClient, api_key_from_environment, parse_json_object, response_text,
)

ASSETS = Path(__file__).with_name('vlm_four_screws_v7')
LABELS = [
    '参考图A：人工确认的0颗、1颗、2颗：',
    '参考图B：人工确认的3颗、4颗：',
    '待测图1（待测图2是当前状态）：',
    '待测图2（待测图2是当前状态）：',
]


def validate_v7(result):
    if not isinstance(result, dict):
        raise ValueError('V7 未返回 JSON 对象')
    count = result.get('count')
    if count not in {'0', '1', '2', '3', '4', 'unknown'}:
        raise ValueError('V7 count 无效')
    expected = 'unknown' if count == 'unknown' else 'yes' if count == '4' else 'no'
    if result.get('complete') != expected:
        raise ValueError('V7 count 与 complete 矛盾')
    if result.get('closest_reference') not in {'0', '1', '2', '3', '4', 'unknown'}:
        raise ValueError('V7 closest_reference 无效')
    # Preserve the original V7 response, exposing the existing preview field.
    return {**result, 'latest_decision': expected}


class FourScrewsV7Client(VlmClient):
    input_labels = LABELS

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.session.trust_env = False
        self.prompt = (ASSETS/'prompt.txt').read_text(encoding='utf-8')
        self.references = ((ASSETS/'reference_a.jpg').read_bytes(),
                           (ASSETS/'reference_b.jpg').read_bytes())

    def prepare_images(self, frames):
        if len(frames) < 2:
            raise ValueError('V7 需要同一相机连续的前帧和当前帧')
        return (*self.references, *frames[-2:])

    def run_bytes(self, task, images):
        if task != 'four-screws' or len(images) != 4:
            raise ValueError('V7 输入必须为两张参考图和两张待测图')
        content = [{'type': 'text', 'text': self.prompt}]
        for label, data in zip(LABELS, images, strict=True):
            url = 'data:image/jpeg;base64,'+base64.b64encode(data).decode('ascii')
            content.extend([{'type': 'text', 'text': label},
                            {'type': 'image_url', 'image_url': {'url': url}}])
        key = api_key_from_environment(self.mode)
        if not key:
            raise RuntimeError('VLM API key 未配置，请检查 --vlm-env-file')
        response = self.session.post(
            self.api_url, headers={'Authorization': 'Bearer '+key, 'Content-Type': 'application/json'},
            json={'model': self.model, 'messages': [{'role': 'user', 'content': content}],
                  'temperature': 0, 'max_tokens': 400}, timeout=self.timeout_s,
        )
        response.raise_for_status()
        return validate_v7(parse_json_object(response_text(response.json())))
