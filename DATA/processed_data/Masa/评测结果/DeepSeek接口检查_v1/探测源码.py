"""Bounded API and synthetic two-image check; no dataset images or navigation."""
import argparse
import base64
from datetime import datetime, timezone
from hashlib import sha256
import io
import json
from pathlib import Path
import random
import re
import sys
import time

from PIL import Image, ImageDraw

if __package__ in (None, ''):
    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from agents.curl_transport import CurlTransport
from agents.vlm import APIConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    # Explicit candidate file: do not inherit default-provider environment overrides.
    values = {}
    for line in args.config.read_text(encoding='utf-8').splitlines():
        if line.strip() and not line.lstrip().startswith('#'):
            key, value = line.split('=', 1)
            values[key.strip()] = value.strip().strip('\"\'')
    config = APIConfig(values['VLM_BASE_URL'], values['VLM_MODEL'], values['VLM_API_KEY'], timeout=45, max_tokens=256)
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=False)
    headers = {'Authorization': 'Bearer ' + config.api_key}
    transport = CurlTransport(timeout=45)
    records = []

    def save(name, value):
        text = json.dumps(value, ensure_ascii=False, indent=2)
        text = text.replace(config.api_key, '[REDACTED]')
        text = re.sub(r'sk-[A-Za-z0-9_-]+', '[REDACTED]', text)
        if 'data:image/' in text:
            raise ValueError('Image payload must not be logged')
        with (out / name).open('x', encoding='utf-8') as stream:
            stream.write(text)

    def request(kind, method, path, payload=None):
        began = time.monotonic()
        record = dict(kind=kind, method=method, path=path)
        try:
            response = transport.request(method, config.base_url.rstrip('/') + path, headers, json=payload)
            record['http_status'] = response.status_code
            try:
                body = response.json()
            except ValueError:
                body = {'non_json_response': True}
            if not response.is_success:
                record['error'] = body.get('error', {'message': 'Non-success HTTP response'})
            elif kind == 'model_list':
                record['models'] = [item.get('id') for item in body.get('data', [])]
            else:
                choices = body.get('choices', [])
                first = choices[0] if choices else {}
                record.update(returned_model=body.get('model'), usage=body.get('usage'),
                              finish_reason=first.get('finish_reason'),
                              content=first.get('message', {}).get('content'))
        except Exception as exc:
            record['transport_error_type'] = type(exc).__name__
        record['seconds'] = time.monotonic() - began
        records.append(record)
        save(f'{len(records):02d}_{kind}.json', record)
        return record

    prompt = ('Inspect the two supplied images in order. Return only a JSON object with '
              'first_color, first_shape, second_color, second_shape. '
              'Colors: red, green, blue. Shapes: circle, square, triangle. '
              'Describe the colored shape on the white background in each image.')
    rng = random.Random(9030)
    colors = rng.sample(['red', 'green', 'blue'], 2)
    shapes = rng.sample(['circle', 'square', 'triangle'], 2)
    images = []
    image_hashes = []
    for i, (color, shape) in enumerate(zip(colors, shapes)):
        canvas = Image.new('RGB', (192, 192), 'white')
        draw = ImageDraw.Draw(canvas)
        if shape == 'circle':
            draw.ellipse((40, 40, 152, 152), fill=color)
        elif shape == 'square':
            draw.rectangle((40, 40, 152, 152), fill=color)
        else:
            draw.polygon(((96, 32), (32, 152), (160, 152)), fill=color)
        buffer = io.BytesIO()
        canvas.save(buffer, format='PNG')
        content = buffer.getvalue()
        (out / f'输入_{i + 1}.png').write_bytes(content)
        images.append(content)
        image_hashes.append(sha256(content).hexdigest())
    save('运行配置.json', dict(utc=datetime.now(timezone.utc).isoformat(), provider=config.public(),
        scope='Synthetic shapes only; not a navigation or general visual capability benchmark.',
        maximum_requests=3, retries=0, prompt=prompt, image_sha256=image_hashes,
        seed=9030, source_sha256=sha256(Path(__file__).read_bytes()).hexdigest()))
    listed = request('model_list', 'GET', '/models')
    checks = []
    if listed.get('http_status') == 200:
        for order in ((0, 1), (1, 0)):
            content = [{'type': 'text', 'text': prompt}]
            for index in order:
                content.append({'type': 'image_url', 'image_url': {
                    'url': 'data:image/png;base64,' + base64.b64encode(images[index]).decode('ascii')}})
            payload = dict(model=config.model, messages=[{'role': 'user', 'content': content}],
                           max_tokens=config.max_tokens, temperature=0, stream=False,
                           response_format={'type': 'json_object'})
            record = request('two_image_' + ''.join(str(i) for i in order), 'POST', '/chat/completions', payload)
            expected = dict(first_color=colors[order[0]], first_shape=shapes[order[0]],
                            second_color=colors[order[1]], second_shape=shapes[order[1]])
            try:
                actual = json.loads(record.get('content') or '')
            except (ValueError, TypeError):
                actual = None
            checks.append(dict(order=list(order), expected=expected, parsed=actual, passed=actual == expected))
            if record.get('http_status') != 200:
                break
    summary = dict(requested_model=config.model, base_url=config.base_url,
                   model_list_http_status=listed.get('http_status'),
                   model_listed=config.model in listed.get('models', []), requests=len(records),
                   visual_checks=checks, synthetic_two_image_passed=len(checks) == 2 and all(c['passed'] for c in checks),
                   navigation_evaluated=False, parameter_count_verified=False)
    save('检查结论.json', summary)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == '__main__':
    main()
