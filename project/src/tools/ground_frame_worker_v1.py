"""One RGB image in, one frozen512-vector out; no scene or answer input."""
import base64
import json
from pathlib import Path
import sys

def emit(v):print(json.dumps(v,ensure_ascii=False,allow_nan=False),flush=True)

def main():
    from project.src.agents.frozen_crossview_encoder_v2 import FrozenCrossViewEncoder
    from project.src.platform_core.ground_encoding_v1 import decode_frame
    root=Path(__file__).resolve().parents[3]
    payload=json.loads(sys.stdin.buffer.read(2*1024**2))
    if set(payload)!={'kind','image_b64'} or payload['kind'] not in ('ground','aerial'):raise ValueError('Single-image fields only')
    assets=json.loads((root/'project/src/platform_core/ground_live_assets_v1.json').read_text(encoding='utf-8'))['models']
    frame=decode_frame(base64.b64decode(payload['image_b64'],validate=True),ground=payload['kind']=='ground')
    encoder=FrozenCrossViewEncoder(root,assets,max_forward_calls=1,
        before_forward=lambda e:emit(dict(type='forward_attempt',**e)))
    try:
        feature=encoder.encode(frame,payload['kind'])
        emit(dict(type='forward_success',kind=payload['kind'],pixel_sha256=frame.fingerprint(),feature=feature))
        report=encoder.close()
        emit(dict(type='result',result=dict(feature=feature,forward_calls=1,frozen_encoder_report=report)))
    finally:
        if not encoder.closed:encoder.close()

if __name__=='__main__':
    try:main()
    except Exception as e:emit(dict(type='error',error_type=type(e).__name__));sys.exit(1)
