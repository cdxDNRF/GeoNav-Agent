"""Isolated frozen CPU pair encoder. stdin contains only2 sanitized images."""
import base64
from dataclasses import asdict
import json
from pathlib import Path
import sys

def emit(value):print(json.dumps(value,ensure_ascii=False,allow_nan=False),flush=True)

def main():
    from project.src.agents.frozen_crossview_encoder_v2 import FrozenCrossViewEncoder
    from project.src.platform_core.ground_encoding_v1 import GroundEncodingSession,decode_frame
    root=Path(__file__).resolve().parents[3]
    manifest=json.loads((root/'project/src/platform_core/ground_live_assets_v1.json').read_text(encoding='utf-8'))
    payload=json.loads(sys.stdin.buffer.read(28*1024**2))
    if set(payload)!={'target_b64','current_b64'}:raise ValueError('Image-only worker')
    encoder=None
    class ReceiptEncoder(FrozenCrossViewEncoder):
        def encode(self,frame,kind):
            value=super().encode(frame,kind)
            emit(dict(type='vector_ready',kind=kind,attempt=self.calls,pixel_sha256=frame.fingerprint(),vector=value))
            return value
    try:
        encoder=ReceiptEncoder(root,manifest['models'],max_forward_calls=2,
            before_forward=lambda value:emit(dict(type='attempt',**value)))
        s=GroundEncodingSession(encoder,event_sink=lambda e:emit(dict(type='evidence',event=e)))
        s.begin(decode_frame(base64.b64decode(payload['target_b64'],validate=True),ground=True),grid_size=2,budget=1)
        packet=s.observe_public(dict(position=(0,0),grid_size=2,step_index=0,remaining_budget=1,
            legal_actions=('up','right','down','left'),current_image=decode_frame(base64.b64decode(payload['current_b64'],validate=True))))
        report=encoder.close()
        emit(dict(type='result',result=dict(packet=asdict(packet),event=s.events[-1],forward_calls=encoder.calls,
            frozen_encoder_report=report,navigation_available=False,navigation_SR=None,actions=0,
            observation_context='standalone local RGB pair;2x2/B1 are interface placeholders, no environment executed')))
        s.close()
    finally:
        if encoder is not None and not encoder.closed:encoder.close()

if __name__=='__main__':
    try:main()
    except Exception as error:
        emit(dict(type='error',error_type=type(error).__name__,no_automatic_retry=True));sys.exit(1)
