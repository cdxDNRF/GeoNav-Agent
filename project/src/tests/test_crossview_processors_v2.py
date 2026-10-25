"""Resolve the library's implicit Torchvision/PIL backend difference, before model use."""
import sys,unittest,json
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[3];sys.path.insert(0,str(ROOT/'project/src'))
from data.mmgag_views_v1 import ground_pixels
from transformers.models.clip.image_processing_pil_clip import CLIPImageProcessorPil
class ProcessorTests(unittest.TestCase):
    def test_explicit_pil_matches_bound_preprocessor(self):
        model=ROOT/'models/clip_vit_b32_public_v1';config=json.loads((model/'preprocessor_config.json').read_text())
        ref=CLIPImageProcessorPil.from_pretrained(str(model),local_files_only=True)
        rng=np.random.default_rng(73)
        for w,h in [(512,384),(384,512),(517,369),(224,224)]:
            rgb=rng.integers(0,256,size=(h,w,3),dtype=np.uint8)
            a=ground_pixels(rgb,config);b=ref(images=rgb,return_tensors='np')['pixel_values'][0]
            self.assertLessEqual(float(np.abs(a-b).max()),1e-5)
if __name__=='__main__':unittest.main()
