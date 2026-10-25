"""Pure in-memory interface/preprocessor tests; no real encoder forward."""
from dataclasses import FrozenInstanceError
from io import BytesIO
from pathlib import Path
import json
import sys
import unittest
from unittest.mock import patch
import numpy as np
from PIL import Image,ImageOps,PngImagePlugin
from project.src.platform_core.ground_encoding_v1 import GroundEncodingSession,decode_frame
from project.src.platform_core.local_goal_evidence_v1 import RGBFrame
from project.src.agents.frozen_crossview_encoder_v1 import FrozenCrossViewEncoder,pixels

ROOT=Path(__file__).resolve().parents[3]
ALL=('up','right','down','left')


def frame(value=100):return RGBFrame(2,2,bytes([value,20,30])*4)
def encoded(value=100):
    stream=BytesIO();Image.new('RGB',(2,2),(value,20,30)).save(stream,format='PNG');return stream.getvalue()
def observation(pos=(2,2),step=0,budget=10,image=None):
    return dict(position=pos,grid_size=5,step_index=step,remaining_budget=budget-step,legal_actions=ALL,current_image=image or frame())


class FakeEncoder:
    mode='synthetic'
    def __init__(self):self.calls=[]
    def encode(self,image,kind):
        self.calls.append((image.fingerprint(),kind))
        return tuple([float(image.pixels[0])+1.,1.]+[0.]*510)


class Tests(unittest.TestCase):
    def setUp(self):
        self.encoder=FakeEncoder();self.session=GroundEncodingSession(self.encoder)
    def begin(self,**kw):self.session.begin(frame(200),grid_size=5,budget=kw.get('budget',10))
    def test_only_single_rgb_goal(self):
        for value in ({'path':'secret'},[frame()],b'filename'):
            with self.assertRaises(ValueError):self.session.begin(value,grid_size=5,budget=10)
        self.assertEqual(self.encoder.calls,[])
    def test_hidden_fields_fail_before_aerial_encoding(self):
        for field in ('goal_index','source_id','all_features','candidate_bank','visited','target_image'):
            self.begin();obs=observation();obs[field]=0
            before=len(self.encoder.calls)
            with self.assertRaises(ValueError):self.session.observe_public(obs)
            self.assertEqual(len(self.encoder.calls),before)
    def test_feature_packet_is_immutable_and_no_action(self):
        self.begin();packet=self.session.observe_public(observation())
        self.assertEqual(len(packet.current_feature),512);self.assertEqual(packet.visited,(12,))
        self.assertNotIn('action',vars(packet));self.assertNotIn('goal_index',vars(packet))
        with self.assertRaises(FrozenInstanceError):packet.position=(0,0)
    def test_seen_pixel_reuse_and_no_cross_episode_cache(self):
        self.begin();self.session.observe_public(observation())
        packet=self.session.observe_public(observation((1,2),1),last_action='up')
        self.assertEqual(len(self.encoder.calls),2);self.assertEqual(packet.visited,(12,7))
        self.assertTrue(self.session.events[-1]['current_cache_hit'])
        first_id=self.session.session_id;self.begin();self.session.observe_public(observation())
        self.assertEqual(len(self.encoder.calls),4);self.assertNotEqual(first_id,self.session.session_id)
        self.assertEqual(self.session.visited,[12])
    def test_bad_transition_latches_without_forward(self):
        self.begin();self.session.observe_public(observation());before=len(self.encoder.calls)
        with self.assertRaises(ValueError):self.session.observe_public(observation((4,4),1),last_action='up')
        with self.assertRaises(ValueError):self.session.observe_public(observation((1,2),1),last_action='up')
        self.assertEqual(len(self.encoder.calls),before)
    def test_budget_and_type_checks(self):
        for obs in (observation(step=1),observation(budget=9),dict(observation(),grid_size=True),dict(observation(),position=(True,2))):
            self.begin()
            with self.assertRaises(ValueError):self.session.observe_public(obs)
    def test_boundary_observation_consumes_step(self):
        self.begin();self.session.observe_public(observation((0,0)))
        packet=self.session.observe_public(observation((0,0),1),last_action='up')
        self.assertEqual(packet.remaining_budget,9);self.assertEqual(packet.visited,(0,0))
    def test_budget_zero_terminal_observation(self):
        self.begin(budget=1);self.session.observe_public(observation(budget=1))
        obs=observation((2,3),1,budget=1);obs['legal_actions']=()
        self.assertEqual(self.session.observe_public(obs,last_action='right').remaining_budget,0)
        with self.assertRaises(ValueError):self.session.observe_public(obs,last_action='right')
    def test_encoder_failure_latches(self):
        self.begin()
        with patch.object(self.encoder,'encode',side_effect=RuntimeError('failure')):
            with self.assertRaises(RuntimeError):self.session.observe_public(observation())
        with self.assertRaises(ValueError):self.session.observe_public(observation())
    def test_bad_encoder_vector(self):
        for value in ([0.]*512,[float('nan')]*512,[1.]*513):
            with patch.object(self.encoder,'encode',return_value=value):
                with self.assertRaises(ValueError):self.begin()
            self.assertFalse(self.session.active)
    def test_event_sink_failure_latches(self):
        sink=lambda event:(_ for _ in ()).throw(OSError('disk'))
        s=GroundEncodingSession(self.encoder,event_sink=sink)
        with self.assertRaises(OSError):s.begin(frame(),grid_size=5,budget=10)
        self.assertFalse(s.active)
    def test_bridge_ignores_aerial_target_and_external_history(self):
        class Public:
            current_image=encoded();position=(2,2);grid_size=5;remaining_budget=10;legal_actions=ALL
            @property
            def target_image(self):raise AssertionError('Must ignore original aerial target')
            @property
            def visited(self):raise AssertionError('Must build own causal history')
            @property
            def goal_index(self):raise AssertionError('Must ignore truth')
        self.begin();packet=self.session.observe_environment(Public())
        self.assertEqual(packet.visited,(12,));self.assertEqual(len(self.encoder.calls),2)
    def test_bad_bridge_image_latches(self):
        class Public:
            current_image=b'not_image';position=(2,2);grid_size=5;remaining_budget=10;legal_actions=ALL
        self.begin()
        with self.assertRaises(Exception):self.session.observe_environment(Public())
        self.assertFalse(self.session.active)
    def test_close_drops_goal_and_view_memory(self):
        self.begin();self.session.observe_public(observation());self.session.close()
        self.assertIsNone(self.session.goal);self.assertEqual(self.session.cache,{})
        with self.assertRaises(ValueError):self.session.observe_public(observation())
    def test_exif_orientation_and_metadata_sanitization(self):
        image=Image.new('RGB',(4,2),(80,20,30));exif=image.getexif();exif[274]=6
        stream=BytesIO();image.save(stream,format='JPEG',exif=exif)
        goal=decode_frame(stream.getvalue(),ground=True);current=decode_frame(stream.getvalue())
        self.assertEqual((goal.width,goal.height),(2,4));self.assertEqual((current.width,current.height),(4,2))
        self.assertEqual(set(vars(goal)),{'width','height','pixels'})
    def test_decode_limits_and_rgb(self):
        with self.assertRaises(ValueError):decode_frame(b'')
        with self.assertRaises(ValueError):decode_frame('a path')
        stream=BytesIO();Image.new('L',(2,3),40).save(stream,format='PNG')
        value=decode_frame(stream.getvalue());self.assertEqual(len(value.pixels),18)
    def test_preprocessor_contract(self):
        config=json.loads((ROOT/'models/clip_vit_b32_public_v1/preprocessor_config.json').read_text())
        im=np.arange(8*12*3,dtype=np.uint8).reshape(8,12,3)
        f=RGBFrame(12,8,im.tobytes())
        actual=pixels(f,'ground',config)
        resized=Image.fromarray(im).resize((336,224),Image.Resampling.BICUBIC).crop((56,0,280,224))
        ref=(np.asarray(resized,dtype=np.float32)/np.float32(255)-np.asarray(config['image_mean'],np.float32))/np.asarray(config['image_std'],np.float32)
        self.assertLessEqual(float(np.max(np.abs(actual-ref.transpose(2,0,1)))),1e-6)
        self.assertEqual(pixels(f,'aerial',config).shape,(3,224,224))
    def test_forward_cap_rejects_before_model_load(self):
        encoder=FrozenCrossViewEncoder.__new__(FrozenCrossViewEncoder);encoder.closed=False;encoder.calls=8;encoder.max_calls=8
        with patch.object(encoder,'_load',side_effect=AssertionError('must not load')):
            with self.assertRaises(ValueError):encoder.encode(frame(),'ground')
    def test_unbound_assets_fail_before_framework_import(self):
        with self.assertRaises(ValueError):FrozenCrossViewEncoder(ROOT,[])
    def test_uncalibrated_evidence_not_confidence(self):
        self.begin();self.session.observe_public(observation());event=self.session.events[-1]
        self.assertIsNone(event['confidence']);self.assertFalse(event['navigation_available'])
        self.assertLessEqual(abs(event['raw_similarity']),1.)


if __name__=='__main__':unittest.main()
