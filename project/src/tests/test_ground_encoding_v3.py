"""Durability boundary tests; deterministic fake vectors, zero model forward."""
import json
import tempfile
import unittest
from unittest.mock import patch
import numpy as np
from project.src.platform_core.local_goal_evidence_v1 import RGBFrame
from project.src.tools import verify_ground_encoding_v3 as runner

class Tests(unittest.TestCase):
    def setUp(self):
        self.tmp=tempfile.TemporaryDirectory(prefix='test_tmp-',dir=runner.OUT)
        self.folder=__import__('pathlib').Path(self.tmp.name)
        self.frame=RGBFrame(1,1,bytes([17,24,36]))
        self.vector=tuple(float(x) for x in np.arange(512,dtype=np.float32))
    def tearDown(self):
        assert self.folder.resolve().is_relative_to(runner.OUT.resolve())
        self.tmp.cleanup()
    def test_success_vector_is_durable_before_return(self):
        from project.src.agents.frozen_crossview_encoder_v2 import FrozenCrossViewEncoder
        cls=runner.make_journaled_encoder();encoder=cls.__new__(cls);encoder.calls=1
        with patch.object(FrozenCrossViewEncoder,'encode',return_value=self.vector) as fake,patch.object(runner,'OUT',self.folder):
            result=encoder.encode(self.frame,'ground')
        fake.assert_called_once_with(self.frame,'ground');self.assertEqual(result,self.vector)
        receipts=[json.loads(s) for s in (self.folder/'编码成功.jsonl').read_text(encoding='utf-8').splitlines()]
        self.assertEqual(len(receipts),1);runner.verify([receipts[0]['vector']])
        value=np.load(runner.ROOT/receipts[0]['vector']['path'],allow_pickle=False)
        np.testing.assert_array_equal(value,np.asarray(self.vector,np.float32))
        self.assertEqual(value.dtype,np.float32);self.assertEqual(receipts[0]['pixel_sha256'],self.frame.fingerprint())
    def test_persistence_failure_propagates_without_retry(self):
        from project.src.agents.frozen_crossview_encoder_v2 import FrozenCrossViewEncoder
        cls=runner.make_journaled_encoder();encoder=cls.__new__(cls);encoder.calls=1
        with patch.object(FrozenCrossViewEncoder,'encode',return_value=self.vector) as fake,patch.object(runner,'persist_vector',side_effect=OSError('full')):
            with self.assertRaises(OSError):encoder.encode(self.frame,'ground')
        fake.assert_called_once();self.assertEqual(list(self.folder.iterdir()),[])
    def test_existing_vector_never_overwritten(self):
        runner.persist_vector(self.folder,1,'ground',self.frame,self.vector)
        before=(self.folder/'编码_01_ground.npy').read_bytes()
        with self.assertRaises(FileExistsError):runner.persist_vector(self.folder,1,'ground',self.frame,tuple([0.]*512))
        self.assertEqual(before,(self.folder/'编码_01_ground.npy').read_bytes())
        self.assertEqual(len((self.folder/'编码成功.jsonl').read_text(encoding='utf-8').splitlines()),1)
    def test_invalid_vector_no_success_receipt(self):
        for vector in ([0.]*511,[float('nan')]*512):
            with self.assertRaises(ValueError):runner.persist_vector(self.folder,1,'ground',self.frame,vector)
        self.assertEqual(list(self.folder.iterdir()),[])
    def test_failed_forward_has_no_success_receipt(self):
        from project.src.agents.frozen_crossview_encoder_v2 import FrozenCrossViewEncoder
        cls=runner.make_journaled_encoder();encoder=cls.__new__(cls);encoder.calls=1
        with patch.object(FrozenCrossViewEncoder,'encode',side_effect=RuntimeError('forward failed')) as fake,patch.object(runner,'OUT',self.folder):
            with self.assertRaises(RuntimeError):encoder.encode(self.frame,'ground')
        fake.assert_called_once();self.assertEqual(list(self.folder.iterdir()),[])

if __name__=='__main__':unittest.main()
