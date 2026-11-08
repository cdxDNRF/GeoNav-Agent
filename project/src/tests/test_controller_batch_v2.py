"""4.17 regressions in memory; original fixture/code remain immutable."""
import copy
import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch
from project.src.tests.test_controller_hub_v1 import BatchHub,MemoryPath,SceneOwner,OWNER,ROOT
from project.src.platform_core.controller_batch_v2 import ControllerBatches
from project.src.platform_core.controller_hub_v1 import ControllerHub


class EvidenceHub(BatchHub):
    def __init__(self):super().__init__();self.owner.scenes.reap=lambda:None
    def create(self,spec,owner):
        v=super().create(spec,owner);raw=self.owner.scenes.runs[v['id']]
        raw.update(dataset_id='masa-s2',task=dict(task_key=spec['task_key']),seed=spec['seed'],budget=10,
            controller_class='model' if spec['controller']=='M0' else 'rule',definition=dict(input_sha256={}),readonly=False)
        return copy.deepcopy(raw)
    def step(self,rid,owner,kind):
        super().step(rid,owner,kind);raw=self.owner.scenes.runs[rid]
        if raw['status']=='completed':raw['result']['steps']=raw['step']
        return copy.deepcopy(raw)
    def save_run(self,rid,owner,kind):
        super().save_run(rid,owner,kind);self.saved[rid]['readonly']=True;return copy.deepcopy(self.saved[rid])


class FixTests(unittest.TestCase):
    def setUp(self):
        self.h=EvidenceHub();self.b=ControllerBatches(self.h,ROOT);self.files={};self.b._folder=lambda bid:MemoryPath(self.files,bid)
        def write(p,v):self.files[p.name]=json.dumps(v)
        self.patches=[patch('project.src.platform_core.controller_batch_v1.atomic',write),patch('project.src.platform_core.controller_batch_v1.durable_snapshot',write)]
        for p in self.patches:p.start()
    def tearDown(self):
        for p in self.patches:p.stop()
    def create(self):return self.b.create(dict(name='synthetic',dataset_id='masa-s2',task_keys=['t0'],controllers=['M0','frontier'],seed=0),OWNER)
    def complete(self):
        v=self.create()
        for _ in range(4):v=self.b.step(v['id'],OWNER)
        return v
    def catalog_folders(self,ids):
        folders=[SimpleNamespace(name=i,is_dir=lambda:True) for i in ids]
        self.b.directory=SimpleNamespace(exists=lambda:True,iterdir=lambda:iter(folders))
    def test_invalid_definition_top_level_isolated(self):
        for value in [None,[],42,'text',True]:
            with self.subTest(value=value):
                v=self.create();good=v['id'];bad='d'*32;self.files[bad+'/definition.json']=json.dumps(value);self.files[bad+'/progress.json']='{}'
                self.catalog_folders([bad,good]);c=self.b.catalog(OWNER)
                self.assertEqual(c['broken'],1);self.assertEqual([b['id'] for b in c['batches']],[good])
    def test_invalid_progress_top_level_isolated(self):
        for value in [None,[],42,'text',True]:
            with self.subTest(value=value):
                v=self.create();bid=v['id'];self.files[bid+'/progress.json']=json.dumps(value);self.catalog_folders([bid])
                self.assertEqual(self.b.catalog(OWNER)['broken'],1)
    def test_signed_wrong_shape_still_rejected(self):
        v=self.create();bid=v['id'];_,s=self.b._load(bid);s['rows']=[None,None];self.b._write(bid,s)
        with self.assertRaises(ValueError):self.b.state(bid,OWNER)
    def test_missing_earlier_trace_disables_metrics_and_export(self):
        v=self.complete();bid=v['id'];lost=v['rows'][0]['saved_ref']['id'];saved=self.h.saved.pop(lost);before=dict(self.files)
        x=self.b.state(bid,OWNER);self.assertFalse(x['evidence_complete']);self.assertFalse(x['controllable'])
        self.assertIsNone(x['metrics']['arms']['M0']['SR']);self.assertEqual(x['metrics']['arms']['M0']['planned'],1)
        self.assertEqual(x['cached_metrics']['arms']['M0']['SR'],1);self.assertFalse(x['rows'][0]['evidence_valid'])
        for fmt in ['json','csv']:
            with self.assertRaises(ValueError):self.b.export(bid,OWNER,fmt)
        self.assertEqual(self.files,before);self.h.saved[lost]=saved;x=self.b.state(bid,OWNER)
        self.assertTrue(x['evidence_complete']);self.assertEqual(x['metrics']['arms']['M0']['SR'],1)
    def test_missing_last_trace_does_not_break_inspection(self):
        v=self.complete();self.h.saved.pop(v['rows'][-1]['saved_ref']['id']);x=self.b.state(v['id'],OWNER)
        self.assertFalse(x['evidence_complete']);self.assertIsNone(x['current'])
    def test_changed_result_binding_detected(self):
        v=self.complete();self.h.saved[v['rows'][0]['saved_ref']['id']]['result']['SG_cells']=1
        self.assertFalse(self.b.state(v['id'],OWNER)['evidence_complete'])
    def test_changed_task_seed_controller_detected(self):
        for field,value in [('seed',1),('controller_class','mixed'),('task',dict(task_key='t1'))]:
            v=self.complete();self.h.saved[v['rows'][0]['saved_ref']['id']][field]=value
            self.assertFalse(self.b.state(v['id'],OWNER)['evidence_complete'])
    def test_missing_evidence_blocks_remaining_actions_without_overwrite(self):
        v=self.create();bid=v['id'];self.b.step(bid,OWNER);v=self.b.step(bid,OWNER);self.h.saved.pop(v['rows'][0]['saved_ref']['id']);before=dict(self.files)
        with self.assertRaises(ValueError):self.b.step(bid,OWNER)
        self.assertEqual(self.files,before);self.assertEqual(self.h.created,1)
    def test_complete_read_and_export_do_not_reexecute(self):
        v=self.complete();n=self.h.created;before=dict(self.files)
        self.assertTrue(self.b.state(v['id'],OWNER)['evidence_complete']);self.assertIn('SG_cells',self.b.export(v['id'],OWNER,'csv'));self.assertEqual(self.h.created,n);self.assertEqual(self.files,before)
    def test_catalog_marks_evidence_issue_and_keeps_other_batches(self):
        v=self.complete();self.h.saved.pop(v['rows'][0]['saved_ref']['id']);good=self.create();self.catalog_folders([v['id'],good['id']]);c=self.b.catalog(OWNER)
        self.assertEqual(c['broken'],1);self.assertEqual(len(c['batches']),2);self.assertFalse(c['batches'][0]['evidence_complete'])
    def test_real_expiry_before_step_allows_explicit_restart(self):
        w=SceneOwner();now=[0.];w.scenes.clock=lambda:now[0];w.masa=SimpleNamespace(bindings=lambda *a:{})
        b=ControllerBatches(ControllerHub(w),ROOT);b._folder=self.b._folder
        v=b.create(dict(name='TTL',dataset_id='masa-s2',task_keys=['t0'],controllers=['M0','frontier'],seed=0),OWNER);bid=v['id'];b.step(bid,OWNER);now[0]=181.
        with self.assertRaisesRegex(ValueError,'明确'):b.step(bid,OWNER)
        v=b.state(bid,OWNER);self.assertEqual(v['status'],'paused');self.assertTrue(v['needs_restart'])
        v=b.step(bid,OWNER,True);self.assertEqual(v['action_attempts'],2);self.assertEqual(v['rows'][0]['attempts'][0]['steps'],1)
        self.assertEqual(v['rows'][0]['step'],1)
    def test_real_expiry_before_state_is_visible_as_interruption(self):
        w=SceneOwner();now=[0.];w.scenes.clock=lambda:now[0];w.masa=SimpleNamespace(bindings=lambda *a:{})
        b=ControllerBatches(ControllerHub(w),ROOT);b._folder=self.b._folder
        v=b.create(dict(name='TTL',dataset_id='masa-s2',task_keys=['t0'],controllers=['M0'],seed=0),OWNER);b.step(v['id'],OWNER);now[0]=181.
        v=b.state(v['id'],OWNER);self.assertTrue(v['needs_restart']);self.assertEqual(v['status'],'paused')


if __name__=='__main__':unittest.main()
