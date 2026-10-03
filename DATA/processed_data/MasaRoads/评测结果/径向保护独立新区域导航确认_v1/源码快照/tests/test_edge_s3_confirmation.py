"""S3-specific count, source isolation, hard-gate and test-image regressions."""
from dataclasses import replace
from pathlib import Path
import sys
import tempfile
import unittest
from types import SimpleNamespace

import numpy as np
from PIL import Image

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from env.environment import image_payload
from env.episode import Episode
from eval.edge_s2_confirmation import given_target_payload
from eval.edge_s3_confirmation import protocol_info,isolated_splits,engineering_checks,summarize,interval


def run(sr=.55,sg=1.,repeat=.05,n=250):
    return dict(metrics=dict(episodes=n,successes=round(sr*n),sr=sr,mean_sg_all_episodes=sg,
        repeat_visit_rate_micro=repeat,out_of_bounds_rate=0.),
        by_distance={str(d):dict(sr=sr,mean_sg_all_episodes=sg) for d in range(4,9)},
        by_source={f'img_{i}':dict(sr=sr,mean_sg_all_episodes=sg) for i in range(10)})


class EdgeS3Tests(unittest.TestCase):
    def test_wrong_target_uses_test_when_area_and_cell_names_collide(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for s,value in [('train',11),('val',99),('test',211)]:
                p=root/f'patches/{s}/img_0';p.mkdir(parents=True)
                Image.fromarray(np.full((300,300,3),value,np.uint8)).save(p/'patch_3.jpg')
            ep=Episode('probe','test','img_0',0,4,4)
            actual=given_target_payload(ep,'CueWrong',{'probe':dict(cue_cell=3)},root)
            self.assertEqual(actual,image_payload(root/'patches/test/img_0/patch_3.jpg'))
            for s in ('train','val'):self.assertNotEqual(actual,image_payload(root/f'patches/{s}/img_0/patch_3.jpg'))

    def test_250_balanced_tasks_preserve_duplicate_routes(self):
        eps=[Episode(f'{i}-{d}-{j}','test',f'img_{i}',0,4+(d-4)*5,d,source_tile=f'map{i}')
             for i in range(10) for d in range(4,9) for j in range(5)]
        for e in eps:e.validate()
        self.assertEqual(protocol_info(eps)['unique_routes'],50)
        self.assertEqual(protocol_info(eps)['duplicate_draws'],200)
        for bad in (eps[:-1],eps[:-1]+[eps[0]],eps[:-1]+[replace(eps[-1],split='val')],
            eps[:-1]+[replace(eps[-1],dist=7)]):
            with self.assertRaises(ValueError):protocol_info(bad)

    def test_all_three_split_pairs_require_filename_and_patch_hash_isolation(self):
        eps={s:[SimpleNamespace(source_tile=s+'_map')] for s in ('train','val','test')}
        ms={s:dict(area_sha256={'img_0':s+'_hash'}) for s in eps}
        self.assertEqual(len(isolated_splits(eps,ms)),3)
        eps['test'][0].source_tile='val_map'
        with self.assertRaises(ValueError):isolated_splits(eps,ms)
        eps['test'][0].source_tile='test_map';ms['test']['area_sha256']['img_0']='train_hash'
        with self.assertRaises(ValueError):isolated_splits(eps,ms)

    def test_s3_thresholds_are_inclusive_and_sg_is_an_independent_gate(self):
        main=[run(sg=2.,repeat=.10) for _ in range(3)];rule=run(.50,2.)
        self.assertTrue(all(engineering_checks(main,rule).values()))
        main[0]['metrics']['mean_sg_all_episodes']=2.01
        self.assertFalse(engineering_checks(main,rule)['mean_SG_2p0'])
        main=[run(.549) for _ in range(3)]
        self.assertFalse(engineering_checks(main,rule)['mean_SR_55pct'])

    def test_each_seed_must_finish_all_planned_tasks(self):
        main=[run() for _ in range(3)];main[2]['metrics']['episodes']=249
        # A missing record fails, rather than being silently treated as budget exhaustion.
        with self.assertRaises(ValueError):engineering_checks(main,run(.5,2.))
        main=[run(n=249) for _ in range(3)]
        self.assertFalse(engineering_checks(main,run(.5,2.))['all_250_normal_terminals'])

    def test_positive_sources_and_revisit_are_hard_s3_gates(self):
        main=[run(.7) for _ in range(3)];rule=run(.5,2.)
        for r in main:
            for i in range(6,10):r['by_source'][f'img_{i}']['sr']=.5
        self.assertTrue(engineering_checks(main,rule)['six_of_ten_sources_positive'])
        for r in main:r['by_source']['img_5']['sr']=.5
        self.assertFalse(engineering_checks(main,rule)['six_of_ten_sources_positive'])
        main[0]['metrics']['repeat_visit_rate_micro']=.101
        self.assertFalse(engineering_checks(main,rule)['each_seed_revisit_10pct'])

    def test_target_evidence_is_separate_from_engineering_pass(self):
        records={}
        for s in range(3):
            for c in ('Baseline','CueFull','CueMean','CueWrong'):records['Edge',s,c]=run(.55 if c=='CueFull' else .52)
            records['ZeroEdge',s,'CueFull']=run(.52)
        summary=summarize(records,{'Frontier':run(.45,2.),'FixedRegion':run(.3,3.)})
        self.assertTrue(summary['formal_S3_numeric_checks_passed'])
        self.assertFalse(summary['target_transfer_numeric_checks_passed'])
        self.assertFalse(summary['formal_S3_passed']) # Receipt requires completed audit.
        self.assertEqual(summary['effects']['Frontier']['source_interval']['source_count'],10)

    def test_cluster_interval_averages_seed_pairs_before_resampling(self):
        left=[run(.7),run(.6),run(.5)];right=[run(.4)]*3
        first=interval(left,right,'sr')
        for v in first['interval95']:self.assertAlmostEqual(v,.2)
        # Replicated rule references do not become thirty independent source groups.
        self.assertEqual(first['source_count'],10)
        self.assertEqual(first,interval(list(reversed(left)),right,'sr'))


if __name__=='__main__':unittest.main()
