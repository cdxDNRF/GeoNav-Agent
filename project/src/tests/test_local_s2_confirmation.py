"""Guard preregistered three-seed decisions against best-run and ablation selection."""
import copy
from pathlib import Path
import sys
import unittest

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from eval.local_s2_confirmation import (MAIN_ARMS, specifications, choose_default,
    summarize, aggregate, variants_for, engineering_checks)
from eval.audit_local_s2_confirmation import reproduce_selection, verify_engineering


def result(sr,sg=1.2):
    m={'episodes':100,'successes':round(100*sr),'sr':sr,'mean_sg_all_episodes':sg,
       'repeat_visit_rate_micro':.03,'out_of_bounds_rate':0}
    grouped={'sr':sr,'mean_sg_all_episodes':sg}
    return {'metrics':m,'by_source':{a:dict(grouped) for a in ('a','b','c','d')},
            'by_distance':{str(d):dict(grouped) for d in range(4,9)}}


class ConfirmationTests(unittest.TestCase):
    def setUp(self):
        self.main={'Small256_Full':[result(.65)]*3,
                   'Spatial256_Full':[result(.69,1.1),result(.67,1.1),result(.70,1.1)],
                   'Small256_NoTarget':[result(.75,1)]*3,
                   'Spatial256_NoTarget':[result(.76,1)]*3,
                   'HistoricalBoundary':[result(.66)]*3}
        self.rule={'sr':.53,'mean_sg_all_episodes':2.11}

    def test_fixed_plan_has_exact_denominators_and_never_selects_MeanCue(self):
        plans=specifications()
        self.assertEqual(len(plans),15)
        self.assertEqual(len({p['folder'] for p in plans}),15)
        self.assertEqual(sum(len(p['conditions']) for p in plans),27)
        self.assertTrue(all(tuple(p['conditions'])==variants_for(p['arm']) for p in plans))
        choice=choose_default(self.main,self.rule)
        self.assertEqual(choice['proposed_default'],'Small256_NoTarget')
        self.assertEqual(choice['reference_seed'],0)

    def test_smaller_NoTarget_wins_when_extra_spatial_gain_is_below_2pp(self):
        choice=choose_default(self.main,self.rule)
        self.assertEqual(choice['no_target_winner'],'Small256_NoTarget')
        self.assertEqual(choice['full_winner'],'Spatial256_Full')
        self.assertEqual(reproduce_selection(self.main,self.rule,choice),'Small256_NoTarget')

    def test_one_high_seed_does_not_override_two_negative_seeds(self):
        self.main['Spatial256_Full']=[result(.95,1),result(.64,1),result(.64,1)]
        choice=choose_default(self.main,self.rule)
        self.assertEqual(choice['full_winner'],'Small256_Full')

    def test_worse_SG_and_distance_failure_prevent_default_replacement(self):
        self.main['Small256_NoTarget']=[result(.76,2)]*3
        self.main['Spatial256_NoTarget']=[result(.77,2)]*3
        choice=choose_default(self.main,self.rule)
        self.assertEqual(choice['engineering_candidate'],'Spatial256_Full')
        self.assertEqual(choice['proposed_default'],'Spatial256_Full')
        self.main['HistoricalBoundary']=[result(.70)]*3
        choice=choose_default(self.main,self.rule)
        self.assertEqual(choice['proposed_default'],'HistoricalBoundary')
        self.setUp()
        self.main['Small256_NoTarget']=copy.deepcopy(self.main['Small256_NoTarget'])
        for item in self.main['Small256_NoTarget']:
            item['by_distance']['4']['sr']=.2
        choice=choose_default(self.main,self.rule)
        self.assertFalse(choice['engineering_candidate_checks']['each_distance_SR_30pct'])
        self.assertEqual(choice['proposed_default'],'HistoricalBoundary')

    def test_missing_seed_and_mismatched_task_denominators_are_rejected(self):
        with self.assertRaises(ValueError):aggregate(self.main['Small256_Full'][:2])
        runs=[result(.65) for _ in range(3)];runs[0]['metrics']['episodes']=99
        with self.assertRaises(ValueError):aggregate(runs)

    def test_target_evidence_does_not_pass_because_navigation_SR_is_high(self):
        results={}
        for arm in MAIN_ARMS:
            for seed in range(3):
                for c in variants_for(arm):
                    results[arm,seed,c]=self.main[arm][seed] if c in ('Full','NoTarget') else result(.8,.8)
        summary=summarize(results,{'Frontier':self.rule,'FixedRegion':{'sr':.29,'mean_sg_all_episodes':2.3}})
        self.assertFalse(summary['local_visual_S2_numeric_and_target_checks_passed'])
        self.assertFalse(summary['formal_S2_passed'])
        self.assertEqual(summary['default_selection']['proposed_default'],'Small256_NoTarget')

    def test_independent_selection_check_rejects_changed_default_or_gate(self):
        choice=choose_default(self.main,self.rule)
        corrupt=copy.deepcopy(choice);corrupt['proposed_default']='Spatial256_NoTarget'
        with self.assertRaises(AssertionError):reproduce_selection(self.main,self.rule,corrupt)
        checks=engineering_checks(self.main['Small256_NoTarget'],self.rule)
        verify_engineering(self.main['Small256_NoTarget'],self.rule,checks)
        checks['mean_SR_60pct']=False
        with self.assertRaises(AssertionError):verify_engineering(self.main['Small256_NoTarget'],self.rule,checks)


if __name__=='__main__':unittest.main()
