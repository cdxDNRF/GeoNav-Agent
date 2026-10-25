"""Meaningful offline boundary checks: wrong/missing records and repeated physical state."""
import ast
import copy
import hashlib
import unittest
from pathlib import Path

from eval.crossview_evidence_close_v1 import counts, summarize_current, validate_row, text_json, digest


def row(pair, truth=0):
    scores = [0.0]*25
    ranking = sorted(range(25),key=lambda j:hashlib.sha256(f'CV-002/tie/v1:{pair}:{j}'.encode()).hexdigest())
    rank=ranking.index(truth)+1
    return dict(pair_id=pair,scores=scores,ranking=ranking,truth_evaluator_only=truth,true_rank=rank,top1=rank==1,top5=rank<=5)


class EvidenceBoundaries(unittest.TestCase):
    def test_ties_do_not_use_answer(self):
        a,b=row('same',0),row('same',24)
        self.assertEqual(a['ranking'],b['ranking'])
        validate_row(a);validate_row(b)

    def test_invalid_scores_and_rank_rejected(self):
        for bad in ['nan','permutation','truth','count']:
            x=row('x')
            if bad=='nan':x['scores'][0]=float('nan')
            if bad=='permutation':x['ranking'][1]=x['ranking'][0]
            if bad=='truth':x['truth_evaluator_only']=25
            if bad=='count':x['top1']=not x['top1']
            with self.assertRaises(ValueError):validate_row(x)

    def test_missing_or_duplicate_denominator_rejected(self):
        with self.assertRaises(ValueError):counts([row('x'),row('x')])
        with self.assertRaises(ValueError):summarize_current([],{'development_tasks':[]})

    def test_raw_feature_model_access_rejected(self):
        for p in ['DATA/raw_data/MM-GAG/anything.json','models/config.json','DATA/processed_data/MM-GAG/x.npy']:
            with self.assertRaises(ValueError):digest(p)
        with self.assertRaises(ValueError):text_json('DATA/raw_data/MM-GAG/a.json')

    def test_numeric_inputs_are_saved_decisions_not_model_modules(self):
        p=Path(__file__).resolve().parents[1]/'eval/crossview_evidence_close_v1.py'
        tree=ast.parse(p.read_text(encoding='utf-8'))
        imports=[n.module for n in ast.walk(tree) if isinstance(n,ast.ImportFrom)]
        imports += [a.name for n in ast.walk(tree) if isinstance(n,ast.Import) for a in n.names]
        self.assertFalse(any(x and x.split('.')[0] in {'torch','transformers','numpy','PIL','requests','agents','train'} for x in imports))

    def test_real_queue_is_metadata_only_and_duplicate_record_fails(self):
        # Existing JSON only; no pixel, feature or checkpoint read.
        q=text_json('DATA/processed_data/MM-GAG/同任务对齐训练与目标对照协议_v1/固定编码与训练开发队列.json')
        rows=text_json('DATA/processed_data/MM-GAG/同任务拍摄点对齐执行_v1/逐题排序结果.json')['rows']
        bad=copy.deepcopy(rows);bad[-1]=bad[0]
        with self.assertRaises(ValueError):summarize_current(bad,q)
        q=copy.deepcopy(q);q['development_tasks'][0]['aerial_feature_indices'][0]+=1
        with self.assertRaises(ValueError):summarize_current(rows,q)


if __name__=='__main__':unittest.main()
