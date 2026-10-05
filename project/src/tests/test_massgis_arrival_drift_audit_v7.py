import ast
import importlib.util
from pathlib import Path
import unittest

MODULE=Path(__file__).resolve().parents[1]/"eval"/"audit_massgis_arrival_drift_v7.py"
SPEC=importlib.util.spec_from_file_location("massgis_arrival_drift_audit_v7",MODULE)
AUDIT=importlib.util.module_from_spec(SPEC);SPEC.loader.exec_module(AUDIT)


class IndependentAuditTests(unittest.TestCase):
    def test_pair_index_has_one_entry_per_paired_row_without_grid_field(self):
        rows=[{"policy":"M0","seed":0,"pair_id":"p0","quality10":"toward","quality15":"away"}]
        indexed=AUDIT.pair_detail_index(rows)
        self.assertEqual(indexed[("M0",0,"p0")]["quality10"],"toward")
        self.assertEqual(len(indexed),1)
        with self.assertRaises(ValueError):
            AUDIT.pair_detail_index(rows+rows)

    def test_scalar_reconstruction_hash_is_stable(self):
        import numpy as np
        mean=np.zeros(512,dtype=np.float32);current=np.ones(512,dtype=np.float32)
        h=AUDIT.feature_input_hash(mean,current,(1,7),20,(17,),10)
        self.assertEqual(len(h),64)
        self.assertEqual(h,AUDIT.feature_input_hash(mean,current,(1,7),20,(17,),10))

    def test_auditor_has_no_model_runtime_or_forward_calls(self):
        tree=ast.parse(MODULE.read_text(encoding="utf-8-sig"))
        imports=set();forbidden=set()
        for node in ast.walk(tree):
            if isinstance(node,ast.Import): imports.update(x.name.split(".")[0] for x in node.names)
            elif isinstance(node,ast.ImportFrom) and node.module: imports.add(node.module.split(".")[0])
            elif isinstance(node,ast.Call):
                name=node.func.attr.lower() if isinstance(node.func,ast.Attribute) else node.func.id.lower() if isinstance(node.func,ast.Name) else ""
                if name in {"forward","predict","act","load_area_policy","step"}: forbidden.add(name)
        self.assertFalse(imports&{"torch","torchvision","transformers"})
        self.assertFalse(forbidden)


if __name__=="__main__": unittest.main()

