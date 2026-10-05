import ast
import importlib.util
from pathlib import Path
import unittest

MODULE=Path(__file__).resolve().parents[1]/"eval"/"audit_massgis_arrival_drift_v11.py"
SPEC=importlib.util.spec_from_file_location("massgis_arrival_drift_audit_v11",MODULE)
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

    def test_v6_binding_uses_expected_seal_index_keys(self):
        binding=AUDIT.jread(AUDIT.V6/"输入绑定.json")
        self.assertEqual(len(binding["eval2_sealed_files_sha256"]),753)
        self.assertEqual(len(binding["eval2_frozen_sources_sha256"]),73)
        self.assertEqual(len(binding["eval2_selected_inputs_sha256"]),9)

    def test_virtual_evaluation_keys_are_not_treated_as_filesystem_paths(self):
        entries={
            "EVAL2-sealed:DATA/old/a.json":"a"*64,
            "EVAL2-source:DATA/old/source.py":"b"*64,
            "EVAL2-input:DATA/old/model.pt":"c"*64,
            "project/src/analysis.py":"d"*64,
        }
        self.assertEqual(AUDIT.real_input_entries(entries),{"project/src/analysis.py":"d"*64})
        self.assertEqual(AUDIT.logical_input_entries(entries,"EVAL2-sealed:"),{"DATA/old/a.json":"a"*64})
        self.assertEqual(AUDIT.logical_input_entries(entries,"EVAL2-source:"),{"DATA/old/source.py":"b"*64})
        self.assertEqual(AUDIT.logical_input_entries(entries,"EVAL2-input:"),{"DATA/old/model.pt":"c"*64})

    def test_context_reconstruction_is_28d_and_caps_visit_bucket(self):
        context=AUDIT.visit_context((1,7),12,(1,7,17,17),10)
        self.assertEqual(context.shape,(28,))
        self.assertAlmostEqual(float(context[0]),1/9)
        self.assertAlmostEqual(float(context[1]),7/9)
        self.assertAlmostEqual(float(context[2]),12/20)
        self.assertAlmostEqual(float(context[3]),1/3)
        self.assertEqual(float(context[3+3]),1.0)

    def test_quality_counters_keep_grid_counts_nested(self):
        quality=AUDIT.new_quality_counters()
        quality["M0"][10]["toward"]+=1
        quality["M0"][15]["away"]+=2
        quality["M0"]["pair"]["tie"]+=3
        self.assertEqual(quality["M0"][10]["toward"],1)
        self.assertEqual(quality["M0"][15]["away"],2)
        self.assertEqual(quality["M0"]["pair"]["tie"],3)


    def test_context_group_labels_are_not_double_prefixed(self):
        groups={"g10":[1,2,3],"g15":[4,5]}
        self.assertEqual(AUDIT.context_group_counts(groups),{"g10":3,"g15":2})

if __name__=="__main__": unittest.main()


