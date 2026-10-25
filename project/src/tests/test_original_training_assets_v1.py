"""Memory-only trust-boundary tests. No files, checkpoints, GPU or services."""
import json
import pickle
import sys
import unittest
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.original_training_assets_v1 import MAX_PICKLE, Symbol, static_pickle, primitive
from eval.audit_original_training_assets_v1 import validate_gate, training_primitives_from_opcodes


def missing_gate():
    return {"requirements": [{"id": k, "status": "missing"} for k in ["source_split", "encoder_recipe", "feature_bank", "training_run", "checkpoint_run_map", "paper_evaluation_map"]],
            "development_target_qualified": False, "new_source_isolation_verified": False}


class TrustTests(unittest.TestCase):
    def test_primitive_roundtrip_no_pickle_execution(self):
        example = {"epoch": 202, "global_step": 812, "pytorch-lightning_version": "2.2.1",
                   "loops": {"batches": [1, 2], "done": False}, "lr": 1e-5}
        data = pickle.dumps(example, protocol=2)
        decoded, _ = static_pickle(data)
        self.assertEqual(decoded, example)
        scalars, _ = training_primitives_from_opcodes(data)
        self.assertEqual(scalars, {k: example[k] for k in ("epoch", "global_step", "pytorch-lightning_version")})

    def test_unknown_callable_rejected_without_execution(self):
        # A crafted pickle would execute os.system via a regular unpickler.
        payload = b"\x80\x02cos\nsystem\nX\x04\x00\x00\x00echo\x85R."
        with self.assertRaisesRegex(ValueError, "GLOBAL"): static_pickle(payload)

    def test_tensor_global_is_inert(self):
        payload = b"\x80\x02ctorch._utils\n_rebuild_tensor_v2\n)R."
        obj, _ = static_pickle(payload)
        self.assertIsInstance(obj, Symbol)
        self.assertEqual(primitive(obj), {"omitted": "tensor"})

    def test_unsupported_opcode_and_trailing_payload(self):
        with self.assertRaises(ValueError): static_pickle(b"\x80\x02\x8f.")
        with self.assertRaises(ValueError): static_pickle(pickle.dumps(1, protocol=2)+b"trailing")

    def test_metadata_byte_cap(self):
        with self.assertRaises(ValueError): static_pickle(b"x"*(MAX_PICKLE+1))

    def test_missing_assets_block_even_fake_qualification(self):
        g=missing_gate(); g.update(development_target_qualified=True, new_source_isolation_verified=True)
        result=validate_gate(g)
        self.assertFalse(result["formal_execution_allowed"])
        self.assertEqual(len(result["missing"]), 6)

    def test_verified_label_without_evidence_rejected(self):
        g=missing_gate(); g["requirements"][0]["status"]="verified"
        with self.assertRaisesRegex(ValueError, "lacks"): validate_gate(g)

    def test_duplicate_requirement_rejected(self):
        g=missing_gate(); g["requirements"][-1]=g["requirements"][0]
        with self.assertRaisesRegex(ValueError, "duplicate"): validate_gate(g)

    def test_json_serialization_omits_tensor_marker(self):
        s=json.dumps(primitive({"a": Symbol("tensor"), "b": 812}))
        self.assertIn('"omitted": "tensor"',s)


if __name__ == "__main__": unittest.main()
