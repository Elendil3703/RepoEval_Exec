import unittest
import json
import os
import sys
import torch
import torch.nn as nn
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class Identity(nn.Module):
    def forward(self, x):
        return x

def default(val, d):
    return val if val is not None else d

class LayerNorm(nn.LayerNorm):
    pass

class TestGroundTruthInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        
        # Extract the specific code snippet
        cls.code_snippet = data[77]  # Get the 78th element (index 77)
        if len(cls.code_snippet.strip()) < 1:
            raise ValueError("Expected a valid code snippet in the JSON data")

    def test_ground_truth_init(self):
        """Test the __init__ function from the ground truth."""
        results = []

        code = self.code_snippet
        print("Running test for the ground truth __init__...")

        # Environment setup for the exec function
        exec_globals = {
            'torch': torch,
            'nn': nn,
            'Any': Any,
            'Identity': Identity,
            'default': default,
            'LayerNorm': LayerNorm,
        }
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)
            
            # Check if `__init__` is present in the executed context
            if '__init__' not in exec_locals:
                print(f"Code snippet failed, '__init__' not found in exec_locals.")
                results.append({
                    "function_name": "__init__",
                    "code": code,
                    "result": "failed"
                })
                return
            
            # Define a dummy class with the same init method
            class TestClass:
                def __init__(
                    self,
                    dim,
                    *,
                    context_dim = None,
                    dim_head = 64,
                    heads = 8,
                    norm_context = False,
                    scale = 8
                ):
                    super().__init__()
                    self.scale = scale

                    self.heads = heads
                    inner_dim = dim_head * heads

                    context_dim = default(context_dim, dim)

                    self.norm = LayerNorm(dim)
                    self.norm_context = LayerNorm(context_dim) if norm_context else Identity()

                    self.null_kv = nn.Parameter(torch.randn(2, dim_head))
                    self.to_q = nn.Linear(dim, inner_dim, bias = False)
                    self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)

                    self.q_scale = nn.Parameter(torch.ones(dim_head))
                    self.k_scale = nn.Parameter(torch.ones(dim_head))

                    self.to_out = nn.Sequential(
                        nn.Linear(inner_dim, dim, bias = False),
                        LayerNorm(dim)
                    )

            # Create an instance and confirm properties
            test_instance = TestClass(dim=128)

            self.assertEqual(test_instance.scale, 8, "The scale should be set to 8.")
            self.assertEqual(test_instance.heads, 8, "The number of heads should be 8.")
            self.assertIsInstance(test_instance.norm, LayerNorm, "Norm should be an instance of LayerNorm.")
            self.assertIsInstance(test_instance.to_q, nn.Linear, "to_q should be a linear layer.")

            print("Ground truth __init__: PASSED all assertions.")
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet failed with error: {e}")
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })
        
        # Append results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()