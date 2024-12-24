import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPatchOptimizerResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[284]  # Get the 285th JSON element (index 284)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the JSON element")

    def test_code_snippet(self):
        """Dynamically test the code snippet for patch_optimizer function."""
        results = []  # Collect results to write to JSONL

        code = self.code_snippet

        # ------------------- Static Checks -------------------
        if "def patch_optimizer" not in code:
            print("FAILED: Function 'patch_optimizer' not found in code.")
            results.append({
                "function_name": "patch_optimizer",
                "code": code,
                "result": "failed"
            })
        else:
            exec_globals = {
                'ZeroRedundancyOptimizer': ZeroRedundancyOptimizerMock,  # Mock class
            }
            exec_locals = {}

            try:
                # Dynamically execute code snippet
                exec(code, exec_globals, exec_locals)

                if 'patch_optimizer' not in exec_locals:
                    print("FAILED: 'patch_optimizer' not found in exec_locals.")
                    results.append({
                        "function_name": "patch_optimizer",
                        "code": code,
                        "result": "failed"
                    })
                else:
                    # Use patch_optimizer and test its functionality
                    patch_optimizer = exec_locals['patch_optimizer']
                    opt = OptimizerMock()
                    params = [1, 2, 3]

                    # Test with is_zero=True
                    new_opt = patch_optimizer(opt, params, is_zero=True)
                    self.assertIsInstance(new_opt, ZeroRedundancyOptimizerMock)

                    # Test with is_zero=False
                    new_opt = patch_optimizer(opt, params, is_zero=False)
                    self.assertIsInstance(new_opt, OptimizerMock)

                    print("PASSED: All assertions for patch_optimizer.")
                    results.append({
                        "function_name": "patch_optimizer",
                        "code": code,
                        "result": "passed"
                    })
            except Exception as e:
                print(f"FAILED: Error occurred - {e}")
                results.append({
                    "function_name": "patch_optimizer",
                    "code": code,
                    "result": "failed"
                })

        # ============= Write results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for patch_optimizer
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "patch_optimizer"
        ]

        # Append new results
        existing_records.extend(results)

        # Write back to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

class ZeroRedundancyOptimizerMock:
    def __init__(self, params, optimizer_class, parameters_as_bucket_view, **defaults):
        self.params = params
        self.optimizer_class = optimizer_class
        self.parameters_as_bucket_view = parameters_as_bucket_view
        self.defaults = defaults

class OptimizerMock:
    def __init__(self):
        self.defaults = {}

if __name__ == "__main__":
    unittest.main()