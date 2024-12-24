import unittest
import json
import os
import torch
from torch import nn

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the RepoEval_result.json file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[62]  # Select the 63rd code example

    def test_init_function(self):
        """Dynamically test the __init__ function code snippet with custom checks."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet  # Get the specified code

        # Preparing environment for execution
        exec_globals = {
            'torch': torch,
            'nn': nn,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(f"class TestClass:\n    {code}", exec_globals, exec_locals)

            # Get the dynamically created TestClass
            TestClass = exec_locals['TestClass']

            # Test 1: Check parameter initialization
            obj = TestClass(feats=5)  # Initialize with feats
            expected_shape = (5, *((1,) * (-obj.dim - 1)))
            self.assertTrue(hasattr(obj, 'g'), "__init__ did not create parameter 'g'.")
            self.assertEqual(obj.g.shape, expected_shape, "Parameter 'g' has incorrect shape.")

            # Test 2: Check whether 'stable' and 'dim' attributes are correctly set
            self.assertEqual(obj.stable, False, "Default value for 'stable' should be False.")
            self.assertEqual(obj.dim, -1, "Default value for 'dim' should be -1.")

            # Custom check with different initialization
            obj_stable = TestClass(feats=3, stable=True, dim=-2)
            self.assertEqual(obj_stable.stable, True, "'stable' value not correctly initialized.")
            self.assertEqual(obj_stable.dim, -2, "'dim' value not correctly initialized.")

            print("Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": code,
                "result": "failed"
            })

        # ============= Write results to test_result.jsonl =============
        # Read existing test_result.jsonl (if exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "__init__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # Append new results
        existing_records.extend(results)

        # Write back to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()