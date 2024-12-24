import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[30]  # Get the 31st JSON element (index 30)

    def test_init_function(self):
        """Test the __init__ function with different scenarios."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        # Dynamic test
        exec_globals = {
            'transformers': __import__('transformers'),
            'hf_get_branch_class': lambda config: type('BranchClass', (object,), {}),
            'Any': Any,
        }
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check for __init__ presence
            init_func = exec_locals.get('__init__', None)
            self.assertIsNotNone(init_func, "__init__ function not defined.")

            # Define a mock PreTrainedModel
            class MockPreTrainedModel:
                config = {}

            # Test cases
            test_cases = [
                {"num_layers_unfrozen": 0, "expected_frozen_head": False},
                {"num_layers_unfrozen": 3, "expected_frozen_head": True},
                {"num_layers_unfrozen": -1, "expected_frozen_head": False},
            ]

            for i, case in enumerate(test_cases):
                print(f"Running test case {i}...")
                instance = exec_locals["__init__"](MockPreTrainedModel, num_layers_unfrozen=case["num_layers_unfrozen"])
                
                # Check if frozen_head is set based on the condition
                has_frozen_head = hasattr(instance, "frozen_head")
                self.assertEqual(
                    has_frozen_head,
                    case["expected_frozen_head"],
                    f"Test case {i} failed: frozen_head existence mismatch."
                )

                print(f"Test case {i}: PASSED.")

                passed_count += 1
                results.append({
                    "function_name": "__init__",
                    "code": self.code_snippet,
                    "result": "passed"
                })

        except Exception as e:
            print(f"Test case failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.")

        # Writing results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_records.append(json.loads(line))

        # Remove old records for __init__ function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()