import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthCall(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and get the 132nd code snippet (index 131)
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[131]  # Get the 132nd element
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet at index 131")

    def test_caller_function(self):
        """Test the __call__ function from the code snippet."""
        
        results = []  # Collect results for test_result.jsonl
        code = self.code_snippet
        func_name = "__call__"

        # Dynamic setup of execution context
        exec_globals = {
            'Value': Any,
            '_comparison_table': {},
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if __call__ is implemented correctly
            if func_name not in exec_locals:
                raise ValueError(f"Function '{func_name}' is not defined.")

            # Simulate a realistic _comparison_table
            def mock_comparison(key, query):
                return key == query

            exec_globals['_comparison_table'] = {exec_locals['self']: mock_comparison}

            # Test different scenarios
            call_fn = exec_locals[func_name]

            # Case: key is None
            with self.assertRaises(ValueError) as cm:
                call_fn(None, "query_value")
            self.assertEqual(str(cm.exception), "key is None!")

            # Case: query is None
            with self.assertRaises(ValueError) as cm:
                call_fn("key_value", None)
            self.assertEqual(str(cm.exception), "query is None!")

            # Case: valid values and function returns True
            result = call_fn("key_value", "key_value")
            self.assertTrue(result)

            # Case: valid values and function returns False
            result = call_fn("key_value", "different_value")
            self.assertFalse(result)

            # If we reach here, all assertions passed
            results.append({
                "function_name": func_name,
                "code": code,
                "result": "passed"
            })
            print(f"Test for {func_name}: PASSED all assertions.\n")

        except Exception as e:
            results.append({
                "function_name": func_name,
                "code": code,
                "result": "failed"
            })
            print(f"Test for {func_name}: FAILED with error: {e}\n")

        # ============= Write the test results into test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for __call__
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != func_name
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()