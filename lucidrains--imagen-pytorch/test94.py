import unittest
import json
import os
import inspect
from types import SimpleNamespace

TEST_RESULT_JSONL = "test_result.jsonl"

class TestDefaultFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[93]  # Get the 94th JSON element (index 93)
        if not cls.code_snippet:
            raise ValueError("Expected code snippet in the 94th JSON element")

    def test_default_function(self):
        """Test the default function for various cases."""
        code = self.code_snippet
        results = []  # Collect test results to write to JSONL

        exec_globals = {}
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            if 'default' not in exec_locals:
                raise ValueError("Function 'default' not found in the executed locals.")

            # Get the default function from the executed code
            default = exec_locals['default']

            # Test cases
            test_cases = [
                (42, lambda: 100, 42),
                (None, lambda: 100, 100),
                (None, 200, 200),
                (0, 50, 0),
                ('', 'default', ''),
                (None, 'default', 'default')
            ]

            # Run the tests
            for idx, (val, d, expected) in enumerate(test_cases):
                with self.subTest(test_case_idx=idx):
                    result = default(val, d)
                    self.assertEqual(
                        result,
                        expected,
                        f"Test case {idx} failed: default({val}, {d}) returned {result}, expected {expected}."
                    )

            results.append({
                "function_name": "default",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Function default: FAILED with error: {e}")
            results.append({
                "function_name": "default",
                "code": code,
                "result": "failed"
            })

        # ============= Write Test Results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "default"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "default"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()