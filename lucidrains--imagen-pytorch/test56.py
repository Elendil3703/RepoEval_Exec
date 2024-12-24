import unittest
import json
import os
from typing import Any  # Ensure Any is available in the injected environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPadTupleToLength(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[55]  # Get the 56th JSON element
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the 56th JSON entry")

    def test_code_snippet(self):
        """Dynamically test the pad_tuple_to_length function."""
        code = self.code_snippet
        results = []  # Collect test results to write into JSONL

        # ------------------- Dynamic Execution Logic -------------------
        exec_globals = {
            'Any': Any,  # Inject Any
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if pad_tuple_to_length is defined
            if 'pad_tuple_to_length' not in exec_locals:
                raise AssertionError("Function 'pad_tuple_to_length' not found in executed locals.")

            # Retrieve the function
            pad_tuple_to_length = exec_locals['pad_tuple_to_length']

            # Define test cases
            test_cases = [
                ((1, 2), 4, 0, (1, 2, 0, 0)),   # Tuple shorter than length, fill with 0
                ((1, 2, 3), 3, None, (1, 2, 3)), # Tuple already at requested length
                ((1,), 2, 'a', (1, 'a')),        # Tuple shorter than length, fill with 'a'
                ((1, 2, 3), 2, None, (1, 2)),    # Tuple longer than length, should truncate
                ((), 3, 5, (5, 5, 5)),           # Empty tuple, fill entirely
            ]

            # Run test cases
            for i, (t, length, fillvalue, expected) in enumerate(test_cases):
                result = pad_tuple_to_length(t, length, fillvalue)
                self.assertEqual(result, expected, f"Failed on test case {i}: {t}, {length}, {fillvalue}")
                print(f"Test case {i} PASSED: {t}, {length}, {fillvalue} -> {result}")

            results.append({
                "function_name": "pad_tuple_to_length",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet failed with error: {e}")
            results.append({
                "function_name": "pad_tuple_to_length",
                "code": code,
                "result": "failed",
                "error": str(e)
            })

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                existing_records = [json.loads(line) for line in f if line.strip()]

        existing_records = [rec for rec in existing_records if rec.get("function_name") != "pad_tuple_to_length"]
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()