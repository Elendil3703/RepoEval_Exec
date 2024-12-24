import unittest
import json
import os
import sys
import re
from typing import Any
import rasp  # Ensuring rasp module is available in the environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestShiftByFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[179]  # Get the 180th JSON element (index 179)

    def test_shift_by(self):
        """Test the shift_by function with various cases."""
        if "def shift_by" not in self.code_snippet:
            raise ValueError("The function 'shift_by' was not found in the code snippet")

        func_pattern = r"def\s+shift_by\s*\("
        if not re.search(func_pattern, self.code_snippet):
            raise ValueError("Incorrect signature for 'shift_by' function")

        # Prepare the environment for exec
        exec_globals = {
            'rasp': rasp,  # Inject rasp
            'Any': Any,
        }
        exec_locals = {}

        # Compile and execute the code snippet
        exec(self.code_snippet, exec_globals, exec_locals)

        # Retrieve the shift_by function
        if 'shift_by' not in exec_locals:
            raise ValueError("'shift_by' function was not found after execution")

        shift_by = exec_locals['shift_by']

        results = []  # Collect the results for JSONL output
        passed_count = 0
        failed_count = 0

        # Define test cases
        test_cases = [
            (0, rasp.SOp(), "shift_by(0)"),  # Check no shift
            (1, rasp.SOp(), "shift_by(1)"),  # Check positive shift
            (-1, rasp.SOp(), "shift_by(-1)"),  # Check negative shift
        ]

        for offset, sop, expected_name in test_cases:
            with self.subTest(offset=offset):
                try:
                    result = shift_by(offset, sop)
                    self.assertIsInstance(result, rasp.SOp, "Result is not a rasp.SOp instance")
                    self.assertEqual(result.name, expected_name, f"Shifted SOp name is not '{expected_name}'")

                    passed_count += 1
                    results.append({
                        "function_name": "shift_by",
                        "offset": offset,
                        "result": "passed"
                    })
                except Exception as e:
                    failed_count += 1
                    results.append({
                        "function_name": "shift_by",
                        "offset": offset,
                        "result": "failed",
                        "error": str(e)  # Capture the error
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.")

        # Ensure total test count matches
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # ============= Write results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "shift_by"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "shift_by"
        ]

        # Append the new results
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()