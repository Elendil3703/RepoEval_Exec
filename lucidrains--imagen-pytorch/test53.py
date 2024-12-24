import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

def exists(val):
    return val is not None

def default(val, default_val):
    return val if exists(val) else default_val

class TestCastTupleFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[52]  # Get the 53rd JSON element (index 52)
        if not cls.code_snippet:
            raise ValueError("Expected at least one code snippet in the 53rd JSON array")

    def test_cast_tuple(self):
        """Test 'cast_tuple' function from the code snippet with various cases."""
        results = []  # Collect results for JSONL writing
        passed_count = 0
        failed_count = 0

        # Prepare the environment for execution
        exec_globals = {
            'Any': Any,
            'exists': exists,
            'default': default,
        }
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Retrieve the cast_tuple function
            cast_tuple = exec_locals.get('cast_tuple')

            if not cast_tuple:
                raise ValueError("Function 'cast_tuple' not found in the executed code.")

            # Test cases
            test_cases = [
                (42, None, (42,)),
                ([1, 2, 3], None, (1, 2, 3)),
                (5, 3, (5, 5, 5)),
                ((1,), 1, (1,))
            ]

            for i, (val, length, expected) in enumerate(test_cases):
                with self.subTest(test_index=i):
                    print(f"Running test {i} with value: {val}, length: {length}")
                    result = cast_tuple(val, length)
                    self.assertEqual(result, expected, f"Test {i} failed")
                    passed_count += 1
                    results.append({
                        "function_name": "cast_tuple",
                        "test_index": i,
                        "input": {"val": val, "length": length},
                        "expected": expected,
                        "result": "passed"
                    })

        except Exception as e:
            print(f"Test failed with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "cast_tuple",
                "code": self.code_snippet,
                "result": "failed",
                "error": str(e)
            })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # Append results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for cast_tuple
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "cast_tuple"
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