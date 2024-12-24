import unittest
import json
import sys
import os
from typing import Union, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class SOp:
    pass

class SequenceMap:
    def __init__(self, func, a, b):
        pass

class Map:
    def __init__(self, func, a):
        pass

class NumericValue:
    pass

class TestSOpMulFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[120]  # Get the 121st JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 121st JSON array")

    def test_mul_function(self):
        """Dynamically test all code snippets in the JSON for __mul__ implementation."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # ------------------- Static Checks -------------------
                if "__mul__" not in code:
                    print(f"Code snippet {i}: FAILED, '__mul__' method not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__mul__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "return SequenceMap" not in code or "return Map" not in code:
                    print(f"Code snippet {i}: FAILED, expected return pattern not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__mul__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Testing -------------------
                exec_globals = {
                    'Union': Union,
                    'SOp': SOp,
                    'NumericValue': NumericValue,
                    'SequenceMap': SequenceMap,
                    'Map': Map,
                }
                exec_locals = {}

                try:
                    # Dynamic execution of the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if __mul__ exists in exec_locals
                    sop_instance = SOp()
                    other_sop = SOp()
                    
                    result = sop_instance.__mul__(other_sop)
                    self.assertIsInstance(
                        result,
                        SequenceMap,
                        f"Code snippet {i} did not return SequenceMap when 'other' is SOp."
                    )
                    
                    other_value = 5
                    result = sop_instance.__mul__(other_value)
                    self.assertIsInstance(
                        result,
                        Map,
                        f"Code snippet {i} did not return Map when 'other' is NumericValue."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__mul__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__mul__",
                        "code": code,
                        "result": "failed"
                    })

        # Final Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__mul__"
        ]

        # Append new results
        existing_records.extend(results)

        # Re-write test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()