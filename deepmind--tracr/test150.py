import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestHasPrevFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[149]  # Get the 150th JSON element (index 149)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the specified JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "def has_prev" not in code:
                    print(f"Code snippet {i}: FAILED, function 'has_prev' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "has_prev",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and testing logic
                exec_globals = {
                    'Any': Any,  # Inject Any
                    'rasp': FakeRasp()  # Inject mock RASP library
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Test the presence of has_prev
                    if 'has_prev' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'has_prev' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "has_prev",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test has_prev function logic
                    sequence = exec_globals['rasp'].SOp(...)  # Define your sequence input
                    output = exec_locals['has_prev'](sequence)

                    # Validate the output
                    expected_output = ...  # Define the expected output
                    self.assertEqual(output, expected_output, f"Code snippet {i}: Output mismatch.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "has_prev",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "has_prev",
                        "code": code,
                        "result": "failed"
                    })

        # Test Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "has_prev"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "has_prev"
        ]

        # Append new results
        existing_records.extend(results)

        # Overwrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

# Mock RASP Library for testing purposes
class FakeRasp:
    class SOp:
        def __init__(self, *args):
            pass

    class SelectorAnd:
        def __init__(self, *args):
            pass

    class Select:
        def __init__(self, *args):
            pass

    class Aggregate:
        def __init__(self, *args, **kwargs):
            pass

    class Full:
        def __init__(self, *args):
            pass

    class Comparison:
        EQ = "eq"
        LT = "lt"

    indices = "indices"

if __name__ == "__main__":
    unittest.main()