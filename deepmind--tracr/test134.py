import unittest
import json
import os
from typing import Any  # Ensure Any is available for injected code

TEST_RESULT_JSONL = "test_result.jsonl"

class Selector:
    pass  # Placeholder for the Selector class

class TestGroundTruthInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[133]  # Get the 134th JSON element (0-indexed, so 133)

        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 134th JSON array")

    def test_init_function(self):
        """Dynamically test all code snippets in the JSON related to __init__."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to be written to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                exec_globals = {
                    'Selector': Selector,  # Inject Selector class
                    'Any': Any,            # Inject Any
                }
                exec_locals = {}

                try:
                    # Attempt to dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure the class with __init__ exists
                    if '__init__' not in code:
                        print(f"Code snippet {i}: FAILED, '__init__' not found in code.\n")
                        failed_count += 1
                        # Record failure result
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Attempt to instantiate the class
                    selector_instance = Selector()
                    class_to_test = next(cls for cls in exec_locals.values() if isinstance(cls, type))

                    # Create an instance and check if the initializer operates correctly
                    instance = class_to_test(selector_instance)

                    # Check if 'selector' attribute is correctly set
                    self.assertIs(
                        instance.selector,
                        selector_instance,
                        f"Code snippet {i}: 'selector' attribute not correctly set in initializer.",
                    )
                    self.assertIsInstance(
                        instance.selector,
                        Selector,
                        f"Code snippet {i}: 'selector' is not an instance of Selector.",
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Append results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for __init__
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
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