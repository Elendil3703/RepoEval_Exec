import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestRepoEvalInitResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[132]  # Get the 133rd JSON element (index 132)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 133rd JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect testing results to write to JSONL

        sop_class_code = """
class SOp:
    pass
"""
        predicate_class_code = """
class Predicate:
    pass
"""

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                init_pattern = r"def\s+__init__\s*\(.*?\):"
                if not re.search(init_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature or missing '__init__'.\n")
                    failed_count += 1
                    # Write failed record
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Prepare the execution environment
                exec_globals = {
                    'SOp': None,
                    'Predicate': None,
                    'Any': Any,  # Inject Any
                }
                exec_locals = {}

                try:
                    # Execute code for supporting classes
                    exec(sop_class_code, exec_globals)
                    exec(predicate_class_code, exec_globals)
                    
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Test instantiation of the class
                    test_instance = exec_locals['__class__'](exec_globals['SOp'](), exec_globals['SOp'](), exec_globals['Predicate']())

                    # Assertions to check the initialization
                    self.assertIsInstance(test_instance.keys, exec_globals['SOp'], f"Code snippet {i} did not set 'keys' as SOp instance.")
                    self.assertIsInstance(test_instance.queries, exec_globals['SOp'], f"Code snippet {i} did not set 'queries' as SOp instance.")
                    self.assertIsInstance(test_instance.predicate, exec_globals['Predicate'], f"Code snippet {i} did not set 'predicate' as Predicate instance.")

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

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        existing_records.append(json.loads(line))

        # Remove old records for the __init__ function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # Append new results
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()