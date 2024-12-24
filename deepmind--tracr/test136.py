import unittest
import json
import os
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class Selector:
    pass

class TestInitFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[135]  # Get the 136th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON data.")

    def test_init_function(self):
        """Test the __init__ function for the specified behavior."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Records to be written to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Static check: ensure the code snippet defines the __init__ function
                if "def __init__" not in code:
                    print(f"Code snippet {i}: FAILED, '__init__' function not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and testing logic
                exec_globals = {
                    'Selector': Selector,
                    'Any': Any,  # Inject Any if needed
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if __init__ exists after execution
                    if '__init__' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '__init__' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Simulate using the __init__ method with a correct and incorrect Selector instance
                    class TestClass(exec_locals['__init__']):
                        pass

                    # Correct usage
                    try:
                        test_instance = TestClass(Selector())
                    except AssertionError:
                        self.fail(f"Code snippet {i} failed with correct Selector instance.")

                    # Incorrect usage
                    with self.assertRaises(AssertionError, msg=f"Code snippet {i} did not raise an AssertionError with incorrect type."):
                        TestClass(None)

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

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the __init__ function
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