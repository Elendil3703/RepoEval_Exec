import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthPostInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        if len(data) < 226:
            raise ValueError("Expected at least 226 code snippets in the JSON data.")
        cls.code_snippets = data[225]  # Get the 226th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the selected JSON array.")

    def test_post_init_function(self):
        """Dynamically test all code snippets in the JSON for __post_init__ function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check if "__post_init__" is in the code snippet as a static check
                if "__post_init__" not in code:
                    print(f"Code snippet {i}: FAILED, '__post_init__' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__post_init__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and testing
                class DummyBlock:
                    def __init__(self, residual_space):
                        self.residual_space = residual_space

                exec_globals = {
                    'DummyBlock': DummyBlock,
                    'Any': Any
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Create a class with __post_init__ method
                    class TestClass:
                        def __init__(self, sub_blocks):
                            self.sub_blocks = sub_blocks
                            # Call the injected __post_init__ method
                            exec_locals['__post_init__'](self)

                    # Test case 1: All blocks have the same residual_space
                    try:
                        test_obj = TestClass(sub_blocks=[DummyBlock(5), DummyBlock(5), DummyBlock(5)])
                        print(f"Code snippet {i}: PASSED for matching spaces.\n")
                        passed_count += 1
                        results.append({
                            "function_name": "__post_init__",
                            "code": code,
                            "result": "passed"
                        })
                    except AssertionError:
                        print(f"Code snippet {i}: FAILED, AssertionError raised with matching spaces.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__post_init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test case 2: Blocks with different residual_spaces should raise an AssertionError
                    try:
                        test_obj = TestClass(sub_blocks=[DummyBlock(5), DummyBlock(6), DummyBlock(5)])
                        print(f"Code snippet {i}: FAILED, no AssertionError for non-matching spaces.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__post_init__",
                            "code": code,
                            "result": "failed"
                        })
                    except AssertionError:
                        print(f"Code snippet {i}: PASSED for non-matching spaces (AssertionError expected).\n")
                        passed_count += 1
                        results.append({
                            "function_name": "__post_init__",
                            "code": code,
                            "result": "passed"
                        })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__post_init__",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
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
            if rec.get("function_name") != "__post_init__"
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