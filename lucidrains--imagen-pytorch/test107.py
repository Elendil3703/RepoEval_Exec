import unittest
import json
import sys
import os
from typing import Any  # Ensure the injected environment has Any

TEST_RESULT_JSONL = "test_result.jsonl"

class MockDataLoader:
    """Mock class to simulate a dataloader."""
    pass

class TestAddTrainDataloader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[106]  # Get the 107th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 107th JSON array")

    def test_add_train_dataloader(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static Checks -------------------
                if "def add_train_dataloader" not in code:
                    print(f"Code snippet {i}: FAILED, 'add_train_dataloader' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "add_train_dataloader",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execute and Test -------------------
                exec_globals = {
                    'sys': sys,
                    'Any': Any,  # Inject Any
                    'MockDataLoader': MockDataLoader,
                    'exists': lambda x: x is not None  # Mock exists function
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Dynamically add a class that uses add_train_dataloader
                    class TestModel:
                        def __init__(self):
                            self.train_dl = None
                            self.prepared = False

                        add_train_dataloader = exec_locals.get('add_train_dataloader', None)

                    if not TestModel.add_train_dataloader:
                        print(f"Code snippet {i}: FAILED, 'add_train_dataloader' not found after exec.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "add_train_dataloader",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test logic
                    model = TestModel()
                    dl = MockDataLoader()

                    # Check adding dataloader when it does not exist
                    model.add_train_dataloader(dl)
                    self.assertIs(model.train_dl, dl, f"Code snippet {i} did not add train dataloader correctly.")

                    # Check adding dataloader when it is already added
                    with self.assertRaises(AssertionError, msg=f"Code snippet {i} did not raise error for existing train dataloader."):
                        model.add_train_dataloader(dl)

                    # Check adding dataloader when the model is prepared
                    model.train_dl = None
                    model.prepared = True
                    with self.assertRaises(AssertionError, msg=f"Code snippet {i} did not raise error when model is prepared."):
                        model.add_train_dataloader(dl)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "add_train_dataloader",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "add_train_dataloader",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "add_train_dataloader"
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