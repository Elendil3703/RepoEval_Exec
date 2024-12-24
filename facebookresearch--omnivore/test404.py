import unittest
import json
import os
from typing import Mapping

TEST_RESULT_JSONL = "test_result.jsonl"

class OmnivisionOptimAMPConf:
    # Dummy class for the purpose of testing
    def __init__(self, **kwargs):
        self.config = kwargs

class TestOmnivisionOptimAMPConfPostInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[403]  # Get the 404th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 404th JSON array")

    def test_post_init(self):
        """Test the __post_init__ method logic."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                exec_globals = {
                    'OmnivisionOptimAMPConf': OmnivisionOptimAMPConf,
                    'Mapping': Mapping,  # Inject necessary types
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if __post_init__ is defined
                    if '__post_init__' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '__post_init__' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__post_init__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create a dummy class to test the __post_init__ logic
                    class DummyClass:
                        def __init__(self, amp=None):
                            self.amp = amp

                        exec_locals['__post_init__'](self)

                    # Test cases for __post_init__
                    # Test case 1: amp is an instance of OmnivisionOptimAMPConf
                    instance1 = DummyClass(amp=OmnivisionOptimAMPConf())
                    self.assertIsInstance(
                        instance1.amp, OmnivisionOptimAMPConf,
                        f"Code snippet {i} failed: Expected amp to be OmnivisionOptimAMPConf instance"
                    )

                    # Test case 2: amp is a dictionary and should be converted
                    instance2 = DummyClass(amp={'key': 'value'})
                    self.assertIsInstance(
                        instance2.amp, OmnivisionOptimAMPConf,
                        f"Code snippet {i} failed: Expected amp to be converted to OmnivisionOptimAMPConf instance"
                    )

                    # Test case 3: amp is None and should be converted
                    instance3 = DummyClass(amp=None)
                    self.assertIsInstance(
                        instance3.amp, OmnivisionOptimAMPConf,
                        f"Code snippet {i} failed: Expected None amp to be converted to OmnivisionOptimAMPConf instance"
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
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

        # Test statistics
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

        # Remove old records for __post_init__
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__post_init__"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()