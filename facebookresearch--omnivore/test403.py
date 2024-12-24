import unittest
import json
import sys
import os
from dataclasses import dataclass, make_dataclass
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class Sample:
    pass

class Batch:
    pass

@dataclass
class DummyClass(Sample):
    pass

def create_batch_sample_cls(cls):
    """Dynamically creates a dataclass which is a `Batch` and a `Sample`.

    This function also registers the class in globals() to make the class picklable.
    """
    cls_name = f"{Batch.__name__}{cls.__name__}"
    batch_sample_cls = make_dataclass(cls_name, fields=(), bases=(cls, Batch))
    batch_sample_cls.__module__ = __name__
    globals()[cls_name] = batch_sample_cls
    return batch_sample_cls

class TestCreateBatchSampleCls(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[402]  # Get the 403rd JSON element (index 402)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the selected JSON array")

    def test_create_batch_sample_cls(self):
        """Dynamically test the code snippet's create_batch_sample_cls implementation."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect result data to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Execute the code and test create_batch_sample_cls
                exec_globals = {
                    'sys': sys,
                    'Any': Any,
                    'dataclass': dataclass,
                    'make_dataclass': make_dataclass,
                    'Sample': Sample,
                    'Batch': Batch,
                    'DummyClass': DummyClass
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if create_batch_sample_cls is present
                    if 'create_batch_sample_cls' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'create_batch_sample_cls' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "create_batch_sample_cls",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the BatchSample creation logic
                    created_cls = exec_locals['create_batch_sample_cls'](DummyClass)

                    # Assert the class name
                    expected_cls_name = f"BatchDummyClass"
                    self.assertEqual(
                        created_cls.__name__,
                        expected_cls_name,
                        f"Code snippet {i} did not create class with name '{expected_cls_name}'."
                    )

                    # Assert the class is picklable (i.e. exists in globals)
                    self.assertIn(
                        expected_cls_name,
                        globals(),
                        f"Code snippet {i} did not register {expected_cls_name} in globals."
                    )

                    # Assert correct base classes
                    self.assertTrue(
                        issubclass(created_cls, (DummyClass, Batch)),
                        f"Code snippet {i} did not create a class with correct base classes."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "create_batch_sample_cls",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "create_batch_sample_cls",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
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

        # Remove old records for create_batch_sample_cls
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "create_batch_sample_cls"
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