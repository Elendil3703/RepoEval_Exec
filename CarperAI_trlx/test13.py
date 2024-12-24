import unittest
import json
import os
from enum import Enum
from typing import Any, Type

TEST_RESULT_JSONL = "test_result.jsonl"

class SchedulerName(Enum):
    COSINE_ANNEALING = "cosine_annealing"
    LINEAR = "linear"

class CosineAnnealingLR:
    pass

class LinearLR:
    pass

class TestGetSchedulerClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[12]  # Get the 14th JSON element (0-indexed)
        if not cls.code_snippet:
            raise ValueError("Expected a valid code snippet in the JSON data")

    def test_get_scheduler_class(self):
        """Test the get_scheduler_class function dynamically."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        code = self.code_snippet
        print("Testing the get_scheduler_class function...")

        # ------------------- Dynamic Execution -------------------
        exec_globals = {
            'SchedulerName': SchedulerName,
            'CosineAnnealingLR': CosineAnnealingLR,
            'LinearLR': LinearLR,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if get_scheduler_class is defined
            if 'get_scheduler_class' not in exec_locals:
                print("FAILED: 'get_scheduler_class' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "get_scheduler_class",
                    "code": code,
                    "result": "failed"
                })
                return

            get_scheduler_class = exec_locals['get_scheduler_class']

            # Test 1: Check for COSINE_ANNEALING
            scheduler_class = get_scheduler_class(SchedulerName.COSINE_ANNEALING)
            self.assertEqual(
                scheduler_class,
                CosineAnnealingLR,
                "COSINE_ANNEALING should return CosineAnnealingLR"
            )

            # Test 2: Check for LINEAR
            scheduler_class = get_scheduler_class(SchedulerName.LINEAR)
            self.assertEqual(
                scheduler_class,
                LinearLR,
                "LINEAR should return LinearLR"
            )

            # Test 3: Unsupported Scheduler
            with self.assertRaises(ValueError) as context:
                get_scheduler_class("unsupported_scheduler")

            self.assertIn(
                "is not a supported scheduler",
                str(context.exception),
                "Error message for unsupported scheduler is incorrect"
            )

            print("PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "get_scheduler_class",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "get_scheduler_class",
                "code": code,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= Write Test Results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for get_scheduler_class
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "get_scheduler_class"
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