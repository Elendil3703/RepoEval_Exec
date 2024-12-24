import unittest
import json
import sys
import re
import os
from typing import Any  # Ensure Any is available

TEST_RESULT_JSONL = "test_result.jsonl"

_TRAINERS = {}

def register_trainer(name):
    """Decorator used to register a trainer
    Args:
        name: Name of the trainer
    """
    def register_class(cls):
        _TRAINERS[name] = cls
        return cls
    return register_class

class TestTrainerUnavailable(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[9]  # Get the 10th JSON element (index 9)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 10th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Additional logic checks -------------------
                # 1) Static check: Determine if the snippet defines '_trainer_unavailble' function
                if "_trainer_unavailble" not in code:
                    print(f"Code snippet {i}: FAILED, '_trainer_unavailble' not found in code.\n")
                    failed_count += 1
                    # Write failed record
                    results.append({
                        "function_name": "_trainer_unavailble",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+_trainer_unavailble\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for '_trainer_unavailble'.\n")
                    failed_count += 1
                    # Write failed record
                    results.append({
                        "function_name": "_trainer_unavailble",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution and logic test -------------------
                exec_globals = {
                    'sys': sys,
                    '_TRAINERS': _TRAINERS,
                    'register_trainer': register_trainer,
                    'Any': Any,  # Inject Any
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if '_trainer_unavailble' exists
                    if '_trainer_unavailble' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_trainer_unavailble' not found after exec.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_trainer_unavailble",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Use '_trainer_unavailble' to test logic
                    # We'll attempt to get a trainer and expect an ImportError
                    trainer_name = 'test_trainer'
                    _trainer_unavailble = exec_locals['_trainer_unavailble']
                    # Call the function to register the unavailable trainer
                    _trainer_unavailble(trainer_name)

                    # Now, attempt to instantiate the trainer and expect ImportError
                    try:
                        trainer_class = _TRAINERS[trainer_name]
                        # Attempt to instantiate or call the trainer class/function
                        trainer_class()
                        print(f"Code snippet {i}: FAILED, expected ImportError but none was raised.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_trainer_unavailble",
                            "code": code,
                            "result": "failed"
                        })
                        continue
                    except ImportError as e:
                        print(f"Code snippet {i}: PASSED, ImportError raised as expected: {e}\n")
                        passed_count += 1
                        results.append({
                            "function_name": "_trainer_unavailble",
                            "code": code,
                            "result": "passed"
                        })
                    except Exception as e:
                        print(f"Code snippet {i}: FAILED, unexpected exception raised: {e}\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_trainer_unavailble",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_trainer_unavailble",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results into test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if not exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "_trainer_unavailble"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_trainer_unavailble"
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