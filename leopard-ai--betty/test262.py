import unittest
import json
import sys
import os
from typing import Any  # Make sure Any is available in the injected environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPatchDataLoader(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[261]  # Get the 262nd JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 262nd JSON array")

    def test_patch_data_loader(self):
        """Dynamically test patch_data_loader function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # List to collect test results for JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check if the code contains the definition of patch_data_loader
                if "def patch_data_loader" not in code:
                    print(f"Code snippet {i}: FAILED, 'patch_data_loader' function not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "patch_data_loader",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Prepare execution environment
                exec_globals = {
                    'sys': sys,
                    'Any': Any,
                    'get_distributed_data_loader': lambda loader, world_size, rank: f"distributed_{loader}_{world_size}_{rank}"
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if patch_data_loader is present in exec_locals
                    if 'patch_data_loader' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'patch_data_loader' not found after exec.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "patch_data_loader",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create a dummy class with attributes needed for testing
                    class DummyClass:
                        def __init__(self, strategy, world_size=None, rank=None, accelerator=None):
                            self._strategy = strategy
                            self._world_size = world_size
                            self._rank = rank
                            self.accelerator = accelerator

                    # Test distributed strategy
                    instance = DummyClass(strategy="distributed", world_size=4, rank=1)
                    patched_loader = exec_locals['patch_data_loader'](instance, 'my_loader')
                    self.assertEqual(patched_loader, "distributed_my_loader_4_1")

                    # Test accelerate strategy
                    instance = DummyClass(strategy="accelerate", accelerator=type('Accelerator', (object,), {'prepare': lambda self, x: f"accelerated_{x}"})())
                    patched_loader = exec_locals['patch_data_loader'](instance, 'my_loader')
                    self.assertEqual(patched_loader, "accelerated_my_loader")

                    # Test non-strategy (default)
                    instance = DummyClass(strategy="normal")
                    patched_loader = exec_locals['patch_data_loader'](instance, 'my_loader')
                    self.assertEqual(patched_loader, 'my_loader')

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "patch_data_loader",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "patch_data_loader",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "patch_data_loader"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()