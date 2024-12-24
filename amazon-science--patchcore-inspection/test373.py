import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCreateIndexFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[372]  # Get the 373rd JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the given JSON element")

    def test_code_snippets(self):
        """Dynamically test all code snippets for _create_index function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Static Checks -------------------
                if "_create_index" not in code:
                    print(f"Code snippet {i}: FAILED, '_create_index' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_create_index",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "def _create_index" not in code:
                    print(f"Code snippet {i}: FAILED, function '_create_index' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_create_index",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic Execution and Logic Tests -------------------
                exec_globals = {
                    'faiss': sys.modules.get('faiss', Any),  # Mock faiss if not available
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Verify _create_index is present
                    if '_create_index' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_create_index' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_create_index",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create a mock object with attribute on_gpu to test _create_index
                    class MockClass:
                        def __init__(self, on_gpu):
                            self.on_gpu = on_gpu

                    instance = MockClass(on_gpu=True)
                    method = exec_locals['_create_index'].__get__(instance, MockClass)

                    # Test the returned indexes
                    gpu_index = method(128)
                    instance.on_gpu = False
                    cpu_index = method(128)

                    # Check assumptions based on the mock environment
                    self.assertTrue(hasattr(gpu_index, 'faiss_index'), f"Code snippet {i} did not return a GPU index for on_gpu=True.")
                    self.assertTrue(hasattr(cpu_index, 'faiss_index'), f"Code snippet {i} did not return a CPU index for on_gpu=False.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_create_index",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_create_index",
                        "code": code,
                        "result": "failed"
                    })

        # Test Summary
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

        # Remove old records for _create_index
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_create_index"
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