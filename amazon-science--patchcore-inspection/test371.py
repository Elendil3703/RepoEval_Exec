import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestIndexToGpu(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[370]  # Get the 371st JSON element
        if not cls.code_snippet:
            raise ValueError("Expected at least one code snippet in the JSON element")

    def test_index_to_gpu(self):
        """Dynamically test the _index_to_gpu function in the code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write into the JSONL

        code = self.code_snippet
        with self.subTest():
            print("Running test for _index_to_gpu function...")
            
            if "_index_to_gpu" not in code:
                print("Code snippet: FAILED, '_index_to_gpu' not found in code.\n")
                failed_count += 1
                results.append({
                    "function_name": "_index_to_gpu",
                    "code": code,
                    "result": "failed"
                })
                return

            exec_globals = {
                'faiss': __import__('faiss', fromlist=['']),
                'sys': sys,
                'Any': Any,  # Include Any injection
            }
            exec_locals = {}

            try:
                # Execute the code snippet dynamically
                exec(code, exec_globals, exec_locals)

                # Check if _index_to_gpu function exists
                if '_index_to_gpu' not in exec_locals:
                    print("Code snippet: FAILED, '_index_to_gpu' not found in exec_locals.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_index_to_gpu",
                        "code": code,
                        "result": "failed"
                    })
                    return
                
                # Create a mock class to simulate the environment
                class MockOptions:
                    def __call__(self):
                        return None
                
                class MockIndex:
                    pass

                class MockClass:
                    on_gpu = True
                    _gpu_cloner_options = MockOptions()
                
                instance = MockClass()
                index = MockIndex()

                result = exec_locals['_index_to_gpu'](instance, index)
                self.assertIsInstance(result, exec_globals['faiss'].Index)

                print("Code snippet: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "_index_to_gpu",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "_index_to_gpu",
                    "code": code,
                    "result": "failed"
                })

        print(f"Test Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Writing test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function _index_to_gpu
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_index_to_gpu"
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