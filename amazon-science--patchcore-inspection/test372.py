import unittest
import json
import sys
import os
from unittest.mock import MagicMock

TEST_RESULT_JSONL = "test_result.jsonl"

class TestIndexToCPUFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[371]  # Get the 372nd JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 372th JSON array")

    def test_index_to_cpu(self):
        """Dynamically test all code snippets in the JSON."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to be written into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Static check: Ensure _index_to_cpu is defined
                if "_index_to_cpu" not in code:
                    print(f"Code snippet {i}: FAILED, '_index_to_cpu' not found in code.\n")
                    failed_count += 1
                    # Write failed record
                    results.append({
                        "function_name": "_index_to_cpu",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and logic test
                exec_globals = {
                    'faiss': MagicMock(),
                }
                exec_locals = {}

                try:
                    # Dynamically execute code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if _index_to_cpu is correctly defined
                    if '_index_to_cpu' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, '_index_to_cpu' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "_index_to_cpu",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create a mock instance to test _index_to_cpu
                    instance = MagicMock()
                    
                    # Test when on_gpu is True
                    instance.on_gpu = True
                    index = MagicMock()
                    exec_locals['_index_to_cpu'](instance, index)
                    exec_globals['faiss'].index_gpu_to_cpu.assert_called_once_with(index)

                    # Test when on_gpu is False
                    exec_globals['faiss'].index_gpu_to_cpu.reset_mock()  # Reset mock call history
                    instance.on_gpu = False
                    result = exec_locals['_index_to_cpu'](instance, index)
                    exec_globals['faiss'].index_gpu_to_cpu.assert_not_called()
                    self.assertEqual(result, index, f"Code snippet {i}: expected index to be returned as-is.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "_index_to_cpu",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_index_to_cpu",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it doesn't exist, ignore)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records with function_name == "_index_to_cpu"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_index_to_cpu"
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