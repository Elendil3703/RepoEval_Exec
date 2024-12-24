import unittest
import json
import os
import faiss
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCreateIndexResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and get the 377th element (index 376)
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[376]
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the given JSON array")

    def test_create_index(self):
        """Dynamically test the create index function in the JSON with customized checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- Dynamic Execution and Checks -------------------
                exec_globals = {
                    'faiss': faiss,
                    'Any': Any,  # Inject Any to the context
                }
                exec_locals = {}

                try:
                    # Dynamically executing the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Retrieve the _create_index function
                    if '_create_index' not in exec_locals:
                        raise Exception(f"Code snippet {i}: '_create_index' function not found.")

                    create_index = exec_locals['_create_index']

                    # Test create_index with dimension input
                    dimension = 128
                    index = create_index(None, dimension)

                    # Check if the index is of the correct type
                    self.assertIsInstance(
                        index, 
                        faiss.IndexIVFPQ, 
                        f"Code snippet {i} did not return a faiss.IndexIVFPQ object."
                    )

                    # Check index parameter configurations
                    self.assertEqual(
                        index.d, dimension,
                        f"Code snippet {i} index dimension does not match the input dimension."
                    )
                    self.assertEqual(
                        index.nlist, 512,
                        f"Code snippet {i} index nlist (number of centroids) does not match expected value."
                    )
                    self.assertEqual(
                        index.pq.m, 64,
                        f"Code snippet {i} index has incorrect number of sub-quantizers."
                    )

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

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        # Read existing test_result.jsonl if it exists
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name "_create_index"
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