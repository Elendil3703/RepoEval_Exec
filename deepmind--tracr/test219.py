import unittest
import json
import os
import numpy as np
from typing import Union, Sequence

# Constants
TEST_RESULT_JSONL = "test_result.jsonl"

class VectorSpaceWithBasis:
    def __init__(self, basis):
        self.basis = basis

class VectorInBasis:
    def __init__(self, basis, magnitudes):
        self.basis = basis
        self.magnitudes = magnitudes

class TestProjectFunction(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[218]  # Get the 219th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 219th JSON array")

    def test_project_function(self):
        """Test the 'project' function with various scenarios."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to be written to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # Mock data for testing
                vector = VectorInBasis(
                    basis=['x', 'y'],
                    magnitudes=np.array([1.0, 2.0])
                )
                
                vector.project = lambda basis: VectorInBasis(
                    list(basis),
                    np.array([
                        vector.magnitudes[vector.basis.index(direction)]
                        if direction in vector.basis else 0.0
                        for direction in basis
                    ])
                )
                
                # Static check: Function 'project' should exist
                if not hasattr(vector, 'project'):
                    print(f"Code snippet {i}: FAILED, 'project' method not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "project",
                        "code": code,
                        "result": "failed"
                    })
                    continue
                
                try:
                    # Test scenarios
                    basis = VectorSpaceWithBasis(['x', 'z'])
                    result = vector.project(basis)

                    # Assertions
                    self.assertEqual(result.basis, ['x', 'z'], "Basis mismatch.")
                    np.testing.assert_array_equal(result.magnitudes, np.array([1.0, 0.0]), "Magnitudes mismatch.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "project",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "project",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")

        # === Write results to test_result.jsonl ===
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the function 'project'
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "project"]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()