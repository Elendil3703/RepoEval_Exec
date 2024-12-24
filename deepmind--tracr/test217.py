import unittest
import json
import sys
import re
import os
import numpy as np
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestVectorInBasisEquality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[216]  # Get the 217th JSON element (index 216)
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet in the 217th JSON element")

    def test_vector_in_basis_equality(self):
        """Test the __eq__ method of VectorInBasis."""
        results = []
        code = self.code_snippet

        # Check if the __eq__ method is properly defined in the snippet
        if "def __eq__" not in code:
            print("Code snippet: FAILED, function '__eq__' not found.\n")
            results.append({
                "function_name": "__eq__",
                "code": code,
                "result": "failed"
            })
            self.fail("Function '__eq__' not found in code snippet")

        # Define a basic VectorInBasis class with the provided __eq__ method
        class VectorInBasis:
            def __init__(self, basis_directions, magnitudes):
                self.basis_directions = basis_directions
                self.magnitudes = magnitudes

            exec_globals = {
                'np': np,
            }
            exec_locals = {}

            try:
                exec(code, exec_globals, exec_locals)
                self.__eq__ = exec_locals['__eq__']
            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                results.append({
                    "function_name": "__eq__",
                    "code": code,
                    "result": "failed"
                })
                self.fail(f"Execution of code snippet raised an exception: {e}")

        # Create some VectorInBasis instances for testing
        v1 = VectorInBasis('dir1', np.array([1, 2, 3]))
        v2 = VectorInBasis('dir1', np.array([1, 2, 3]))
        v3 = VectorInBasis('dir2', np.array([1, 2, 3]))
        v4 = VectorInBasis('dir1', np.array([1, 2]))  # Different shape

        # Test equality for equal vectors
        self.assertTrue(v1 == v2, "Vectors v1 and v2 should be equal")
        
        # Test inequality for different basis
        self.assertFalse(v1 == v3, "Vectors v1 and v3 should not be equal due to different basis_directions")

        # Test inequality for different magnitudes shape
        self.assertFalse(v1 == v4, "Vectors v1 and v4 should not be equal due to different magnitudes shape")

        print("Code snippet: PASSED all assertions.\n")
        results.append({
            "function_name": "__eq__",
            "code": code,
            "result": "passed"
        })
        
        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for __eq__
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__eq__"
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