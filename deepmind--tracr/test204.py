import unittest
import json
import sys
import os
import numpy as np
from typing import Callable, List

TEST_RESULT_JSONL = "test_result.jsonl"

class VectorSpaceWithBasis:
    def __init__(self, num_dims: int, basis: List['BasisDirection']):
        self.num_dims = num_dims
        self.basis = basis

    def __contains__(self, vector):
        # Simplified containment check, usually involves checks against vector space properties
        return isinstance(vector, VectorInBasis) and vector.num_dims == self.num_dims

class BasisDirection:
    # Represents a direction in a vector space basis
    pass

class VectorInBasis:
    def __init__(self, magnitudes: List[float], num_dims: int):
        self.magnitudes = magnitudes
        self.num_dims = num_dims

class Linear:
    def __init__(self, input_space: VectorSpaceWithBasis, output_space: VectorSpaceWithBasis, matrix: np.ndarray):
        self.input_space = input_space
        self.output_space = output_space
        self.matrix = matrix

class TestFromAction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[203]  # Get the 204th JSON element

    def test_from_action(self):
        """Dynamically test the from_action function implementation."""
        passed_count = 0
        failed_count = 0
        results = []

        # Extract the code snippet
        code = self.code_snippet

        # ------------------- Execute and Test Logic -------------------
        exec_globals = {
            'np': np,
            'VectorSpaceWithBasis': VectorSpaceWithBasis,
            'BasisDirection': BasisDirection,
            'VectorInBasis': VectorInBasis,
            'Linear': Linear
        }
        exec_locals = {}

        try:
            # Execute the provided code snippet
            exec(code, exec_globals, exec_locals)

            # Check if from_action method exists
            if 'from_action' not in exec_locals:
                failed_count += 1
                results.append({
                    "function_name": "from_action",
                    "code": code,
                    "result": "failed"
                })
                return  # Exit if the function is not defined

            from_action = exec_locals['from_action']

            # Test 1: Check correct matrix construction
            basis = [BasisDirection() for _ in range(3)]
            input_space = VectorSpaceWithBasis(3, basis)
            output_basis = [VectorInBasis([1, 0, 0], 3),
                            VectorInBasis([0, 1, 0], 3),
                            VectorInBasis([0, 0, 1], 3)]
            output_space = VectorSpaceWithBasis(3, output_basis)

            def identity_action(direction):
                index = input_space.basis.index(direction)
                return output_basis[index]

            linear_transform = from_action(input_space, output_space, identity_action)
            expected_matrix = np.eye(3)
            self.assertTrue(np.array_equal(linear_transform.matrix, expected_matrix))

            passed_count += 1
            results.append({
                "function_name": "from_action",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            failed_count += 1
            results.append({
                "function_name": "from_action",
                "code": code,
                "result": "failed"
            })

        # Test Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
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
            if rec.get("function_name") != "from_action"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()