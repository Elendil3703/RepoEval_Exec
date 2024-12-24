import unittest
import json
import os
from typing import List

TEST_RESULT_JSONL = "test_result.jsonl"

class VectorInBasis:
    def __init__(self, basis_directions, magnitudes):
        self.basis_directions = basis_directions
        self.magnitudes = magnitudes

    # 用于测试比较向量
    def __eq__(self, other):
        return (self.basis_directions == other.basis_directions and
                all(abs(a - b) < 1e-9 for a, b in zip(self.magnitudes, other.magnitudes)))


class TestCallMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[202]  # Get the 203rd JSON element (index 202)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")
    
    def test_call_method(self):
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                exec_globals = {
                    'VectorInBasis': VectorInBasis,
                }
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    if '__call__' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, function '__call__' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "__call__",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Prepare a mock class to test against
                    input_space = [VectorInBasis(['i', 'j'], [1, 2])]
                    output_space = VectorInBasis(['x', 'y'], [0, 0])
                    matrix = [[1, 0], [0, 1]]  # Identity matrix

                    class MockClass:
                        input_space = input_space
                        output_space = output_space
                        matrix = matrix

                        def __call__(self, x: VectorInBasis) -> VectorInBasis:
                            if x not in self.input_space:
                                raise TypeError(f"x={x} not in self.input_space={self.input_space}.")
                            return VectorInBasis(
                                basis_directions=sorted(self.output_space.basis_directions),
                                magnitudes=[m1 * m2 for m1, m2 in zip(x.magnitudes, self.matrix[0])]
                            )

                    # Test the call function
                    mock_instance = exec_locals['__call__'].__get__(MockClass(), MockClass)
                    test_vector = VectorInBasis(['i', 'j'], [1, 2])

                    # Correct transformation
                    transformed_vector = mock_instance(test_vector)
                    expected_vector = VectorInBasis(['x', 'y'], [1, 2])  # Expected result with identity matrix
                    self.assertEqual(transformed_vector, expected_vector)

                    # Incorrect transformation
                    with self.assertRaises(TypeError):
                        mock_instance(VectorInBasis(['a', 'b'], [0, 0]))  # Not in input space

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__call__",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__call__",
                        "code": code,
                        "result": "failed"
                    })

        # Print test summary
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

        # Remove old records for "__call__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__call__"
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