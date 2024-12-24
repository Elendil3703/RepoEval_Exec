import unittest
import json
import os
import numpy as np
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class FakeBasis:
    """A mock class to simulate bases.VectorInBasis behavior for testing purposes."""
    def __init__(self, basis=None, magnitudes=None):
        self.basis = basis or []
        self.magnitudes = magnitudes or np.array([])

    def __eq__(self, other):
        return (self.basis == other.basis) and np.allclose(self.magnitudes, other.magnitudes)

    def project(self, space):
        # Mock method to simulate whatever `project` is supposed to do
        return self

class TestApplyFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[223]  # Get the 224th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 224th JSON array")

    def test_apply_function(self):
        """Dynamically test the apply function."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check if the necessary components are present
                if "def apply" not in code:
                    print(f"Code snippet {i}: FAILED, 'apply' function not found.\n")
                    failed_count += 1
                    results.append({"function_name": "apply", "code": code, "result": "failed"})
                    continue

                exec_globals = {
                    'np': np,
                    'bases': FakeBasis,
                    '_np_softmax': lambda x: np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True),  # Mock softmax
                }
                exec_locals = {}

                try:
                    # Dynamic execution of code snippet
                    exec(code, exec_globals, exec_locals)
                    apply_func = exec_locals.get('apply')

                    if not apply_func:
                        print(f"Code snippet {i}: FAILED, 'apply' function not found in exec_locals.\n")
                        failed_count += 1
                        results.append({"function_name": "apply", "code": code, "result": "failed"})
                        continue

                    # Mock class with apply method
                    class MockClass:
                        def __init__(self):
                            self.residual_space = FakeBasis(["dimension_1", "dimension_2"])
                            self.w_qk = self
                            self.w_ov_residual = lambda x: x
                            self.causal = False
                            self.matrix = np.array([[1, 0], [0, 1]])

                        left_space = right_space = residual_space

                        def apply(self, x):
                            return apply_func(self, x)

                    # Create an instance of the mock object
                    obj = MockClass()

                    # Fake input
                    input_vector = FakeBasis(["dimension_1", "dimension_2"], np.array([[1, 2], [3, 4]]))

                    # Expected output must be calculated based on the understanding of apply
                    expected_output = FakeBasis(["dimension_1", "dimension_2"], np.array([[1, 2], [3, 4]]))

                    # Test the apply function
                    result = obj.apply(input_vector)
                    self.assertEqual(result, expected_output, f"Code snippet {i} did not produce expected output.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({"function_name": "apply", "code": code, "result": "passed"})

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({"function_name": "apply", "code": code, "result": "failed"})

        # Statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        existing_records = [rec for rec in existing_records if rec.get("function_name") != "apply"]
        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()