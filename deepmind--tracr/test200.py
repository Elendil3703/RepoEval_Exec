import unittest
import json
import os
import numpy as np

TEST_RESULT_JSONL = "test_result.jsonl"

class TestLogicalAndFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[199]  # Get the 200th JSON element (index 199)
        if not cls.code_snippet:
            raise ValueError("Expected the code snippet to be non-empty")

    def test_logical_and(self):
        """Test logical_and function from the code snippet."""
        passed_count = 0
        failed_count = 0
        results = []

        # Prepare the execution context with relevant imports and mocks
        exec_globals = {
            'np': np,
            'hidden_space': self.MockHiddenSpace(),
            'input1_space': self.MockInputSpace(),
            'input2_space': self.MockInputSpace(),
            'one_space': self.MockOneSpace(),
            'bases': self.MockBases(),
            'to_hidden': self.mock_to_hidden,
        }
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(self.code_snippet, exec_globals, exec_locals)

            # Check if logical_and function exists
            if 'logical_and' not in exec_locals:
                raise Exception("Function 'logical_and' not found after execution")

            logical_and = exec_locals['logical_and']

            # Test cases for logical_and
            test_cases = [
                (exec_globals['one_space'][0], exec_globals['bases'].VectorInBasis(exec_globals['hidden_space'].basis, -np.ones(3))),
                (exec_globals['input1_space'].basis[0], exec_globals['hidden_space'].null_vector() + exec_globals['hidden_space'].vector_from_basis_direction(None)),
                (exec_globals['input2_space'].basis[0], exec_globals['hidden_space'].null_vector() + exec_globals['hidden_space'].vector_from_basis_direction(None)),
            ]

            for i, (direction, expected_outcome) in enumerate(test_cases):
                with self.subTest(test_index=i):
                    result = logical_and(direction)
                    self.assertTrue(np.allclose(result.components, expected_outcome.components), f"Test case {i} failed.")
                    print(f"Test case {i}: PASSED.")
                    passed_count += 1
                    results.append({
                        "function_name": "logical_and",
                        "test_index": i,
                        "result": "passed"
                    })

        except Exception as e:
            print(f"FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "logical_and",
                "result": "failed",
                "error": str(e)
            })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed.")
        self.assertEqual(passed_count + failed_count, len(test_cases), "Test count mismatch!")

        # Write results to test_result.jsonl
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
            if rec.get("function_name") != "logical_and"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

    class MockHiddenSpace:
        def basis(self):
            return None

        def num_dims(self):
            return 3

        def null_vector(self):
            return self.Vector(np.zeros(3))

        def vector_from_basis_direction(self, _):
            return self.Vector(np.ones(3))

        class Vector:
            def __init__(self, components):
                self.components = components

    class MockInputSpace:
        @property
        def basis(self):
            return [1, 2, 3]

    class MockOneSpace:
        def __contains__(self, item):
            return item == 1

        def __getitem__(self, _):
            return 1

    class MockBases:
        @staticmethod
        def VectorInBasis(_, components):
            return TestLogicalAndFunction.MockHiddenSpace.Vector(components)

    @staticmethod
    def mock_to_hidden(_, __):
        return None

if __name__ == "__main__":
    unittest.main()