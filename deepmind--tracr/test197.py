import unittest
import json
import os
from typing import Callable
from unittest.mock import MagicMock

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMapCategoricalToNumericalMLPResult(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the JSON file and retrieve required code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[196]  # Retrieve the 197th JSON element (index 196)
        assert len(cls.code_snippet) > 0, "Expected at least one code snippet in the JSON array"

    def test_map_categorical_to_numerical_mlp(self):
        """Test map_categorical_to_numerical_mlp function with various scenarios."""
        results = []

        for i, code in enumerate(self.code_snippet):
            with self.subTest(code_index=i):
                exec_globals = {}
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if the function is defined
                    if 'map_categorical_to_numerical_mlp' not in exec_locals:
                        raise Exception(f"'map_categorical_to_numerical_mlp' not defined in code snippet {i}.")

                    map_categorical_to_numerical_mlp = exec_locals['map_categorical_to_numerical_mlp']

                    # Mock necessary classes
                    MockVectorSpaceWithBasis = MagicMock()
                    MockValue = MagicMock()
                    MockLinear = MagicMock()
                    MockProject = MagicMock()
                    MockMLP = MagicMock()

                    # Setup mock behavior
                    mock_input_space = MockVectorSpaceWithBasis()
                    mock_output_space = MockVectorSpaceWithBasis()
                    mock_operation = MagicMock(return_value=2.0)  # Example operation

                    mock_output_space.basis = [MockValue(value=1.0)]  # Mock basis

                    # Inject mocks
                    exec_globals.update({
                        'bases': {'VectorSpaceWithBasis': MockVectorSpaceWithBasis, 'ensure_dims': MagicMock()},
                        'vectorspace_fns': {'Linear': MockLinear, 'project': MockProject},
                        'transformers': {'MLP': MockMLP},
                        'Callable': Callable
                    })

                    # Call the function
                    mlp_result = map_categorical_to_numerical_mlp(
                        input_space=mock_input_space,
                        output_space=mock_output_space,
                        operation=mock_operation
                    )

                    # Assert MLP is called and check the behavior
                    MockMLP.assert_called_once()
                    MockLinear.from_action.assert_called_once_with(
                        mock_input_space, mock_output_space, unittest.mock.ANY
                    )
                    MockProject.assert_called_once_with(mock_output_space, mock_output_space)

                    results.append({
                        "function_name": "map_categorical_to_numerical_mlp",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}")
                    results.append({
                        "function_name": "map_categorical_to_numerical_mlp",
                        "code": code,
                        "result": "failed"
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

        # Remove old records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "map_categorical_to_numerical_mlp"
        ]

        # Append new results
        existing_records.extend(results)

        # Re-write results to test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()