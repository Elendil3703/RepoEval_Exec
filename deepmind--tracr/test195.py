import unittest
import json
import os
from typing import Callable

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMapCategoricalMLP(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[194]  # Get the 195th JSON element (index 194)
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet at the 195th position in the JSON array")

    def test_map_categorical_mlp(self):
        """Test map_categorical_mlp function from the extracted code snippet."""
        results = []  # To collect results for writing to JSONL

        code = self.__class__.code_snippet  # Extract the current code snippet

        # Set up the initial globals and locals for exec
        exec_globals = {
            'Callable': Callable,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if map_categorical_mlp is defined
            self.assertIn(
                'map_categorical_mlp',
                exec_locals,
                "The function 'map_categorical_mlp' is not defined in the executed code."
            )

            # Dummy classes and function for testing
            class TestBasisDirection:
                pass

            class TestVectorSpaceWithBasis:
                def __contains__(self, direction):
                    return isinstance(direction, TestBasisDirection)

                def vector_from_basis_direction(self, direction):
                    if isinstance(direction, TestBasisDirection):
                        return "vector"

                def null_vector(self):
                    return "null_vector"

            def dummy_operation(direction):
                return direction

            class MLP:
                def __init__(self, *args):
                    pass

            class vectorspace_fns:
                @staticmethod
                def Linear():
                    class LinearClass:
                        @staticmethod
                        def from_action(*args):
                            return "linear_action"
                    return LinearClass()

                @staticmethod
                def project(*args):
                    return "projection"

            class transformers:
                MLP = MLP

            # Attempt to rebuild the function using mock objects
            map_categorical_mlp = exec_locals['map_categorical_mlp']

            input_space = TestVectorSpaceWithBasis()
            output_space = TestVectorSpaceWithBasis()
            operation = dummy_operation

            mlp = map_categorical_mlp(input_space, output_space, operation)

            # Check conditions related to the function's outputs
            self.assertIsInstance(mlp, MLP, "The result is not an instance of MLP.")

            print("Test for 'map_categorical_mlp': PASSED.\n")
            results.append({
                "function_name": "map_categorical_mlp",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Test for 'map_categorical_mlp': FAILED with error: {e}\n")
            results.append({
                "function_name": "map_categorical_mlp",
                "code": code,
                "result": "failed",
                "error": str(e)
            })

        # ====== Write the test result to test_result.jsonl ======
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for map_categorical_mlp
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "map_categorical_mlp"
        ]

        # Append the new result
        existing_records.extend(results)

        # Rewrite the results back to the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()