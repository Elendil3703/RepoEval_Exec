import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[206]
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet at index 206 in the JSON array")

    def test_project_function(self):
        """Test the 'project' function to ensure it behaves correctly."""
        passed_count = 0
        failed_count = 0
        results = []

        # Injected dependencies for the execution environment
        exec_globals = {
            'VectorSpaceWithBasis': mock_VectorSpaceWithBasis,
            'Linear': mock_Linear,
            'bases': mock_bases,
        }
        exec_locals = {}

        code = self.code_snippet

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Assume 'project' function is now in exec_locals
            if 'project' not in exec_locals:
                print("Error: 'project' function not found in code execution.")
                failed_count += 1
                results.append({
                    "function_name": "project",
                    "code": code,
                    "result": "failed"
                })
            else:
                project = exec_locals['project']

                # Example of testing 'project' with mock objects
                from_space = mock_VectorSpaceWithBasis()
                to_space = mock_VectorSpaceWithBasis()

                projection = project(from_space, to_space)

                # Check some expected properties of the projection
                self.assertIsInstance(projection, mock_Linear)
                self.assertEqual(projection.from_space, from_space)
                self.assertEqual(projection.to_space, to_space)

                # Mock the behavior inside the action function
                direction = mock_bases.BasisDirection()
                to_space.contains.return_value = True
                to_space.vector_from_basis_direction.return_value = mock_VectorInBasis()

                result_vector = projection.action(direction)
                to_space.vector_from_basis_direction.assert_called_with(direction)
                self.assertEqual(result_vector, to_space.vector_from_basis_direction())

                print("Test: PASSED for project function.")
                passed_count += 1
                results.append({
                    "function_name": "project",
                    "code": code,
                    "result": "passed"
                })
        except Exception as e:
            print(f"Test: FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "project",
                "code": code,
                "result": "failed"
            })

        # Final test summary
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # Write results to JSONL file similar to the reference code
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records for function_name "project"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "project"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Mocks for testing purpose, replace with actual implementations
class mock_VectorSpaceWithBasis:
    def __contains__(self, item):
        pass  # Implement vector space logic

    def vector_from_basis_direction(self, direction):
        pass  # Implement vector conversion logic

    def null_vector(self):
        pass  # Implement null vector logic


class mock_Linear:
    def __init__(self):
        self.from_space = None
        self.to_space = None

    @classmethod
    def from_action(cls, from_space, to_space, action):
        instance = cls()
        instance.from_space = from_space
        instance.to_space = to_space
        instance.action = action
        return instance


class mock_bases:
    class BasisDirection:
        pass


class mock_VectorInBasis:
    pass


if __name__ == "__main__":
    unittest.main()