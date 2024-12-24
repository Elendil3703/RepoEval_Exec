import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class MockBasisDirection:
    """Mock class for BasisDirection to simulate behaviour."""
    def __init__(self, direction_type, value=None):
        self.direction_type = direction_type
        self.value = value

class MockInputSpace:
    """Mock class for InputSpace to simulate vector operations."""
    def vector_from_basis_direction(self, basis_direction):
        if basis_direction.direction_type == "_BOS_DIRECTION":
            return [1]
        elif basis_direction.direction_type == "_ONE_DIRECTION":
            return [1]
        elif basis_direction.direction_type == "indices":
            return [basis_direction.value]
        elif basis_direction.direction_type == "tokens":
            return [basis_direction.value + 10]
        return [0]

class MockVectorInBasis:
    """Mock class for VectorInBasis to simulate stack operation."""
    @staticmethod
    def stack(embedded_vectors):
        # Simplified stacking for demonstration purposes
        return sum(embedded_vectors, [])

# The actual _embed_input function call and test setup
class TestEmbedInputFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[169]  # Get the 170th JSON element

        if "_embed_input" not in cls.code_snippet:
            raise ValueError("Expected function '_embed_input' not found in the code snippet")

    def test_embed_input_function(self):
        """Test the _embed_input function with various test cases."""
        # Define mock objects for bases
        bases = {
            'BasisDirection': MockBasisDirection,
            'VectorInBasis': MockVectorInBasis
        }

        # Extract and execute the provided code snippet in a controlled environment
        exec_globals = {
            'bases': bases,
            '_BOS_DIRECTION': "_BOS_DIRECTION",
            '_ONE_DIRECTION': "_ONE_DIRECTION",
        }
        exec_locals = {}
        exec(self.code_snippet, exec_globals, exec_locals)
        
        # Verify if _embed_input is available to use
        if "_embed_input" not in exec_locals:
            self.fail("Failed to define '_embed_input' in the exec environment.")

        _embed_input = exec_locals["_embed_input"]

        # Testing with predefined inputs and expected outputs
        input_space = MockInputSpace()
        test_cases = [
            ([0, 1, 2], [[2], [11], [1]]),
            ([], [1, 1]),
            ([3, 4, 5], [[4], [13], [1]])
        ]

        results = []
        passed_count = 0
        failed_count = 0

        for i, (input_seq, expected) in enumerate(test_cases):
            with self.subTest(input_seq=input_seq):
                try:
                    output = _embed_input(input_seq, input_space)
                    expected_output = MockVectorInBasis.stack(expected)
                    self.assertEqual(output, expected_output)
                    print(f"Test {i} PASSED: input {input_seq}, expected {expected_output}, got {output}.")
                    results.append({
                        "function_name": "_embed_input",
                        "code": self.code_snippet,
                        "result": "passed"
                    })
                    passed_count += 1
                except AssertionError as e:
                    print(f"Test {i} FAILED: input {input_seq}, expected {expected_output}, got {output}.")
                    results.append({
                        "function_name": "_embed_input",
                        "code": self.code_snippet,
                        "result": "failed"
                    })
                    failed_count += 1
                except Exception as e:
                    print(f"Test {i} ERROR: {str(e)}")
                    results.append({
                        "function_name": "_embed_input",
                        "code": self.code_snippet,
                        "result": "failed"
                    })
                    failed_count += 1

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_cases)}\n")

        # Writing test results to JSONL
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
            if rec.get("function_name") != "_embed_input"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()