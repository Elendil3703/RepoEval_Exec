import unittest
import json
import os
from typing import List

TEST_RESULT_JSONL = "test_result.jsonl"

class VectorSpaceWithBasis:
    def __init__(self, basis: List[str]):
        self.basis = basis

    def __eq__(self, other):
        if not isinstance(other, VectorSpaceWithBasis):
            return False
        return sorted(self.basis) == sorted(other.basis)

class TestJoinVectorSpaces(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[221]  # Get the 222nd JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 222nd JSON array")

    def test_join_vector_spaces(self):
        """Dynamically test all code snippets for join_vector_spaces."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "def join_vector_spaces" not in code:
                    print(f"Code snippet {i}: FAILED, function 'join_vector_spaces' not found.\n")
                    failed_count += 1
                    # Record the failure
                    results.append({
                        "function_name": "join_vector_spaces",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'VectorSpaceWithBasis': VectorSpaceWithBasis,
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure join_vector_spaces function exists
                    if 'join_vector_spaces' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'join_vector_spaces' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "join_vector_spaces",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    join_vector_spaces = exec_locals['join_vector_spaces']

                    # Test cases for join_vector_spaces
                    vs1 = VectorSpaceWithBasis(['a', 'b', 'c'])
                    vs2 = VectorSpaceWithBasis(['b', 'c', 'd'])
                    expected = VectorSpaceWithBasis(['a', 'b', 'c', 'd'])

                    result = join_vector_spaces(vs1, vs2)
                    self.assertEqual(result, expected)

                    vs3 = VectorSpaceWithBasis(['e', 'f'])
                    vs4 = VectorSpaceWithBasis(['f', 'g'])
                    expected2 = VectorSpaceWithBasis(['e', 'f', 'g'])

                    result2 = join_vector_spaces(vs3, vs4)
                    self.assertEqual(result2, expected2)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "join_vector_spaces",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "join_vector_spaces",
                        "code": code,
                        "result": "failed"
                    })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for join_vector_spaces
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "join_vector_spaces"
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