import unittest
import json
import os
from typing import Sequence

TEST_RESULT_JSONL = "test_result.jsonl"

class Map:
    def __init__(self, f, inner):
        self.f = f
        self.inner = inner

class Value:
    pass

class EvalMapTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[142]  # Get the 143rd JSON element (index 142)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_eval_map(self):
        """Dynamically test the eval_map function from the JSON code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Injected dependencies and execution environment setup
                exec_globals = {
                    'Map': Map,
                    'Value': Value,
                    'Sequence': Sequence
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if eval_map is defined
                    if 'eval_map' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'eval_map' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "eval_map",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    eval_map = exec_locals['eval_map']

                    # Define a test case for eval_map
                    class TestEvalMap:
                        def evaluate(self, inner, xs):
                            return xs  # Just return the list for simplicity

                        def eval_map(self, sop: Map, xs: Sequence[Value]) -> Sequence[Value]:
                            return [
                                sop.f(x) if x is not None else None
                                for x in self.evaluate(sop.inner, xs)
                            ]

                    # Create instances and a function to map
                    instance = TestEvalMap()
                    def square(x):
                        return x * x

                    sop = Map(square, None)

                    # Input sequence
                    input_seq = [1, 2, None, 3]

                    # Expected output
                    expected_output = [1, 4, None, 9]

                    # Call eval_map
                    output = eval_map(instance, sop, input_seq)
                    
                    # Test that output matches expectation
                    self.assertEqual(output, expected_output, f"Code snippet {i} failed: output {output} != expected {expected_output}")
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "eval_map",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "eval_map",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the test results to test_result.jsonl
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
            if rec.get("function_name") != "eval_map"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()