import unittest
import json
import os
from typing import Sequence, Callable, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class SequenceMap:
    def __init__(self, fst: Callable[[Any], Any], snd: Callable[[Any], Any], f: Callable[[Any, Any], Any]):
        self.fst = fst
        self.snd = snd
        self.f = f

class TestEvalSequenceMap(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[141]  # Get the 142nd JSON element
        if not cls.code_snippet:
            raise ValueError("Expected code snippet at index 141")

    def test_eval_sequence_map(self):
        """Dynamically test the eval_sequence_map function."""
        code = self.code_snippet  # Get the code snippet
        results = []

        exec_globals = {
            'SequenceMap': SequenceMap,
            'Sequence': Sequence,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if eval_sequence_map is present
            if 'eval_sequence_map' not in exec_locals:
                print("eval_sequence_map function not found.")
                results.append({
                    "function_name": "eval_sequence_map",
                    "code": code,
                    "result": "failed"
                })
                self.fail("eval_sequence_map function not found.")
            
            eval_sequence_map = exec_locals['eval_sequence_map']  # Extract the function

            # Sample test case for eval_sequence_map
            def multiply_by_two(x):
                return x * 2

            def add_three(x):
                return x + 3

            def combine(x, y):
                return x + y

            sop = SequenceMap(multiply_by_two, add_three, combine)
            xs = [1, 2, 3]
            
            result = eval_sequence_map(None, sop, xs)
            expected = [6, 9, 12]  # (1*2 + 1+3), (2*2 + 2+3), (3*2 + 3+3)
            self.assertEqual(result, expected, "eval_sequence_map did not return the expected result.")

            print("Test passed.")
            results.append({
                "function_name": "eval_sequence_map",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Test failed with error: {e}")
            results.append({
                "function_name": "eval_sequence_map",
                "code": code,
                "result": "failed"
            })
            self.fail(f"Exception thrown during testing: {e}")

        # Append the new results to the JSONL file
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for eval_sequence_map
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "eval_sequence_map"
        ]

        # Add the new results
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()