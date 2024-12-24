import unittest
import json
import os
from typing import Sequence, Any, TypeVar, Callable
import numpy as np

TEST_RESULT_JSONL = "test_result.jsonl"

Value = TypeVar('Value')
SelectorValue = TypeVar('SelectorValue')

class Select:
    def __init__(self, keys, queries, predicate: Callable[[Any, Any], bool]):
        self.keys = keys
        self.queries = queries
        self.predicate = predicate

# A mock version of the self object to simulate the function calls in eval_select
class MockEvaluator:
    def evaluate(self, expressions, xs: Sequence[Value]) -> Sequence[Value]:
        return expressions  # Assume expressions are already evaluated

class TestEvalSelect(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[145]

    def test_eval_select(self):
        passed_count = 0
        failed_count = 0
        results = []

        # Define test data
        keys = [1, 2, 3]
        queries = [3, 2, 1]
        predicate = lambda k, q: k == q

        select = Select(keys, queries, predicate)
        evaluator = MockEvaluator()
        code = self.code_snippet

        # Execute the code and fetch eval_select function
        exec_globals = {
            'Select': Select,
            'np': np,
            'Sequence': Sequence,
            'SelectorValue': SelectorValue,
            'Value': Value,
            'MockEvaluator': MockEvaluator
        }

        exec_locals = {}
        try:
            exec(code, exec_globals, exec_locals)
            eval_select = exec_locals['eval_select']

            # Use eval_select to evaluate select, keys, queries
            result = eval_select(evaluator, select, [])

            expected_output = [
                [False, False, True],   # Query 3
                [False, True, False],   # Query 2
                [True, False, False]    # Query 1
            ]

            self.assertEqual(
                result,
                expected_output,
                f"Expected {expected_output}, but got {result}."
            )

            print("Code snippet test: PASSED all assertions.")
            passed_count += 1
            results.append({
                "function_name": "eval_select",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet test: FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "eval_select",
                "code": code,
                "result": "failed"
            })

        # Write the results to the test_result.jsonl file
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
            if rec.get("function_name") != "eval_select"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()