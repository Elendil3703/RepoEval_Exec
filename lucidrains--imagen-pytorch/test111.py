import unittest
import json
import os
from typing import Any, Iterator, Tuple

TEST_RESULT_JSONL = "test_result.jsonl"

class MockModel:
    def __init__(self, output_keyword_names):
        self.dl_tuple_output_keywords_names = output_keyword_names

    def forward(self, **kwargs):
        return sum(kwargs.values())  # A simple dummy implementation for testing

    def step_with_dl_iter(self, dl_iter, **kwargs):
        dl_tuple_output = self.cast_tuple(next(dl_iter))
        model_input = dict(list(zip(self.dl_tuple_output_keywords_names, dl_tuple_output)))
        loss = self.forward(**{**kwargs, **model_input})
        return loss

    def cast_tuple(self, v):
        return v if isinstance(v, tuple) else (v,)


class TestStepWithDlIter(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[110]  # Get the 111th JSON element (index 110)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the selected JSON array")

    def test_step_with_dl_iter(self):
        """Test the step_with_dl_iter function with various scenarios."""

        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- Dynamic Execution and Test -------------------

                exec_globals = {
                    'MockModel': MockModel,
                    'cast_tuple': lambda v: v if isinstance(v, tuple) else (v,),
                    'Iterator': Iterator,
                    'Tuple': Tuple,
                    'Any': Any,
                }
                exec_locals = {}

                try:
                    # Execute the code snippet to redefine the function
                    exec(code, exec_globals, exec_locals)

                    if 'step_with_dl_iter' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'step_with_dl_iter' not defined correctly.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "step_with_dl_iter",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Prepare test scenario
                    model = MockModel(['input1', 'input2'])
                    dl_iter = iter([(1, 2)])  # Simple deterministic iterator

                    # Test step_with_dl_iter functionality
                    loss = model.step_with_dl_iter(dl_iter)
                    self.assertEqual(loss, 3, f"Code snippet {i} yielded incorrect loss.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "step_with_dl_iter",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "step_with_dl_iter",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write results to test_result.jsonl =============
        # Load existing records if any
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "step_with_dl_iter"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "step_with_dl_iter"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite the JSONL file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()