import unittest
import json
import os
from typing import Sequence, Any

# Constants
TEST_RESULT_JSONL = "test_result.jsonl"

class ConstantSelector:
    def __init__(self, value, check_length=False):
        self.value = value
        self.check_length = check_length

SelectorValue = Any

class TestEvalConstantSelector(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[146]  # Get the 147th JSON element

    def test_eval_constant_selector(self):
        """Test the eval_constant_selector function."""
        passed_count = 0
        failed_count = 0
        results = []
        
        code = self.code_snippet
        # Execute the code
        exec_globals = {
            'ConstantSelector': ConstantSelector,
            'Sequence': Sequence,
            'Value': Any,
            'SelectorValue': SelectorValue,
        }
        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)
            eval_constant_selector = exec_locals.get('eval_constant_selector', None)

            if eval_constant_selector is None:
                raise AssertionError("Function eval_constant_selector not found in executed code.")

            # Define test cases
            test_cases = [
                # Case where lengths don't match and exception is expected
                {'sel': ConstantSelector([1, 2, 3], check_length=True), 'xs': [1, 2], 'expected_exception': ValueError},
                # Case where lengths match
                {'sel': ConstantSelector([1, 2], check_length=True), 'xs': [1, 2], 'expected': [1, 2]},
                # Case where lengths don't need to match because check_length is False
                {'sel': ConstantSelector([1, 2, 3], check_length=False), 'xs': [1, 2], 'expected': [1, 2, 3]},
            ]

            for i, test in enumerate(test_cases):
                with self.subTest(test_index=i):
                    sel = test['sel']
                    xs = test['xs']
                    if 'expected_exception' in test:
                        with self.assertRaises(test['expected_exception']):
                            eval_constant_selector(self, sel, xs)
                        print(f"Test case {i}: PASSED (Exception raised as expected)")
                        
                        passed_count += 1
                        results.append({
                            "function_name": "eval_constant_selector",
                            "input": (sel.value, xs),
                            "result": "passed"
                        })
                    else:
                        result = eval_constant_selector(self, sel, xs)
                        self.assertEqual(result, test['expected'])
                        print(f"Test case {i}: PASSED (Result matched expected)")
                        
                        passed_count += 1
                        results.append({
                            "function_name": "eval_constant_selector",
                            "input": (sel.value, xs),
                            "result": "passed"
                        })
        except Exception as e:
            print(f"Exception during execution: {e}")
            failed_count += 1
            results.append({
                "function_name": "eval_constant_selector",
                "code": code,
                "result": "failed"
            })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed")

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
            if rec.get("function_name") != "eval_constant_selector"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()