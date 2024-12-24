import unittest
import json
import os
from typing import Any  # Ensure Any is available in injected environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSelectorAndFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[136]  # 获取第137组代码

        if not cls.code_snippet:
            raise ValueError("Expected code snippet to contain content.")

    def test_selector_and(self):
        """Test the 'selector_and' function in the provided code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        code = self.code_snippet

        # ------------------- 动态执行并测试 logic -------------------
        exec_globals = {
            'Any': Any,  # Inject Any if necessary
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if 'selector_and' exists
            if 'selector_and' not in exec_locals:
                print("Code snippet: FAILED, 'selector_and' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "selector_and",
                    "code": code,
                    "result": "failed"
                })
                return

            # Test cases for 'selector_and'
            class Selector: pass
            class Select(Selector): pass
            class SelectorAnd(Selector): 
                def __init__(self, fst, snd):
                    self.fst = fst
                    self.snd = snd

            def _attempt_simplify(fst, snd, func):
                # Dummy implementation for testing
                return None

            exec_locals.update({
                "Selector": Selector,
                "Select": Select,
                "SelectorAnd": SelectorAnd,
                "_attempt_simplify": _attempt_simplify,
            })

            selector_and = exec_locals['selector_and']

            # Create mock selector objects
            fst_select = Select()
            snd_select = Select()
            fst_selector = Selector()
            snd_selector = Selector()

            # Case 1: Testing simplify with instances of Select
            result = selector_and(fst_select, snd_select, simplify=True)
            self.assertIsInstance(result, SelectorAnd, "Failed: Should return SelectorAnd instance.")
            print("Case 1: PASSED")

            # Case 2: Testing without simplification
            result = selector_and(fst_selector, snd_selector, simplify=False)
            self.assertIsInstance(result, SelectorAnd, "Failed: Should return SelectorAnd instance.")
            print("Case 2: PASSED")

            passed_count += 2
            results.extend([{
                "function_name": "selector_and",
                "code": code,
                "result": "passed"
            } for _ in range(2)])

        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "selector_and",
                "code": code,
                "result": "failed"
            })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 2, "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if it doesn't exist)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for function_name == "selector_and"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "selector_and"
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