import unittest
import json
import os
from typing import Union, Any  # 确保注入的环境中有 Union 和 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSOpSub(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[118]  # Get the 119th code snippet

    def test_SOp_sub(self):
        """Dynamically test the __sub__ method in the JSON."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        code = self.code_snippet
        with self.subTest():
            print("Running test for __sub__ method...")
            # ------------------- Static Checks -------------------
            # Check if the right function is present in the code
            if "def __sub__" not in code:
                print("Code snippet: FAILED, '__sub__' function not found.\n")
                failed_count += 1
                results.append({
                    "function_name": "__sub__",
                    "code": code,
                    "result": "failed"
                })
                return

            # ------------------- Dynamic Execution and Logical Test -------------------
            exec_globals = {
                'Union': Union,
                'Any': Any,
                'Map': lambda func, x: (func(i) for i in x),  # Mock Map
                'SequenceMap': lambda func, x, y: (func(i, j) for i, j in zip(x, y)),  # Mock SequenceMap
                'NumericValue': (int, float)  # Mock NumericValue
            }
            exec_locals = {}

            try:
                # Dynamic execution of the code snippet
                exec(code, exec_globals, exec_locals)

                # Check if __sub__ truly exists in locals
                if '__sub__' not in exec_locals:
                    print("Code snippet: FAILED, '__sub__' not found in exec_locals.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__sub__",
                        "code": code,
                        "result": "failed"
                    })
                    return

                class MockSOp:
                    def __iter__(self):
                        return iter([1, 2, 3])

                # Create instances of SOp and test the __sub__ method
                sop = MockSOp()
                other_sop = MockSOp()

                # Test: SOp - SOp
                result = exec_locals['__sub__'](sop, other_sop)
                expected = SequenceMap(lambda x, y: x - y, sop, other_sop)

                self.assertEqual(
                    list(result),
                    list(expected),
                    "Code snippet did not correctly handle SOp - SOp."
                )
                
                # Test: SOp - NumericValue
                result = exec_locals['__sub__'](sop, 5)
                expected = Map(lambda x: x - 5, sop)

                self.assertEqual(
                    list(result),
                    list(expected),
                    "Code snippet did not correctly handle SOp - NumericValue."
                )

                print("Code snippet: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "__sub__",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "__sub__",
                    "code": code,
                    "result": "failed"
                })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (if it exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "__sub__"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__sub__"
        ]

        # Attach new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()