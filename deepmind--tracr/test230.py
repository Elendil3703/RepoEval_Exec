import unittest
import json
import os
from typing import Sequence

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCombineInParallel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[229]  # Get the 230th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 230th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Gather test results to be written to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # ------------------- Check for the method 'combine_in_parallel' -------------------
                if "def combine_in_parallel" not in code:
                    print(f"Code snippet {i}: FAILED, function 'combine_in_parallel' not found.\n")
                    failed_count += 1
                    # Write the failure record
                    results.append({
                        "function_name": "combine_in_parallel",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Check correct signature for 'combine_in_parallel'
                if "-> \"MLP\":" not in code:
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'combine_in_parallel'.\n")
                    failed_count += 1
                    # Write the failure record
                    results.append({
                        "function_name": "combine_in_parallel",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- Dynamic execution and logic testing -------------------
                exec_globals = {
                    'vectorspace_fns': type('vectorspace_fns', (), {'Linear': type('Linear', (), {'combine_in_parallel': lambda x: x})}),
                    'MLP': type('MLP', (), {'__init__': lambda self, fst, snd, residual_space: None}),
                }
                exec_locals = {}

                try:
                    # Dynamic execution of the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Ensure the function 'combine_in_parallel' exists after execution
                    cls = exec_globals.get('MLP', None)
                    if not cls or not hasattr(cls, 'combine_in_parallel'):
                        print(f"Code snippet {i}: FAILED, 'combine_in_parallel' not found in exec_globals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "combine_in_parallel",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Test the logic of the combine_in_parallel method
                    mlps = [cls(1, 2, None), cls(3, 4, None)]
                    combined = cls.combine_in_parallel(cls, mlps)

                    # Perform assertions to ensure logic is correct
                    self.assertIsInstance(combined, cls, f"Code snippet {i}: returned object is not an instance of MLP.")
                    self.assertEqual(combined.fst, [1, 3], f"Code snippet {i}: fst attribute incorrect.")
                    self.assertEqual(combined.snd, [2, 4], f"Code snippet {i}: snd attribute incorrect.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "combine_in_parallel",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "combine_in_parallel",
                        "code": code,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # ============= Write results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if not exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records of function_name == "combine_in_parallel"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "combine_in_parallel"
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