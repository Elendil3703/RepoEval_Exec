import unittest
import json
import os
import pandas as pd
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestShuffleFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and retrieve the specific code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[428]  # Get the 429th JSON element (428th index)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the specified JSON array")
        
    def test_shuffle_function(self):
        """Dynamically test all code snippets with the specified function 'shuffle'."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if 'def shuffle' not in code:
                    print(f"Code snippet {i}: FAILED, function 'shuffle' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "shuffle",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'pd': pd,
                    'Any': Any
                }
                
                exec_locals = {}

                try:
                    # Execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if 'shuffle' is defined
                    if 'shuffle' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'shuffle' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "shuffle",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Prepare a test DataFrame
                    df = pd.DataFrame({'a': range(10), 'b': range(10, 20)})
                    
                    # Test shuffle functionality
                    shuffled_df_1 = exec_locals['shuffle'](df, seed=42)
                    shuffled_df_2 = exec_locals['shuffle'](df, seed=42)
                    shuffled_df_diff = exec_locals['shuffle'](df, seed=24)

                    # Check if shuffling with the same seed results in the same order
                    pd.testing.assert_frame_equal(shuffled_df_1, shuffled_df_2)

                    # Check if shuffling with a different seed results in a different order
                    self.assertFalse(shuffled_df_1.equals(shuffled_df_diff), "Shuffled frames with different seeds should not be the same.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "shuffle",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "shuffle",
                        "code": code,
                        "result": "failed"
                    })

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Writing results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for 'shuffle'
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "shuffle"
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