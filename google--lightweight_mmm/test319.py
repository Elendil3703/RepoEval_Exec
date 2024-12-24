import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarryoverFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[318]  # Get the 319th JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 319th JSON array")

    def test_carryover_function(self):
        """Test all code snippets for the expected transformation behavior in carryover."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                if "carryover(" not in code:
                    print(f"Code snippet {i}: FAILED, 'carryover' use not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "carryover",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'Any': Any,
                    # Mock necessary modules and objects required by code
                    'numpyro': type('numpyro', (), {'plate': lambda *args, **kwargs: args[0], 'sample': lambda *args, **kwargs: 0}),
                    'priors': type('priors', (), {'AD_EFFECT_RETENTION_RATE': 'ad_rr', 'PEAK_EFFECT_DELAY': 'p_ed', 'get_default_priors': lambda: {'ad_rr': 0, 'p_ed': 0}}),
                    'custom_priors': {},
                    '_carryover': lambda *args, **kwargs: 'transformed_media_data',
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)
                    
                    assert 'numpyro' in exec_globals, "numpyro should be defined"
                    assert 'priors' in exec_globals, "priors should be defined"
                    assert 'custom_priors' in exec_globals, "custom_priors should be defined"
                    assert '_carryover' in exec_globals, "_carryover should be defined"
                    
                    # Attempt to call the function to ensure it is defined and executable
                    transformed_data = exec_globals['_carryover'](
                        data="media_data",
                        ad_effect_retention_rate=0.9,
                        peak_effect_delay=2,
                        number_lags=3
                    )
                    self.assertEqual(transformed_data, 'transformed_media_data', 
                                     f"Code snippet {i} did not transform data as expected.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "carryover",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "carryover",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

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
            if rec.get("function_name") != "carryover"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()