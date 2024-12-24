import unittest
import json
import sys
import os
import pandas as pd
import numpy as np
from typing import Sequence, Optional

TEST_RESULT_JSONL = "test_result.jsonl"

class TestComputeSpendFractions(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[335]  # Get the 336th JSON element

        if not cls.code_snippet:
            raise ValueError("Expected a code snippet at the 336th position")

    def test_compute_spend_fractions(self):
        """Dynamically test the _compute_spend_fractions function with various cases."""
        results = []  # Collect test results to write to JSONL

        # Execute the code snippet to define _compute_spend_fractions
        exec_globals = {
            'pd': pd,
            'jnp': np,
            'Optional': Optional,
            'Sequence': Sequence
        }
        exec_locals = {}

        try:
            exec(self.code_snippet, exec_globals, exec_locals)

            if '_compute_spend_fractions' not in exec_locals:
                print(f"FAILED: '_compute_spend_fractions' not defined after exec.\n")
                results.append({
                    "function_name": "_compute_spend_fractions",
                    "code": self.code_snippet,
                    "result": "failed"
                })
                return

            _compute_spend_fractions = exec_locals['_compute_spend_fractions']

            # Test cases
            test_cases = [
                # Normal case
                (np.array([100, 200, 300]), ["Channel1", "Channel2", "Channel3"], "Spend Fraction", True),
                # Zero cost case
                (np.array([0, 250, 500]), ["Channel1", "Channel2", "Channel3"], "Spend Fraction", False),
                # Negative cost case
                (np.array([-50, 250, 500]), ["Channel1", "Channel2", "Channel3"], "Spend Fraction", False),
                # Custom column name
                (np.array([100, 200]), ["A", "B"], "Custom Column", True),
            ]

            passed_count = 0
            failed_count = 0

            for i, (cost_data, channel_names, column_name, should_pass) in enumerate(test_cases):
                with self.subTest(test=i):
                    try:
                        result = _compute_spend_fractions(cost_data, channel_names, column_name)
                        if should_pass:
                            self.assertTrue((result[column_name] > 0).all(), f"Test {i} failed: Expected all positive fractions.")
                            print(f"Test {i}: PASSED\n")
                            passed_count += 1
                        else:
                            self.fail(f"Test {i} failed: Expected exception for non-positive costs.")
                    except Exception as e:
                        if should_pass:
                            print(f"Test {i}: FAILED with unexpected error: {e}\n")
                            failed_count += 1
                        else:
                            print(f"Test {i}: PASSED expected ValueError.\n")
                            passed_count += 1

            results.append({
                "function_name": "_compute_spend_fractions",
                "code": self.code_snippet,
                "result": "passed" if failed_count == 0 else "failed"
            })

        except Exception as e:
            print(f"EXECUTION FAILED: {e}\n")
            results.append({
                "function_name": "_compute_spend_fractions",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Write to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_compute_spend_fractions"
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