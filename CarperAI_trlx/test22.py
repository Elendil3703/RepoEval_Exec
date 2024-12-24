import unittest
import json
import sys
import os
from unittest.mock import Mock
import transformers
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGenerateLayerRegex(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[21]  # Get the 22nd JSON element (0-indexed)

    def test_generate_layer_regex(self):
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                if "generate_layer_regex" not in code:
                    print(f"Code snippet {i}: FAILED, 'generate_layer_regex' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "generate_layer_regex",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {
                    'transformers': transformers,
                    'hf_get_num_hidden_layers': lambda config: config.num_hidden_layers,
                    'regex_for_range': lambda start, end: r'\d+', # dummy implementation for testing
                    'Any': Any
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)

                    if 'generate_layer_regex' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'generate_layer_regex' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "generate_layer_regex",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    generate_layer_regex = exec_locals['generate_layer_regex']

                    # Mock config object
                    config = Mock(spec=transformers.PretrainedConfig)
                    config.num_hidden_layers = 12

                    # Test cases
                    test_cases = [
                        (-1, "(\d)+."),
                        (0, "(?:\d+)."),
                        (6, "(?:\d+)."),
                        (12, "(?:\d+)."),
                    ]

                    for num_layers_unfrozen, expected in test_cases:
                        result = generate_layer_regex(config, num_layers_unfrozen)
                        self.assertEqual(result, expected, f"Expected regex '{expected}' for num_layers_unfrozen={num_layers_unfrozen}, but got '{result}'.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "generate_layer_regex",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "generate_layer_regex",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write the test results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for generate_layer_regex
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "generate_layer_regex"
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