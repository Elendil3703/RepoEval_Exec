import unittest
import json
import sys
import os
import pickle
import numpy as np
import jax.numpy as jnp
from typing import Any
from unittest.mock import mock_open, patch

TEST_RESULT_JSONL = "test_result.jsonl"

class TestLoadModelFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[337]  # Get the 338th JSON element (index 337)

    def test_load_model(self):
        """Dynamically test the load_model function in the JSON with additional checks."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        with self.subTest():
            print("Running test for load_model function...")

            # Static check
            if "def load_model" not in code:
                print("Code snippet: FAILED, 'load_model' function definition not found.\n")
                failed_count += 1
                results.append({
                    "function_name": "load_model",
                    "code": code,
                    "result": "failed"
                })
                return

            # Dynamic execution
            exec_globals = {
                'gfile': mock_open,
                'pickle': pickle,
                'np': np,
                'jnp': jnp,
                'Any': Any,
            }
            exec_locals = {}

            try:
                exec(code, exec_globals, exec_locals)

                if 'load_model' not in exec_locals:
                    print("Code snippet: FAILED, 'load_model' not found in exec_locals.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "load_model",
                        "code": code,
                        "result": "failed"
                    })
                    return

                # Mock model object
                class MockModel:
                    def __init__(self):
                        self.some_attr = np.array([1, 2, 3])
                        self.other_attr = "not an array"

                mock_model = MockModel()

                # Use mock open to simulate file operations
                with patch("builtins.open", mock_open(read_data=pickle.dumps(mock_model))):
                    # Use path to simulate gfile.GFile
                    with patch("gfile.GFile", mock_open(read_data=pickle.dumps(mock_model))):
                        loaded_model = exec_locals['load_model']("fake_path")

                self.assertIsInstance(
                    loaded_model.some_attr, jnp.ndarray,
                    "The attribute 'some_attr' should be converted to jnp.ndarray"
                )
                self.assertEqual(
                    loaded_model.other_attr, "not an array",
                    "The attribute 'other_attr' should remain unchanged"
                )

                print("Code snippet: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "load_model",
                    "code": code,
                    "result": "passed"
                })

            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "load_model",
                    "code": code,
                    "result": "failed"
                })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

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
            if rec.get("function_name") != "load_model"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()