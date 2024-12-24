import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

T5_CONFIGS = {}

class MockT5Config:
    """Mock class to simulate T5Config behavior."""
    def __init__(self, d_model):
        self.d_model = d_model

    @classmethod
    def from_pretrained(cls, name):
        return cls(d_model=512)  # Mock d_model value for testing

class MockModel:
    """Mock class to simulate model behavior with a config."""
    def __init__(self, config):
        self.config = config

class TestGetEncodedDim(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[92]  # Get the 93rd JSON element
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 93rd JSON array")

    def test_code_snippets(self):
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Check for presence of 'T5_CONFIGS' and 'get_encoded_dim'
                if "T5_CONFIGS" not in code:
                    print(f"Code snippet {i}: FAILED, 'T5_CONFIGS' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_encoded_dim",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "def get_encoded_dim" not in code:
                    print(f"Code snippet {i}: FAILED, function 'get_encoded_dim' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_encoded_dim",
                        "code": code,
                        "result": "failed"
                    })
                    continue
                
                exec_globals = {
                    'T5_CONFIGS': {},
                    'T5Config': MockT5Config,
                    'MockModel': MockModel,
                }
                exec_locals = {}
                
                try:
                    exec(code, exec_globals, exec_locals)

                    if 'get_encoded_dim' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'get_encoded_dim' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "get_encoded_dim",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    get_encoded_dim = exec_locals['get_encoded_dim']

                    # Test the function
                    name = "mock_model"
                    d_model = get_encoded_dim(name)

                    self.assertEqual(
                        d_model, 
                        512, 
                        f"Code snippet {i}: Expected d_model to be 512, got {d_model}."
                    )

                    # Check caching behavior
                    T5_CONFIGS = exec_globals['T5_CONFIGS']
                    self.assertIn(
                        name, 
                        T5_CONFIGS, 
                        f"Code snippet {i}: Model '{name}' should be cached in T5_CONFIGS."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "get_encoded_dim",
                        "code": code,
                        "result": "passed"
                    })

                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_encoded_dim",
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
            if rec.get("function_name") != "get_encoded_dim"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()