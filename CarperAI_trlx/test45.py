import unittest
import json
import sys
import re
import os
from typing import Any
import transformers  # Ensure transformers is available

TEST_RESULT_JSONL = "test_result.jsonl"

class MockModelParentClass:
    @classmethod
    def from_config(cls, config, **kwargs):
        return f"MockBaseModel with config {config} and kwargs {kwargs}"

class MockModel:
    _auto_model_parent_class = MockModelParentClass

    def __init__(self, base_model, **kwargs):
        self.base_model = base_model
        self.kwargs = kwargs

    @classmethod
    def _split_kwargs(cls, kwargs):
        # Dummy implementation
        return kwargs, {}

    @classmethod
    def from_config(cls, config, **kwargs):
        if kwargs is not None:
            wrapped_model_kwargs, from_config_kwargs = cls._split_kwargs(kwargs)
        else:
            from_config_kwargs = {}
            wrapped_model_kwargs = {}
        base_model = cls._auto_model_parent_class.from_config(config, **from_config_kwargs)
        model = cls(base_model, **wrapped_model_kwargs)
        return model

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[44]  # Get the 45th JSON element (index 44)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON."""
        passed_count = 0
        failed_count = 0
        results = []

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # Static check for function existence
                if "def from_config" not in code:
                    print(f"Code snippet {i}: FAILED, 'from_config' function not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "from_config",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+from_config\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'from_config'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "from_config",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and testing logic
                exec_globals = {
                    'transformers': transformers,
                    'MockModel': MockModel,
                    'MockModelParentClass': MockModelParentClass
                }
                exec_locals = {}

                try:
                    # Dynamically execute the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Check if from_config exists in exec_locals
                    if 'from_config' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'from_config' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "from_config",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create a mock config and test from_config functionality
                    config = transformers.PretrainedConfig()
                    result_model = exec_locals['from_config'](MockModel, config, param1='value')

                    # Check if model returned is as expected
                    expected_base_model = exec_globals['MockModelParentClass'].from_config(config, param1='value')
                    self.assertEqual(result_model.base_model, expected_base_model)

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "from_config",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "from_config",
                        "code": code,
                        "result": "failed"
                    })

        # Summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

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
            if rec.get("function_name") != "from_config"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()