import unittest
import json
import os
import re
from typing import Dict, Any

TEST_RESULT_JSONL = "test_result.jsonl"

def get_delta_modified_modules(config, modified_modules, num_layers_unfrozen):
    # Mock function to simulate getting modified modules.
    # Replace this with the real implementation if available.
    return modified_modules

MODIFIED_MODULES_DICT = {
    # Mock dictionary for supported model types and their default modified modules.
    # Replace this with the real implementation if available.
    "transformer": {
        "all": ["layer_1", "layer_2"],
        "attention": ["attention_1", "attention_2"],
        "mlp": ["mlp_1", "mlp_2"],
    }
}

class MockConfig:
    def __init__(self, model_type):
        self.model_type = model_type

class TestParseDeltaKwargs(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[23]  # Get the 25th JSON element (index 24)

        # Compile the expected function pattern
        cls.function_pattern = re.compile(r'def\s+parse_delta_kwargs\s*\(\s*delta_kwargs\s*,\s*config\s*,\s*num_layers_unfrozen\s*\)')

        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 25th JSON array")

    def test_function_existence_and_signature(self):
        """Test for the existence and signature of parse_delta_kwargs function."""
        results = []
        passed_count = 0
        failed_count = 0

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                # Static check: Ensure function definition exists with correct signature
                if not self.function_pattern.search(code):
                    failed_count += 1
                    results.append({
                        "function_name": "parse_delta_kwargs",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Dynamic execution and logic testing
                exec_globals = {
                    'get_delta_modified_modules': get_delta_modified_modules,
                    'MODIFIED_MODULES_DICT': MODIFIED_MODULES_DICT,
                    'MockConfig': MockConfig
                }
                exec_locals = {}

                try:
                    exec(code, exec_globals, exec_locals)
                    if 'parse_delta_kwargs' not in exec_locals:
                        failed_count += 1
                        results.append({
                            "function_name": "parse_delta_kwargs",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Testing with valid delta_kwargs
                    delta_kwargs = {"delta_type": "lora", "modified_modules": "all"}
                    config = MockConfig(model_type="transformer")
                    delta_type, modified_kwargs = exec_locals['parse_delta_kwargs'](delta_kwargs, config, 2)
                    
                    self.assertEqual(delta_type, "lora", "Unexpected delta type.")
                    self.assertEqual(
                        modified_kwargs["modified_modules"], ["layer_1", "layer_2"],
                        "Modified modules do not match expected result."
                    )

                    passed_count += 1
                    results.append({
                        "function_name": "parse_delta_kwargs",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    failed_count += 1
                    results.append({
                        "function_name": "parse_delta_kwargs",
                        "code": code,
                        "result": "failed"
                    })

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

        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "parse_delta_kwargs"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()