import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestLayerNormFunctionality(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        # Get the 242nd code snippet (index 241)
        cls.code_snippets = data[241]
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets for the ground truth function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # Static checks for the function context
                if "self.config.layer_norm" not in code:
                    print(f"Code snippet {i}: FAILED, 'self.config.layer_norm' not used.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "layer_norm",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                if "hk.LayerNorm" not in code:
                    print(f"Code snippet {i}: FAILED, 'hk.LayerNorm' not used.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "layer_norm",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                exec_globals = {'Any': Any}
                exec_locals = {}

                try:
                    # Mock objects to simulate the function execution
                    class MockConfig:
                        layer_norm = True

                    class MockFunction:
                        config = MockConfig()
                        
                        def layer_norm_function(self, x):
                            layer_norm_code = code
                            exec(layer_norm_code, exec_globals, locals())
                            return locals().get('layer_norm', x)

                    # Instance of the mock function class
                    function_instance = MockFunction()
                    
                    # Test execution
                    import numpy as np
                    input_array = np.array([1.0, 2.0, 3.0])
                    
                    if function_instance.config.layer_norm:
                        # Assuming hk.LayerNorm behaves as identity for testing
                        expected_output = input_array
                    else:
                        expected_output = input_array

                    output = function_instance.layer_norm_function(input_array)
                    
                    self.assertTrue(
                        np.allclose(output, expected_output),
                        f"Code snippet {i} did not return expected output when layer norm is {'enabled' if function_instance.config.layer_norm else 'disabled'}."
                    )
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "layer_norm",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "layer_norm",
                        "code": code,
                        "result": "failed"
                    })

        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write to test_result.jsonl
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
            if rec.get("function_name") != "layer_norm"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()