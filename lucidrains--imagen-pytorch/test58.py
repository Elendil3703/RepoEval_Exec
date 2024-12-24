import unittest
import json
import sys
import os
import re
from typing import Any  # Ensure Any is available in the injected environment

TEST_RESULT_JSONL = "test_result.jsonl"

class TestMaskedMeanFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and select the 58th code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[57]  # Get the 58th JSON element (index 57)

        if len(cls.code_snippet) < 1:
            raise ValueError("Expected non-empty code snippet")

    def test_code_snippet(self):
        """Test the functionality and correctness of masked_mean function."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        print(f"Running test for masked_mean function...")
        
        # ------------------- Static Checks -------------------
        if "masked_mean" not in code:
            print(f"FAILED, 'masked_mean' not found in code.\n")
            failed_count += 1
            results.append({
                "function_name": "masked_mean",
                "code": code,
                "result": "failed"
            })
            return

        func_pattern = r"def\s+masked_mean\s*\("
        if not re.search(func_pattern, code):
            print(f"FAILED, incorrect signature for 'masked_mean'.\n")
            failed_count += 1
            results.append({
                "function_name": "masked_mean",
                "code": code,
                "result": "failed"
            })
            return
        
        # ------------------- Dynamic Execution and Testing -------------------
        exec_globals = {
            'Any': Any,  # Inject Any
            'rearrange': lambda x, _: x,  # Dummy rearrange function
            'exists': lambda x: x is not None,  # Simple exists function
            'torch': sys.modules.get('torch', None)  # Add torch if available
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if masked_mean is defined and accessible
            if 'masked_mean' not in exec_locals:
                print(f"FAILED, 'masked_mean' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "masked_mean",
                    "code": code,
                    "result": "failed"
                })
                return
            
            masked_mean = exec_locals['masked_mean']

            # Sample test cases (replace with appropriate tensor if torch is available, else mock)
            try:
                import torch
                t = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
                mask = torch.tensor([[1, 0], [0, 1]])
            except ImportError:
                t = [[1.0, 2.0], [3.0, 4.0]]  # Mock list for illustration
                mask = [[1, 0], [0, 1]]  # Mock list for illustration

            # Case 1: Without mask
            result_no_mask = masked_mean(t, dim=1)
            expected_no_mask = sum(t[0]) / len(t[0])  # Simplistic expectation for illustration

            # If torch is used, should use torch equivalent checks
            if sys.modules.get('torch', None):
                self.assertTrue(torch.allclose(result_no_mask, torch.tensor([1.5, 3.5])),
                                "Failed without mask")
            else:
                self.assertEqual(result_no_mask, expected_no_mask, "Failed without mask")

            # Case 2: With mask
            result_with_mask = masked_mean(t, dim=1, mask=mask)

            # Expected logic: First row includes only first element, second row only second element
            if sys.modules.get('torch', None):
                self.assertTrue(torch.allclose(result_with_mask, torch.tensor([1.0, 4.0])),
                                "Failed with mask")
            else:
                self.assertEqual(result_with_mask, [1.0, 4.0], "Failed with mask")

            print(f"PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "masked_mean",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "masked_mean",
                "code": code,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= Write Results to test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for masked_mean
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "masked_mean"]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()