import unittest
import json
import os
from typing import Callable, Dict, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestHookFn(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[398]  # Get the 399th JSON element (index 398)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in JSON array")

    def test_hook_fn(self):
        """Test hook_fn implementation details in the provided code snippet."""
        exec_globals: Dict[str, Any] = {
            'head': lambda x: x,  # default head method simply returns its input
            'self': self,  # pass test instance to mimic instance behavior
        }
        
        exec_locals: Dict[str, Any] = {}
        
        # Mock `self` with necessary attributes for testing
        self.head_input_keys = [None, None]
        self.head_output_keys = ["output1", "output2"]
        self.outputs: Dict[str, Any] = {}
        self.input_key: str = "input"

        # Define mock module and outputs
        mock_module = None
        mock_input = None
        mock_output = "mock_output"

        # Execute the code snippet
        exec(self.code_snippet, exec_globals, exec_locals)

        hook_fn: Callable = exec_locals['hook_fn']

        # Test cases
        passed_count = 0
        failed_count = 0
        results = []

        try:
            # Test 1: Basic functionality with default head function
            hook_fn(mock_module, mock_input, mock_output)
            expected_output = {'input': mock_output}
            self.assertDictEqual(self.outputs, expected_output)
            print("Test 1: PASSED")
            passed_count += 1
            results.append({
                "function_name": "hook_fn",
                "code": self.code_snippet,
                "result": "passed"
            })
        except Exception as e:
            print(f"Test 1: FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "hook_fn",
                "code": self.code_snippet,
                "result": "failed"
            })

        try:
            # Test 2: Trigger ValueError on duplicate output key
            self.outputs = expected_output  # Set outputs to a non-empty dict
            hook_fn(mock_module, mock_input, mock_output)
        except ValueError:
            print("Test 2: PASSED (ValueError correctly raised)")
            passed_count += 1
            results.append({
                "function_name": "hook_fn",
                "code": self.code_snippet,
                "result": "passed"
            })
        except Exception as e:
            print(f"Test 2: FAILED with different error: {e}")
            failed_count += 1
            results.append({
                "function_name": "hook_fn",
                "code": self.code_snippet,
                "result": "failed"
            })
        else:
            print("Test 2: FAILED (ValueError not raised when expected)")
            failed_count += 1
            results.append({
                "function_name": "hook_fn",
                "code": self.code_snippet,
                "result": "failed"
            })

        # Test summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {passed_count + failed_count}\n")
        self.assertEqual(passed_count + failed_count, 2, "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))
        
        # Remove old records for the hook_fn
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "hook_fn"
        ]
        
        # Add new results
        existing_records.extend(results)
        
        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()