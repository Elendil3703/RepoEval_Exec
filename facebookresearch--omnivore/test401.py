import unittest
import json
import sys
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class VisionSample:
    # This is a mock class for testing purposes.
    pass

class TestForwardSubBatch(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        # Get the 401st JSON element (index 400)
        cls.code_snippet = data[400]
        cls.code_content = cls.code_snippet.get("content", "")
        
        if not cls.code_content.strip():
            raise ValueError("The code snippet is empty")

    def test_forward_sub_batch(self):
        """Dynamically test the forward_sub_batch function."""
        passed_count = 0
        failed_count = 0
        results = []

        # Execute the code to test
        exec_globals = {
            'VisionSample': VisionSample,
        }
        exec_locals = {}

        try:
            # Dynamically execute the code snippet
            exec(self.code_content, exec_globals, exec_locals)

            # Check if forward_sub_batch exists
            if 'forward_sub_batch' not in exec_locals:
                self.fail("forward_sub_batch function not found in executed locals.")
                results.append({
                    "function_name": "forward_sub_batch",
                    "code": self.code_content,
                    "result": "failed"
                })
                return

            forward_sub_batch = exec_locals['forward_sub_batch']

            # Mock _get_trunk_fields and trunk for testing
            exec_locals['self']._get_trunk_fields = lambda: (["field1"], {"key1": "field2"})
            exec_locals['self'].trunk = lambda *args, **kwargs: print("Trunk called with:", args, kwargs)

            # Prepare a mock VisionSample
            mock_sample = VisionSample()
            setattr(mock_sample, "field1", "value1")
            setattr(mock_sample, "field2", "value2")

            try:
                forward_sub_batch(exec_locals['self'], mock_sample)
                print("Mock call to forward_sub_batch method successful.")
                passed_count += 1
                results.append({
                    "function_name": "forward_sub_batch",
                    "code": self.code_content,
                    "result": "passed"
                })
            except AssertionError as e:
                print(f"AssertionError in forward_sub_batch: {e}")
                failed_count += 1
                results.append({
                    "function_name": "forward_sub_batch",
                    "code": self.code_content,
                    "result": "failed"
                })

        except Exception as e:
            print(f"Error executing code snippet: {e}")
            failed_count += 1
            results.append({
                "function_name": "forward_sub_batch",
                "code": self.code_content,
                "result": "failed"
            })

        # Final statistics
        if passed_count + failed_count != 1:
            self.fail("Test count mismatch!")

        # Write the results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records with function_name == "forward_sub_batch"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "forward_sub_batch"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()