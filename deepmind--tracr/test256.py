import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestEmbedFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[255]  # Get the 256th JSON element
        if not cls.code_snippet:
            raise ValueError("Expected at least one code snippet in the 256th JSON array")

    def test_embed_function(self):
        """Test the embed function dynamically with additional logic checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect test results to write to JSONL

        code = self.code_snippet
        print("Running test for embed code snippet...")

        # ------------------- Static Checks -------------------
        # 1) Static check: ensure 'embed' function is defined in the snippet
        if "def embed" not in code:
            print("Code snippet: FAILED, 'embed' function not found in code.\n")
            failed_count += 1
            results.append({
                "function_name": "embed",
                "code": code,
                "result": "failed"
            })
            return

        # ------------------- Dynamic Execution and Logic Testing -------------------
        exec_globals = {'Any': Any}
        exec_locals = {}

        try:
            # Dynamic execution of the code snippet
            exec(code, exec_globals, exec_locals)

            # Check if 'embed' function truly exists after execution
            if 'embed' not in exec_locals:
                print("Code snippet: FAILED, 'embed' function not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "embed",
                    "code": code,
                    "result": "failed"
                })
                return

            # Assuming embed is expected to be tested with some mock inputs
            mock_tokens = [0, 1, 2, 3]
            try:
                # Call the embed function
                result = exec_locals['embed'](mock_tokens)
                # Additional logic checks on the result can be placed here

                # For demonstration, let's assume the result is supposed to be some non-empty structure
                self.assertIsNotNone(result, "Embed function returned None.")
                self.assertTrue(len(result) > 0, "Embed function returned an empty result.")

                print("Code snippet: PASSED all assertions.\n")
                passed_count += 1
                results.append({
                    "function_name": "embed",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                print(f"Code snippet: FAILED during embed function execution with error: {e}\n")
                failed_count += 1
                results.append({
                    "function_name": "embed",
                    "code": code,
                    "result": "failed"
                })

        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "embed",
                "code": code,
                "result": "failed"
            })

        # ============ Write test results to test_result.jsonl =============
        # Read existing test_result.jsonl (ignore if not exists)
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove outdated records with function_name == "embed"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "embed"
        ]

        # Append new results
        existing_records.extend(results)

        # Overwrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")


if __name__ == "__main__":
    unittest.main()