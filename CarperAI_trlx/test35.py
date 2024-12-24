import unittest
import json
import sys
import os
from unittest.mock import MagicMock

TEST_RESULT_JSONL = "test_result.jsonl"

class TestHfGetBranchClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[34]  # Get the 35th JSON element (index 34)

    def test_hf_get_branch_class(self):
        """Test code snippets for hf_get_branch_class."""
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        
        # Dynamically prepare execution environment
        exec_globals = {
            "transformers": MagicMock(),
            "GPTModelBranch": "GPTModelBranch",
            "OPTModelBranch": "OPTModelBranch",
            "BloomModelBranch": "BloomModelBranch",
        }
        exec_locals = {}

        try:
            # Execute the code snippet
            exec(code, exec_globals, exec_locals)

            # Retrieve hf_get_branch_class from executed locals
            if "hf_get_branch_class" not in exec_locals:
                print(f"FAILED, 'hf_get_branch_class' not found in exec_locals.")
                failed_count += 1
                results.append({
                    "function_name": "hf_get_branch_class",
                    "code": code,
                    "result": "failed"
                })
            else:
                hf_get_branch_class = exec_locals["hf_get_branch_class"]

                # Define mock configs
                gpt_config = MagicMock()
                gpt_config.architectures = ["GPT2LMHeadModel"]
                opt_config = MagicMock()
                opt_config.architectures = ["OPTForCausalLM"]
                bloom_config = MagicMock()
                bloom_config.architectures = ["BloomModel"]
                unsupported_config = MagicMock()
                unsupported_config.architectures = ["UnsupportedModel"]

                # Perform test assertions using mock objects
                self.assertEqual(hf_get_branch_class(gpt_config), "GPTModelBranch")
                self.assertEqual(hf_get_branch_class(opt_config), "OPTModelBranch")
                self.assertEqual(hf_get_branch_class(bloom_config), "BloomModelBranch")
                
                with self.assertRaises(ValueError):
                    hf_get_branch_class(unsupported_config)

                print(f"PASSED all assertions.")
                passed_count += 1
                results.append({
                    "function_name": "hf_get_branch_class",
                    "code": code,
                    "result": "passed"
                })
        except Exception as e:
            print(f"FAILED with error: {e}")
            failed_count += 1
            results.append({
                "function_name": "hf_get_branch_class",
                "code": code,
                "result": "failed"
            })

        # Write the results to the test_result.jsonl
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
            if rec.get("function_name") != "hf_get_branch_class"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()