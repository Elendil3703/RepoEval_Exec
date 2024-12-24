import unittest
import json
import os
from typing import Dict, Any

TEST_RESULT_JSONL = "test_result.jsonl"

class MockModel:
    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        return {"layer1.weight": [0.1, 0.2], "layer1.bias": [0.3]}

class MockILQLHeads:
    def state_dict(self, *args, **kwargs) -> Dict[str, Any]:
        return {"head1.weight": [0.4, 0.5], "head1.bias": [0.6]}

class ModelWithILQL:
    def __init__(self):
        self.base_model = MockModel()
        self.ilql_heads = MockILQLHeads()

    def state_dict(self, *args, **kwargs):
        base_model_state_dict = self.base_model.state_dict(*args, **kwargs)
        ilql_heads_state_dict = self.ilql_heads.state_dict(*args, **kwargs)
        for k, v in ilql_heads_state_dict.items():
            base_model_state_dict[f"ilql_heads.{k}"] = v
        return base_model_state_dict

class TestStateDictMethod(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[40]  # Get the 41st JSON element (index 40)

    def test_state_dict(self):
        """Test the state_dict method implementation."""
        exec_globals = {}
        exec_locals = {}

        try:
            exec(self.code_snippet, exec_globals, exec_locals)
            test_class = exec_locals['ModelWithILQL'](*exec_globals.get('args', ()), **exec_globals.get('kwargs', {}))
            result_state_dict = test_class.state_dict()
            
            expected_state_dict = {
                "layer1.weight": [0.1, 0.2],
                "layer1.bias": [0.3],
                "ilql_heads.head1.weight": [0.4, 0.5],
                "ilql_heads.head1.bias": [0.6],
            }

            self.assertDictEqual(result_state_dict, expected_state_dict, "State dict does not match expected format.")
            test_result = {"function_name": "state_dict", "code": self.code_snippet, "result": "passed"}
        
        except Exception as e:
            test_result = {"function_name": "state_dict", "code": self.code_snippet, "result": "failed", "error": str(e)}

        # Read existing test results
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old results for the function
        existing_records = [rec for rec in existing_records if rec.get("function_name") != "state_dict"]

        # Append new result
        existing_records.append(test_result)

        # Write back to JSONL
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        
if __name__ == "__main__":
    unittest.main()