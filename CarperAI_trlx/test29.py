import unittest
import json
import os
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class MockModel:
    def state_dict(self, *args, **kwargs):
        return {"weight": 1.0, "bias": 0.1}

class MockValueHead:
    def state_dict(self, *args, **kwargs):
        return {"v_weight": 0.5, "v_bias": 0.05}

class ModelWithValueHead:
    def __init__(self):
        self.base_model = MockModel()
        self.v_head = MockValueHead()

    def state_dict(self, *args, **kwargs):
        """
        Returns the state dictionary of the model. We add the state dictionary of the value head
        to the state dictionary of the wrapped model by prepending the key with `v_head.`.
        """
        base_model_state_dict = self.base_model.state_dict(*args, **kwargs)
        v_head_state_dict = self.v_head.state_dict(*args, **kwargs)
        for k, v in v_head_state_dict.items():
            base_model_state_dict[f"v_head.{k}"] = v
        return base_model_state_dict

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[28]  # Get the 29th JSON element (index 28)

    def test_state_dict(self):
        """Test the state_dict function."""
        model = ModelWithValueHead()
        expected_state_dict = {
            "weight": 1.0,
            "bias": 0.1,
            "v_head.v_weight": 0.5,
            "v_head.v_bias": 0.05,
        }

        results = []

        # Run tests against each code snippet
        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                exec_globals = {
                    'MockModel': MockModel,
                    'MockValueHead': MockValueHead,
                }
                exec_locals = {}
                try:
                    # Execute the snippet
                    exec(code, exec_globals, exec_locals)

                    # Use the exec_locals to get the state_dict and test
                    state_dict_result = exec_locals['ModelWithValueHead']().state_dict()

                    # Check if the state_dict matches the expected result
                    self.assertEqual(state_dict_result, expected_state_dict)
                    print(f"Code snippet {i}: PASSED.")
                    results.append({
                        "function_name": "state_dict",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}")
                    results.append({
                        "function_name": "state_dict",
                        "code": code,
                        "result": "failed"
                    })

        # Write results to JSONL
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
            if rec.get("function_name") != "state_dict"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()