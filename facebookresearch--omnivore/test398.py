import unittest
import json
import os
import torch.nn as nn
from typing import Dict

TEST_RESULT_JSONL = "test_result.jsonl"

class DummyModel(nn.Module):
    def __init__(self):
        super(DummyModel, self).__init__()
        self.layer1 = nn.Linear(10, 10)
        self.layer2 = nn.Linear(10, 5)

    def forward(self, x):
        return self.layer2(self.layer1(x))

def load_state_dict_into_model(state_dict: Dict, model: nn.Module, strict: bool = True):
    """
    Loads a state dict into the given model.

    Args:
        state_dict: A dictionary containing the model's
            state dict, or a subset if strict is False
        model: Model to load the checkpoint weights into
        strict: raise if the state_dict has missing state keys
    """
    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=False)
    err = "State key mismatch."
    if unexpected_keys:
        err += f" Unexpected keys: {unexpected_keys}."
    if missing_keys:
        err += f" Missing keys: {missing_keys}."
    if unexpected_keys or missing_keys:
        if not unexpected_keys and not strict:
            logging.warning(err)
        else:
            raise KeyError(err)
    return model

class TestLoadStateDictIntoModel(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[397]  # Get the 398th JSON element

    def test_load_state_dict(self):
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        # Assume the provided code snippet is in 'code'
        code = self.code_snippet

        # Prepare a dummy model and a correct and incorrect state dict
        model = DummyModel()
        correct_state_dict = model.state_dict()
        incorrect_state_dict = {'layer1.weight': torch.randn(10, 10)}

        try:
            # Dynamically execute the code snippet to obtain the function
            exec_globals = {
                '__name__': '__main__',
                'nn': nn,  # Inject nn module
                'Dict': Dict,  # Inject Dict type
            }
            exec(code, exec_globals)

            # Test 1: Load correct state dict without exception
            try:
                load_state_dict_into_model = exec_globals['load_state_dict_into_model']
                load_state_dict_into_model(correct_state_dict, model, strict=True)
                passed_count += 1
                results.append({
                    "function_name": "load_state_dict_into_model",
                    "code": code,
                    "result": "passed"
                })
            except Exception as e:
                print(f"Failed to load correct state dict: {e}")
                failed_count += 1
                results.append({
                    "function_name": "load_state_dict_into_model",
                    "code": code,
                    "result": "failed"
                })

            # Test 2: Load incorrect state dict and check for KeyError
            try:
                load_state_dict_into_model(incorrect_state_dict, model, strict=True)
                print("Incorrect state dict loaded without exception")
                failed_count += 1
                results.append({
                    "function_name": "load_state_dict_into_model",
                    "code": code,
                    "result": "failed"
                })
            except KeyError:
                passed_count += 1
                results.append({
                    "function_name": "load_state_dict_into_model",
                    "code": code,
                    "result": "passed"
                })

        except Exception as e:
            print(f"Execution error: {e}")
            failed_count += 1
            results.append({
                "function_name": "load_state_dict_into_model",
                "code": code,
                "result": "failed"
            })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 2\n")
        self.assertEqual(passed_count + failed_count, 2, "Test count mismatch!")

        # ============= 将测试结果写入 test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # 删除 function_name == "load_state_dict_into_model" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "load_state_dict_into_model"
        ]

        # 将新结果附加
        existing_records.extend(results)

        # 重写 test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()