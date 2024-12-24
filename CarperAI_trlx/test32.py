import unittest
import json
import sys
import os
from typing import Any
from unittest.mock import MagicMock
from transformers import PreTrainedModel
import torch.nn as nn
from copy import deepcopy

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxResultInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[31]  # Get the 32nd JSON element

    def test_init_function(self):
        """Test the __init__ function in the code snippet."""
        code = self.code_snippet
        passed = False
        try:
            # Dynamically define and execute the class with the __init__ function
            exec_globals = {
                'transformers': transformers,
                'nn': nn,
                'deepcopy': deepcopy,
                'hf_get_decoder_blocks': MagicMock(return_value=[MagicMock() for _ in range(12)]),
                'hf_get_decoder_final_norm': MagicMock(return_value=MagicMock()),
                'hf_get_lm_head': MagicMock(return_value=MagicMock()),
                'hf_get_hidden_size': MagicMock(return_value=768),
            }
            exec_locals = {}

            exec(code, exec_globals, exec_locals)

            # Instantiate PreTrainedModel mock for testing
            base_model_mock = MagicMock(spec=PreTrainedModel)
            base_model_mock.config = MagicMock()

            # Attempt to create an instance using the __init__ function
            num_layers_unfrozen = 4
            test_instance = exec_locals['TestModel'](
                base_model=base_model_mock,
                num_layers_unfrozen=num_layers_unfrozen
            )

            # Assertions to check if __init__ works as expected
            self.assertEqual(
                len(test_instance.decoder_blocks), num_layers_unfrozen,
                "Incorrect number of decoder blocks unfreezing."
            )
            self.assertIsInstance(
                test_instance.final_norm, nn.Module,
                "Final norm is not an instance of nn.Module."
            )
            self.assertIsInstance(
                test_instance.lm_head, nn.Module,
                "LM head is not an instance of nn.Module."
            )
            self.assertFalse(
                any(param.requires_grad for param in test_instance.parameters()),
                "Parameters are not frozen."
            )

            passed = True
            print("Code snippet: PASSED all assertions.\n")
        
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")

        self.assertTrue(passed, "The code snippet did not pass all tests.")

        # ============= 将测试结果写入 test_result.jsonl =============
        # 读取现有 test_result.jsonl（若不存在则忽略）
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # 删除 function_name == "__init__" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__init__"
        ]

        # 将新结果附加
        result = {
            "function_name": "__init__",
            "code": code,
            "result": "passed" if passed else "failed"
        }
        existing_records.append(result)

        # 重写 test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()