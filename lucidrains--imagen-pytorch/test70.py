import unittest
import json
import os
import torch
import torch.nn as nn
from torch.nn import Conv2d

TEST_RESULT_JSONL = "test_result.jsonl"

class TestInitConvFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[69]  # Get the 70th JSON element (index 69)

    def test_init_conv_function(self):
        """Dynamically test the init_conv_ function."""
        results = []  # 收集要写入 JSONL 的测试结果

        code = self.code_snippet

        # Check for minimal expected presence of certain keywords
        if "torch" not in code or "nn.init" not in code:
            results.append({
                "function_name": "init_conv_",
                "code": code,
                "result": "failed"
            })
            return

        exec_globals = {
            'torch': torch,
            'nn': nn,
            'repeat': lambda x, pattern: x.repeat(4, 1, 1, 1)
        }
        exec_locals = {}

        try:
            # Dynamic execution of the code snippet
            exec(code, exec_globals, exec_locals)

            # Ensure `init_conv_` function is defined
            if 'init_conv_' not in exec_locals:
                results.append({
                    "function_name": "init_conv_",
                    "code": code,
                    "result": "failed"
                })
                return

            # Create a Conv2d layer to test the init_conv_ function
            conv = Conv2d(8, 16, 3)
            init_conv_ = exec_locals['init_conv_']

            # Call the init_conv_ function
            init_conv_(self, conv)

            # Verify the weight shape is correct
            self.assertEqual(conv.weight.shape, torch.Size([16, 8, 3, 3]),
                             "Weight shape was not maintained after initialization.")

            # Verify biases are initialized to zero
            self.assertTrue(torch.all(conv.bias == 0),
                            "Biases are not initialized to zero.")

            results.append({
                "function_name": "init_conv_",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            results.append({
                "function_name": "init_conv_",
                "code": code,
                "result": "failed"
            })

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

        # 删除 function_name == "init_conv_" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "init_conv_"
        ]

        # 将新结果附加
        existing_records.extend(results)

        # 重写 test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()