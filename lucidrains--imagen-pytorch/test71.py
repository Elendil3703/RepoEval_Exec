import unittest
import json
import sys
import re
import os
import torch
from torch import nn
from einops.layers.torch import Rearrange

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCarperAITrlxResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[70]  # Get the 71st JSON element (index 70)
        if len(cls.code_snippet) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippet(self):
        """Test Downsample function in the code snippet."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # To collect test results for JSONL writing

        code = self.code_snippet
        print("Running test for code snippet...")

        # 静态检查：检查 Python 代码的准确性和合理性
        if "def Downsample" not in code:
            print("Code snippet: FAILED, 'Downsample' function not found.\n")
            failed_count += 1
            results.append({
                "function_name": "Downsample",
                "code": code,
                "result": "failed"
            })
            return

        # 动态执行并测试逻辑
        exec_globals = {
            'nn': nn,
            'Rearrange': Rearrange,
            'default': lambda x, y: x if x is not None else y  # Providing default function as used in the snippet
        }
        exec_locals = {}

        try:
            # 动态执行代码片段
            exec(code, exec_globals, exec_locals)

            # 检查 Downsample 是否存在
            if 'Downsample' not in exec_locals:
                print("Code snippet: FAILED, 'Downsample' not found in exec_locals.\n")
                failed_count += 1
                results.append({
                    "function_name": "Downsample",
                    "code": code,
                    "result": "failed"
                })
                return

            # 实例化 Downsample 并进行测试
            downsample = exec_locals['Downsample']
            model = downsample(dim=4)

            # 创建一个假的输入来测试模型的实际行为
            input_tensor = torch.randn(1, 4, 8, 8)  # Batch size of 1, 4 channels, 8x8 spatial dimensions
            output_tensor = model(input_tensor)

            # 通过断言来检查输出维度匹配
            self.assertEqual(output_tensor.shape, (1, 16, 4, 4), "Output tensor shape mismatch.")

            print("Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "Downsample",
                "code": code,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "Downsample",
                "code": code,
                "result": "failed"
            })

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Delete old records with function_name == "Downsample"
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "Downsample"
        ]

        # Append new results
        existing_records.extend(results)

        # Rewrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()