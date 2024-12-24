import unittest
import json
import sys
import os
import re
from unittest.mock import MagicMock
import gc

TEST_RESULT_JSONL = "test_result.jsonl"

class TestPostInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[41]  # Get the 42nd JSON element (index 41)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_post_init_functionality(self):
        """Dynamically test all code snippets for the function 'post_init'."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果
        
        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # --------------- Check for function presence ---------------
                if "def post_init(self, state_dict):" not in code:
                    print(f"Code snippet {i}: FAILED, function 'post_init' not found or incorrect signature.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "post_init",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'MagicMock': MagicMock,
                    'gc': gc,
                }
                exec_locals = {}
                
                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # Mock a class with a matching signature to execute post_init
                    class TestModel:
                        def __init__(self):
                            self.ilql_heads = MagicMock()

                    # Create an instance of the model
                    model_instance = TestModel()

                    # Inject the post_init method into the instance
                    exec_locals['post_init'](model_instance, {
                        "ilql_heads.key1": "value1",
                        "ilql_heads.key2": "value2",
                        "other_key": "other_value"
                    })

                    # Check if keys are modified and passed correctly
                    expected_state_dict = {
                        "key1": "value1",
                        "key2": "value2",
                        "other_key": "other_value"
                    }

                    model_instance.ilql_heads.load_state_dict.assert_called_once_with(expected_state_dict, strict=False)
                    
                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "post_init",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "post_init",
                        "code": code,
                        "result": "failed"
                    })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

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

        # 删除 function_name == "post_init" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "post_init"
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