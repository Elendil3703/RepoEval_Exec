import unittest
import json
import os
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruth(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[266]  # Get the 267th JSON element

        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 267th JSON array")

    def test_get_loss(self):
        """Test get_loss function from the code snippets."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- 静态检查 -------------------
                # 检查是否定义了必要的函数与对象
                if "get_loss" not in code:
                    print(f"Code snippet {i}: FAILED, 'get_loss' not found in code.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_loss",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'Any': Any,  # Inject Any type
                    'self': self,  # Mock self for class method
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # 检查 get_loss 是否存在于 exec_locals 中
                    self.assertIn("get_loss", exec_locals, "get_loss function is not defined.")

                    # Mock objects and methods necessary for testing
                    class MockSelf:
                        def __init__(self):
                            self.scaler = None
                            self.gas = 1

                        def training_step_exec(self, batch):
                            return {"loss": batch["loss"], "accuracy": batch["accuracy"]}

                        def _is_default_fp16(self):
                            return False

                    mock_self = MockSelf()

                    # Call the get_loss function using mock data
                    batch = {"loss": 42.0, "accuracy": 0.85}
                    loss, loss_dict = exec_locals['get_loss'](mock_self)

                    # 断言 loss 和 loss_dict 是否符合预期
                    self.assertEqual(loss, 42.0, "Loss value is incorrect.")
                    self.assertEqual(loss_dict, {"loss": 42.0, "accuracy": 0.85}, "Loss dict is incorrect.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "get_loss",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "get_loss",
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

        # 删除 function_name == "get_loss" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "get_loss"
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