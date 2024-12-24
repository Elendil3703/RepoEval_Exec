import unittest
import json
import sys
import os

TEST_RESULT_JSONL = "test_result.jsonl"

class LastLayerToExtractReachedException(Exception):
    pass

class TestCallMethodResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[386]  # Get the 387th JSON element (index 386)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_call_method(self):
        """Dynamically test the __call__ method in all code snippets."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Gather results to write to JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")

                # ------------------- 静态检查 -------------------
                if "def __call__" not in code:
                    print(f"Code snippet {i}: FAILED, '__call__' method not found.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "__call__",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'LastLayerToExtractReachedException': LastLayerToExtractReachedException,
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # Class instance preparation
                    # Assumes the class which has `__call__` method is named `TestClass`
                    class_instance = exec_locals['TestClass']()
                    # Preparing a mock hook_dict and layer_name
                    class_instance.hook_dict = {}
                    class_instance.layer_name = "test_layer"
                    class_instance.raise_exception_to_break = True
                    module, input, output = None, None, "MockOutput"

                    # Test the __call__ method
                    with self.assertRaises(LastLayerToExtractReachedException):
                        class_instance(module, input, output)

                    # Check if the output was saved correctly
                    self.assertIn(
                        class_instance.layer_name,
                        class_instance.hook_dict,
                        f"Code snippet {i} did not correctly set the layer output in hook_dict."
                    )
                    self.assertEqual(
                        class_instance.hook_dict[class_instance.layer_name],
                        output,
                        f"Code snippet {i} did not correctly store the output in hook_dict."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "__call__",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "__call__",
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

        # 删除 function_name == "__call__" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "__call__"
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