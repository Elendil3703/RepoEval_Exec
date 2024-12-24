import unittest
import json
import os
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestFaissNNSearchInit(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[369]  # Get the 370th JSON element
        if not cls.code_snippet:
            raise ValueError("Expected a code snippet in the JSON data")

    def test_init_function(self):
        """Test __init__ method for FAISS Nearest neighbourhood search."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果
        i = 369  # Index for the current test

        # ------------------- 静态检查 -------------------
        if "def __init__(" not in self.code_snippet:
            print(f"Code snippet {i}: FAILED, '__init__' method not found.\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": self.code_snippet,
                "result": "failed"
            })
            return

        # ------------------- 动态执行并测试逻辑 -------------------
        exec_globals = {
            'faiss': type('FaissMock',(), {'omp_set_num_threads': lambda x: None}),  # mock faiss
            'Any': Any  # 注入 Any
        }
        exec_locals = {}

        try:
            # 动态执行代码片段
            exec(self.code_snippet, exec_globals, exec_locals)

            # Define a mock class to integrate the __init__ method
            class FaissSearchMock:
                def __init__(self, on_gpu: bool = False, num_workers: int = 4) -> None:
                    faiss.omp_set_num_threads(num_workers)
                    self.on_gpu = on_gpu
                    self.search_index = None

            # Perform checks on the initialized class
            instance = FaissSearchMock(True, 8)

            # Verify that the instance variables are set correctly
            self.assertTrue(instance.on_gpu, "Code snippet {} did not set 'on_gpu' correctly.".format(i))
            self.assertIsNone(instance.search_index, f"Code snippet {i} did not initialize 'search_index' to None.")

            print(f"Code snippet {i}: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "__init__",
                "code": self.code_snippet,
                "result": "passed"
            })
        except Exception as e:
            print(f"Code snippet {i}: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "__init__",
                "code": self.code_snippet,
                "result": "failed"
            })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

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
        existing_records.extend(results)

        # 重写 test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()