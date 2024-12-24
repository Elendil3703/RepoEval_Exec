import unittest
import json
import sys
import re
import warnings
from typing import Any, Dict, Tuple
from pandas import DataFrame

TEST_RESULT_JSONL = "test_result.jsonl"

class TestSummarizeFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[444]  # Get the 445th JSON element (index 444)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 445th JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                # ------------------- 静态检查 -------------------
                # Ensure the "summarize" definition is in the snippet
                if "def summarize" not in code:
                    print(f"Code snippet {i}: FAILED, function 'summarize' not found.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "summarize",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # Check the signature for summarize
                func_pattern = r"def\s+summarize\s*\(\s*self\s*,\s*over\s*:\s*dict\[.*Column.*,\s*tuple\[.*Column,*.*\]\]\)\s*->\s*DataFrame\s*:"
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'summarize'.\n")
                    failed_count += 1
                    results.append({
                        "function_name": "summarize",
                        "code": code,
                        "result": "failed"
                    })
                    continue
                
                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'sys': sys,
                    'warnings': warnings,
                    'Dict': Dict,
                    'Tuple': Tuple,
                    'Column': Any,  # Assume Column is a placeholder for an actual type
                    'Func': Any,    # Assume Func is a placeholder for an actual type
                    'DataFrame': DataFrame
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # 检查 summarize 是否真的存在
                    if 'summarize' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'summarize' not implemented in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "summarize",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # Create a mock class to test with the summarize function
                    class MockDataFrame:
                        def rollup(self, over: Dict[Any, Tuple[Any, Any]]) -> DataFrame:
                            return DataFrame()

                    # Test the summarize method
                    mock_df = MockDataFrame()
                    with warnings.catch_warnings(record=True) as w:
                        warnings.simplefilter("always")
                        
                        # Call the summarize method
                        result = exec_locals['summarize'](mock_df, {})
                        
                        # Check if FutureWarning was raised
                        self.assertEqual(len(w), 1, f"Code snippet {i} did not raise a FutureWarning.")
                        self.assertTrue(issubclass(w[-1].category, FutureWarning))
                        
                        # Check if summarize method delegates to rollup
                        self.assertIsInstance(result, DataFrame, f"Code snippet {i} did not return a DataFrame.")

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "summarize",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "summarize",
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

        # 删除 function_name == "summarize" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "summarize"
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