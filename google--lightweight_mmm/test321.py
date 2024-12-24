import unittest
import json
import sys
import re
import os
from typing import Any  # 确保注入的环境中有 Any
import jax.numpy as jnp
import numpyro

TEST_RESULT_JSONL = "test_result.jsonl"

def adstock(data, lag_weight, normalise):
    """Ground truth function for the test."""
    if data.ndim == 3:
        lag_weight = jnp.expand_dims(lag_weight, axis=-1)
    # Assuming _adstock is a pre-defined function
    return _adstock(data=data, lag_weight=lag_weight, normalise=normalise)

class TestAdstockResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[320]  # Get the 321st JSON element (0-indexed)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the JSON array")

    def test_code_snippets(self):
        """Dynamically test all code snippets in the JSON with additional checks."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                
                # ------------------- 多一些逻辑检查 -------------------
                # 1) 静态检查：判断 snippet 中是否定义了 adstock 函数
                if "def adstock" not in code:
                    print(f"Code snippet {i}: FAILED, function 'adstock' not found in code.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "adstock",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                func_pattern = r"def\s+adstock\s*\("
                if not re.search(func_pattern, code):
                    print(f"Code snippet {i}: FAILED, incorrect signature for 'adstock'.\n")
                    failed_count += 1
                    # 写入失败记录
                    results.append({
                        "function_name": "adstock",
                        "code": code,
                        "result": "failed"
                    })
                    continue

                # ------------------- 动态执行并测试逻辑 -------------------
                exec_globals = {
                    'sys': sys,
                    'jnp': jnp,
                    'numpyro': numpyro,
                    '_adstock': lambda data, lag_weight, normalise: data * lag_weight,
                    'Any': Any,  # 注入 Any
                }
                exec_locals = {}

                try:
                    # 动态执行代码片段
                    exec(code, exec_globals, exec_locals)

                    # 检查 adstock 是否真的存在
                    if 'adstock' not in exec_locals:
                        print(f"Code snippet {i}: FAILED, 'adstock' not found in exec_locals.\n")
                        failed_count += 1
                        results.append({
                            "function_name": "adstock",
                            "code": code,
                            "result": "failed"
                        })
                        continue

                    # 使用 adstock 测试注册逻辑
                    adstock_func = exec_locals['adstock']
                    data = jnp.array([[1, 2], [3, 4]])
                    lag_weight = jnp.array([0.5, 0.5])
                    expected_output = data * lag_weight

                    result_output = adstock_func(data, lag_weight, normalise=False)
                    self.assertTrue(
                        jnp.array_equal(result_output, expected_output),
                        f"Code snippet {i} did not produce the expected result."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "adstock",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "adstock",
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

        # 删除 function_name == "adstock" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "adstock"
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