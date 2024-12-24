import unittest
import json
import os
import sys
import jax.numpy as jnp  # JAX 的 numpy 版本
from typing import Any  # 确保注入的环境中有 Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestZeroMlpsResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[234]  # Get the 235th JSON element (index 234)

    def test_zero_mlps(self):
        """Test the _zero_mlps method in the JSON code."""
        results = []  # 收集要写入 JSONL 的测试结果

        code = self.code_snippet
        with self.subTest(code_snippet=code):
            print(f"Running test for code snippet...")

            # Inject necessary globals
            exec_globals = {
                'jnp': jnp
            }
            code_with_class = (
                "class TestClass:\n"
                "    " + code.replace("\n", "\n    ")
            )

            try:
                # 动态执行代码片段
                exec(code_with_class, exec_globals)

                # 检查 _zero_mlps 是否定义在 exec_globals['TestClass']
                test_class = exec_globals.get('TestClass', None)
                if test_class is None or not hasattr(test_class, '_zero_mlps'):
                    raise AssertionError("'_zero_mlps' method not found in TestClass.")

                # Create an instance of TestClass
                test_instance = test_class()

                # Define a test case with 'mlp' and non-mlp parameters
                test_params = {
                    'layer1': {
                        'weights': jnp.array([1.0, 2.0]),
                        'bias': jnp.array([0.1, 0.2])
                    },
                    'mlp_layer': {
                        'weights': jnp.array([[0.5, 0.6], [0.7, 0.8]]),
                        'bias': jnp.array([0.3, 0.4])
                    }
                }
                expected_params = {
                    'layer1': {
                        'weights': jnp.array([1.0, 2.0]),
                        'bias': jnp.array([0.1, 0.2])
                    },
                    'mlp_layer': {
                        'weights': jnp.zeros_like(jnp.array([[0.5, 0.6], [0.7, 0.8]])),
                        'bias': jnp.zeros_like(jnp.array([0.3, 0.4]))
                    }
                }

                # Call _zero_mlps
                result_params = test_instance._zero_mlps(test_params)

                # Assert the result matches expected
                for key, subdict in expected_params.items():
                    for subkey, expected_value in subdict.items():
                        self.assertTrue(
                            jnp.array_equal(expected_value, result_params[key][subkey]),
                            f"{key}[{subkey}] does not match expected value: {expected_value} != {result_params[key][subkey]}"
                        )

                print("Code snippet: PASSED all assertions.\n")
                results.append({
                    "function_name": "_zero_mlps",
                    "code": code,
                    "result": "passed"
                })

            except Exception as e:
                print(f"Code snippet: FAILED with error: {e}\n")
                results.append({
                    "function_name": "_zero_mlps",
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

        # 删除 function_name == "_zero_mlps" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_zero_mlps"
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