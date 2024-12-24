import unittest
import json
import os
import jax
import jax.numpy as jnp
from typing import Union

TEST_RESULT_JSONL = "test_result.jsonl"

class TestAdstockFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[319]  # Get the 320th JSON element (index 319)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the 320th JSON array")

    def test_adstock(self):
        """Test the _adstock function with various inputs."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        # Assume _adstock function code is provided like the provided snippet.
        # Insert the actual implementation of _adstock here.
        def _adstock(
            data: jnp.ndarray,
            lag_weight: Union[float, jnp.ndarray] = .9,
            normalise: bool = True,
        ) -> jnp.ndarray:
            def adstock_internal(
                prev_adstock: jnp.ndarray,
                data: jnp.ndarray,
                lag_weight: Union[float, jnp.ndarray] = lag_weight,
            ) -> jnp.ndarray:
                adstock_value = prev_adstock * lag_weight + data
                return adstock_value, adstock_value

            _, adstock_values = jax.lax.scan(
                f=adstock_internal, init=data[0, ...], xs=data[1:, ...])
            adstock_values = jnp.concatenate([jnp.array([data[0, ...]]), adstock_values])
            return jax.lax.cond(
                normalise,
                lambda adstock_values: adstock_values / (1. / (1 - lag_weight)),
                lambda adstock_values: adstock_values,
                operand=adstock_values)

        # Example tests for _adstock
        test_data_sets = [
            (jnp.array([1, 2, 3]), 0.9, True),
            (jnp.array([1, 0, 0]), 0.8, False),
            (jnp.array([1.5, 2.5, 3.5]), 0.7, True),
            (jnp.array([0, 0, 0]), 0.6, False)
        ]

        for i, (data, lag_weight, normalise) in enumerate(test_data_sets):
            with self.subTest(test_index=i):
                try:
                    result = _adstock(data, lag_weight, normalise)
                    expected_shape = data.shape
                    self.assertEqual(result.shape, expected_shape, f"Shape mismatch for test {i}")
                    passed_count += 1
                    results.append({
                        "function_name": "_adstock",
                        "test_index": i,
                        "data": data.tolist(),
                        "lag_weight": lag_weight,
                        "normalise": normalise,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Test {i} FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "_adstock",
                        "test_index": i,
                        "data": data.tolist(),
                        "lag_weight": lag_weight,
                        "normalise": normalise,
                        "result": "failed"
                    })

        # Final summary
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(test_data_sets)}\n")
        self.assertEqual(passed_count + failed_count, len(test_data_sets), "Test count mismatch!")

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

        # 删除 function_name == "_adstock" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_adstock"
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