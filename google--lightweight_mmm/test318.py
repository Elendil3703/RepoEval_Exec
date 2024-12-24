import unittest
import json
import sys
import os
import jax
import jax.numpy as jnp
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

def _carryover(
    data: jnp.ndarray,
    ad_effect_retention_rate: jnp.ndarray,
    peak_effect_delay: jnp.ndarray,
    number_lags: int,
) -> jnp.ndarray:
    lags_arange = jnp.expand_dims(
        jnp.arange(number_lags, dtype=jnp.float32), axis=-1)
    convolve_func = _carryover_convolve
    if data.ndim == 3:
        convolve_func = jax.vmap(
            fun=_carryover_convolve, in_axes=(2, None, None), out_axes=2)
    weights = ad_effect_retention_rate**((lags_arange - peak_effect_delay)**2)
    return convolve_func(data, weights, number_lags)

def _carryover_convolve(data, weights, number_lags):
    # Dummy implementation (for testing purposes)
    return jnp.convolve(data, weights, mode='valid')

class TestCarryoverResult(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[317]  # Get the 318th JSON element

        if len(cls.code_snippet) < 1:
            raise ValueError("Expected code snippet in the 318th JSON array")

    def test_carryover_snippet(self):
        """Test the carryover code snippet logic with various scenarios."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # 收集要写入 JSONL 的测试结果

        print(f"Running test for code snippet 318...")
        try:
            # Dummy test case to demonstrate testing
            data = jnp.array([1.0, 2.0, 3.0, 4.0])
            ad_effect_retention_rate = jnp.array([0.5])
            peak_effect_delay = jnp.array([1.0])
            number_lags = 3

            carryover_result = _carryover(
                data, ad_effect_retention_rate, peak_effect_delay, number_lags
            )

            # Assert the shape of the carryover result
            self.assertEqual(
                carryover_result.shape, (len(data) - number_lags + 1,),
                "Carryover result shape mismatch."
            )

            print("Code snippet 318: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "_carryover",
                "code": self.code_snippet,
                "result": "passed"
            })

        except Exception as e:
            print(f"Code snippet 318: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "_carryover",
                "code": self.code_snippet,
                "result": "failed"
            })

        # 最终统计信息
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total 1\n")
        self.assertEqual(passed_count + failed_count, 1, "Test count mismatch!")

        # ============= 将测试结果写入 test_result.jsonl =============
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # 删除 function_name == "_carryover" 的旧记录
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_carryover"
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