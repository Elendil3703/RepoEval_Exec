import unittest
import json
import sys
import re
import os
from typing import Any
import jax.numpy as jnp
from unittest.mock import MagicMock

#变量重命名，方便调试
TEST_RESULT_JSONL = "test_result.jsonl"

class TestLightweightMMMSetup(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Mock the lightweight_mmm.LightweightMMM class and its methods
        cls.LightweightMMM = MagicMock()
        cls.national_mmm = cls.LightweightMMM()
        cls.geo_mmm = cls.LightweightMMM()
        cls.not_fitted_mmm = cls.LightweightMMM()

        # Mock the fit method to ensure it is called with correct parameters
        cls.national_mmm.fit = MagicMock()
        cls.geo_mmm.fit = MagicMock()

    def test_fit_called_with_correct_parameters(self):
        """Test that the fit method is called with the correct parameters."""
        # expected arguments for national_mmm.fit
        media_national = jnp.ones((50, 5))
        target_national = jnp.ones(50)
        media_prior_national = jnp.ones(5) * 50

        # expected arguments for geo_mmm.fit
        media_geo = jnp.ones((50, 5, 3))
        target_geo = jnp.ones((50, 3))
        media_prior_geo = jnp.ones(5) * 50

        self.national_mmm.fit.assert_called_with(
            media=media_national,
            target=target_national,
            media_prior=media_prior_national,
            number_warmup=2,
            number_samples=2,
            number_chains=1
        )

        self.geo_mmm.fit.assert_called_with(
            media=media_geo,
            target=target_geo,
            media_prior=media_prior_geo,
            number_warmup=2,
            number_samples=2,
            number_chains=1
        )

    def test_not_fitted_mmm_instance(self):
        """Test that not_fitted_mmm is an instance of LightweightMMM but fit is not called."""
        # Ensure that the fit method was never called
        self.not_fitted_mmm.fit.assert_not_called()

if __name__ == "__main__":
    unittest.main()

results = []
test_case = TestLightweightMMMSetup()
test_loader = unittest.TestLoader()
test_names = test_loader.getTestCaseNames(TestLightweightMMMSetup)
for name in test_names:
    test = test_case
    test._testMethodName = name
    try:
        result = test.defaultTestResult()
        test(result)
        for res in result.failures:
            results.append({
                "function_name": "setUpClass",
                "code": "Provided setup code for LightweightMMM testing",
                "result": "failed",
                "details": str(res)
            })
        for res in result.errors:
            results.append({
                "function_name": "setUpClass",
                "code": "Provided setup code for LightweightMMM testing",
                "result": "error",
                "details": str(res)
            })
        if not result.failures and not result.errors:
            results.append({
                "function_name": "setUpClass",
                "code": "Provided setup code for LightweightMMM testing",
                "result": "passed"
            })
    except Exception as e:
        results.append({
            "function_name": "setUpClass",
            "code": "Provided setup code for LightweightMMM testing",
            "result": "failed",
            "details": str(e)
        })

existing_records = []
if os.path.exists(TEST_RESULT_JSONL):
    with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            existing_records.append(json.loads(line))

# 删除 function_name == "setUpClass" 的旧记录
existing_records = [
    rec for rec in existing_records
    if rec.get("function_name") != "setUpClass"
]

# 将新结果附加
existing_records.extend(results)

# 重写 test_result.jsonl
with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
    for record in existing_records:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")