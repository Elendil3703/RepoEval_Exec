import unittest
import json
import os
import numpy as np
from unittest.mock import patch
from typing import Any

TEST_RESULT_JSONL = "test_result.jsonl"

class TestGroundTruthSetUpClass(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Assuming that lightweight_mmm and jnp are accessible in the test environment
        super(OptimizeMediaTest, cls).setUpClass()
        cls.national_mmm = lightweight_mmm.LightweightMMM()
        cls.national_mmm.fit(
            media=jnp.ones((50, 5)),
            target=jnp.ones(50),
            media_prior=jnp.ones(5) * 50,
            number_warmup=2,
            number_samples=2,
            number_chains=1)
        cls.geo_mmm = lightweight_mmm.LightweightMMM()
        cls.geo_mmm.fit(
            media=jnp.ones((50, 5, 3)),
            target=jnp.ones((50, 3)),
            media_prior=jnp.ones(5) * 50,
            number_warmup=2,
            number_samples=2,
            number_chains=1)

    def test_setUpClass_correctness(self):
        # This test ensures that setUpClass correctly initializes the models

        with patch('lightweight_mmm.LightweightMMM') as MockMMM:
            mock_instance = MockMMM.return_value
            mock_instance.fit.return_value = None

            # Calling setUpClass to see if it correctly calls LightweightMMM and fit
            TestGroundTruthSetUpClass.setUpClass()

            MockMMM.assert_called()
            self.assertEqual(mock_instance.fit.call_count, 2, "Expected fit to be called twice.")

            # Check fit calls for national_mmm
            call_args_national = mock_instance.fit.call_args_list[0]
            np.testing.assert_array_equal(call_args_national[1]['media'], np.ones((50, 5)))
            np.testing.assert_array_equal(call_args_national[1]['target'], np.ones(50))
            np.testing.assert_array_equal(call_args_national[1]['media_prior'], np.ones(5) * 50)

            # Check fit calls for geo_mmm
            call_args_geo = mock_instance.fit.call_args_list[1]
            np.testing.assert_array_equal(call_args_geo[1]['media'], np.ones((50, 5, 3)))
            np.testing.assert_array_equal(call_args_geo[1]['target'], np.ones((50, 3)))
            np.testing.assert_array_equal(call_args_geo[1]['media_prior'], np.ones(5) * 50)

        # Write result to test_result.jsonl
        results = [{
            "function_name": "setUpClass",
            "code": "Generated test for setUpClass function",
            "result": "passed"
        }]
        
        # Read existing records
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))
        
        # Overwrite previous results related to `setUpClass`
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "setUpClass"
        ]

        # Extend with new results
        existing_records.extend(results)

        # Write back to file
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")


# Note: __name__ == "__main__" block is omitted as per instruction