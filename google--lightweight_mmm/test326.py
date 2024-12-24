import unittest
import json
import os
from typing import Any
import numpy as np
import jax.numpy as jnp
import numpyro

TEST_RESULT_JSONL = "test_result.jsonl"
_LAG_WEIGHT = "lag_weight"
_EXPONENT = "exponent"

# Mocking the necessary components from the media_transforms module
class media_transforms:
    @staticmethod
    def adstock(data, lag_weight, normalise):
        # Mock implementation of the adstock transformation
        transformed_data = data * lag_weight
        if normalise:
            transformed_data = transformed_data / jnp.sum(transformed_data, axis=0, keepdims=True)
        return transformed_data

    @staticmethod
    def apply_exponent_safe(data, exponent):
        # Mock implementation of applying a safe exponent transformation
        return data ** exponent

def _get_transform_default_priors():
    # Mock implementation for testing purposes
    return {
        "adstock": {
            _LAG_WEIGHT: lambda: jnp.array(0.5),
            _EXPONENT: lambda: jnp.array(1.0)
        }
    }

def transform_adstock(media_data, custom_priors, normalise=False):
    """
    Applies adstock and exponent transformations to the media data.

    Args:
      media_data: The media data to transform.
      custom_priors: Custom priors for the parameters, which override the
        default ones. The possible names of parameters for adstock and exponent
        are "lag_weight" and "exponent".
      normalise: Whether to normalise the output values.

    Returns:
      The transformed media data.
    """
    transform_default_priors = _get_transform_default_priors()["adstock"]
    with numpyro.plate(name=f"{_LAG_WEIGHT}_plate", size=media_data.shape[1]):
        lag_weight = numpyro.sample(
            name=_LAG_WEIGHT,
            fn=custom_priors.get(_LAG_WEIGHT, transform_default_priors[_LAG_WEIGHT]))

    with numpyro.plate(name=f"{_EXPONENT}_plate", size=media_data.shape[1]):
        exponent = numpyro.sample(
            name=_EXPONENT,
            fn=custom_priors.get(_EXPONENT, transform_default_priors[_EXPONENT]))

    if media_data.ndim == 3:
        lag_weight = jnp.expand_dims(lag_weight, axis=-1)
        exponent = jnp.expand_dims(exponent, axis=-1)

    adstock = media_transforms.adstock(
        data=media_data, lag_weight=lag_weight, normalise=normalise)

    return media_transforms.apply_exponent_safe(data=adstock, exponent=exponent)

class TestTransformAdstock(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippets = data[325]  # Get the 326th JSON element (index 325)
        if len(cls.code_snippets) < 1:
            raise ValueError("Expected at least one code snippet in the nth JSON array")

    def test_transform_adstock(self):
        """Testing transform_adstock function."""
        passed_count = 0  # Counter for passed tests
        failed_count = 0  # Counter for failed tests
        results = []      # Collect the results to write into JSONL

        for i, code in enumerate(self.code_snippets):
            with self.subTest(code_index=i):
                print(f"Running test for code snippet {i}...")
                exec_globals = {}
                exec_locals = {}

                try:
                    # Dynamic execution of the code snippet
                    exec(code, exec_globals, exec_locals)

                    # Create mock input data
                    media_data = jnp.array([[1, 2, 3], [4, 5, 6]])
                    custom_priors = {_LAG_WEIGHT: exec_globals['numpyro'].distributions.Uniform(0, 1),
                                     _EXPONENT: exec_globals['numpyro'].distributions.Normal(1, 0.1)}
                    
                    # Call transform_adstock
                    transformed_data = exec_locals['transform_adstock'](media_data, custom_priors, normalise=False)

                    # Check expected output shape
                    expected_shape = media_data.shape
                    self.assertEqual(
                        transformed_data.shape,
                        expected_shape,
                        f"Code snippet {i} produced an unexpected shape for transformed data."
                    )

                    # Check values are finite
                    self.assertTrue(
                        jnp.all(jnp.isfinite(transformed_data)),
                        f"Code snippet {i} resulted in non-finite values in transformed data."
                    )

                    print(f"Code snippet {i}: PASSED all assertions.\n")
                    passed_count += 1
                    results.append({
                        "function_name": "transform_adstock",
                        "code": code,
                        "result": "passed"
                    })
                except Exception as e:
                    print(f"Code snippet {i}: FAILED with error: {e}\n")
                    failed_count += 1
                    results.append({
                        "function_name": "transform_adstock",
                        "code": code,
                        "result": "failed"
                    })

        # Final statistics
        print(f"\nTest Summary: {passed_count} passed, {failed_count} failed, total {len(self.code_snippets)}\n")
        self.assertEqual(passed_count + failed_count, len(self.code_snippets), "Test count mismatch!")

        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for transform_adstock
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "transform_adstock"
        ]

        # Append new results
        existing_records.extend(results)

        # Overwrite test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()