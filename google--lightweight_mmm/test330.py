import unittest
import json
import os
from typing import Optional, Tuple
import jax.numpy as jnp
import lightweight_mmm
from unittest.mock import Mock
import preprocessing

TEST_RESULT_JSONL = "test_result.jsonl"

class TestObjectiveFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and fetch the 330th code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[329]  # Selecting the 330th entry

    def test_objective_function(self):
        passed_count = 0
        failed_count = 0
        results = []

        code = self.code_snippet
        exec_globals = {
            'jnp': jnp,
            'lightweight_mmm': lightweight_mmm,
            'preprocessing': preprocessing
        }
        exec_locals = {}

        try:
            # Execute the code snippet to access _objective_function
            exec(code, exec_globals, exec_locals)

            # Check if function exists in the executed locals
            if '_objective_function' not in exec_locals:
                raise ValueError(f"'_objective_function' not found in the executed code.")

            # Retrieve the function
            _objective_function = exec_locals['_objective_function']

            # Mock objects and variables
            extra_features = jnp.array([[1, 2], [3, 4]])
            media_mix_model = Mock(spec=lightweight_mmm.LightweightMMM)
            media_mix_model.predict.return_value = jnp.array([[10, 20], [30, 40]])
            media_input_shape = (2, 2)
            media_gap = 2
            target_scaler = None
            media_scaler = Mock(spec=preprocessing.CustomScaler)
            media_scaler.transform.side_effect = lambda x: x  # No transformation
            geo_ratio = jnp.array([0.5, 0.5])
            seed = 42
            media_values = jnp.array([100, 200])

            # Call the function
            result = _objective_function(
                extra_features=extra_features,
                media_mix_model=media_mix_model,
                media_input_shape=media_input_shape,
                media_gap=media_gap,
                target_scaler=target_scaler,
                media_scaler=media_scaler,
                geo_ratio=geo_ratio,
                seed=seed,
                media_values=media_values
            )

            # Assert results and behavior
            self.assertIsInstance(result, jnp.float64, "Result should be a jnp.float64")
            self.assertTrue(media_mix_model.predict.called, "Media mix model predict method should be called")
            self.assertEqual(result, -100.0, "The negative sum of predictions should be -100.0 based on mock data")
            
            print(f"Code snippet: PASSED all assertions.\n")
            passed_count += 1
            results.append({
                "function_name": "_objective_function",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Code snippet: FAILED with error: {e}\n")
            failed_count += 1
            results.append({
                "function_name": "_objective_function",
                "code": code,
                "result": "failed"
            })
        
        # Write results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove existing records for _objective_function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "_objective_function"
        ]

        # Appending new results
        existing_records.extend(results)

        # Rewrite JSONL with the new results
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        print("Results have been written to test_result.jsonl")

if __name__ == "__main__":
    unittest.main()