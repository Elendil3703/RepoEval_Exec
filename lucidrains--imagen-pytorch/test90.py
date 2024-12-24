import unittest
import json
import os
from unittest.mock import MagicMock

TEST_RESULT_JSONL = "test_result.jsonl"

class TestCreateFunction(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        # Load the JSON file and extract the 90th code snippet
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[89]  # Get the 90th JSON element (index 89)

    def test_create_function(self):
        """Test the `create` function logic for various scenarios."""
        results = []
        success_count = 0
        failure_count = 0

        code = self.code_snippet

        # Mock classes to simulate the actual ones
        class NullUnet:
            def __init__(self, *args, **kwargs):
                pass

        class Unet:
            def __init__(self, *args, **kwargs):
                pass

        class Unet3D:
            def __init__(self, *args, **kwargs):
                pass

        class Imagen:
            def __init__(self, unets, **kwargs):
                self.unets = unets
                self.kwargs = kwargs

        class DummyConfig:
            unets = [MagicMock() for _ in range(3)]

            def dict(self):
                return {
                    'unets': [{'param1': 1}, {'param2': 2}, {'param3': 3}],
                    'video': True,
                    'other': 'args'
                }

        # Prepare the execution environment
        exec_globals = {
            'NullUnetConfig': MagicMock(),
            'NullUnet': NullUnet,
            'Unet': Unet,
            'Unet3D': Unet3D,
            'Imagen': Imagen,
            'MagicMock': MagicMock
        }

        exec_locals = {}

        try:
            exec(code, exec_globals, exec_locals)
            create_function = exec_locals.get('create')
            
            if create_function is None:
                raise RuntimeError("`create` function not defined in snippet.")
            
            instance = DummyConfig()
            imagen = create_function(instance)

            # Test the resulting Imagen instance
            self.assertIsInstance(imagen, Imagen, "The create function did not return an Imagen instance.")
            self.assertEqual(len(imagen.unets), 3, "The number of unets created is not 3.")

            for unet in imagen.unets:
                self.assertIsInstance(unet, Unet3D, "Unet is not of expected type for video.")

            print("Snippet executed and tested successfully.")
            success_count += 1
            results.append({
                "function_name": "create",
                "code": code,
                "result": "passed"
            })

        except Exception as e:
            print(f"Code snippet FAILED with error: {str(e)}")
            failure_count += 1
            results.append({
                "function_name": "create",
                "code": code,
                "result": "failed"
            })

        # Write the results to test_result.jsonl
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records for the `create` function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "create"
        ]

        # Append the new result
        existing_records.extend(results)

        # Rewrite the test_result.jsonl
        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()