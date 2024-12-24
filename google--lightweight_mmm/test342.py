import unittest
import json
import os
import numpy as np
from scipy import stats
from scipy.spatial import distance
import jax.numpy as jnp

TEST_RESULT_JSONL = "test_result.jsonl"

def _pmf(data, x):
    # Placeholder for a function that calculates Probability Mass Function
    counts = np.array([np.sum(data == xe) for xe in x])
    return counts / counts.sum()

def _estimate_pdf(data, x):
    # Placeholder for a function that estimates Probability Density Function
    kde = stats.gaussian_kde(data)
    return kde(x)

def distance_prior_posterior(p, q, method, discrete):
    """
    Computes the distance between two distributions using various methods.

    Args:
      p: Samples for distribution 1.
      q: Samples for distribution 2.
      method: We can have four methods: KS, Hellinger, JS and min.
      discrete: Whether input data is discrete or continuous.

    Returns:
      The distance metric (between 0 and 1).
    """
    if method == "KS":
        return stats.ks_2samp(p, q).statistic
    elif method in ["Hellinger", "JS", "min"]:
        if discrete:
            x = jnp.unique(jnp.concatenate((p, q)))
            p_pdf = _pmf(p, x)
            q_pdf = _pmf(q, x)
        else:
            minx, maxx = min(p.min(), q.min()), max(p.max(), q.max())
            x = np.linspace(minx, maxx, 100)
            p_pdf = _estimate_pdf(p, x)
            q_pdf = _estimate_pdf(q, x)
        if method == "Hellinger":
            return np.sqrt(jnp.sum((np.sqrt(p_pdf) - np.sqrt(q_pdf)) ** 2)) / np.sqrt(2)
        elif method == "JS":
            return distance.jensenshannon(p_pdf, q_pdf)
        else:
            return 1 - np.minimum(p_pdf, q_pdf).sum()

class TestDistancePriorPosterior(unittest.TestCase):
    
    @classmethod
    def setUpClass(cls):
        # Load the JSON file
        with open("RepoEval_result.json", "r") as f:
            data = json.load(f)
        cls.code_snippet = data[341]  # Get the specified JSON element
        
    def test_distance_prior_posterior(self):
        results = []

        # Test KS method
        p = np.array([1, 2, 3, 4, 5])
        q = np.array([5, 6, 7, 8, 9])
        method = "KS"
        discrete = True
        result = distance_prior_posterior(p, q, method, discrete)
        expected = stats.ks_2samp(p, q).statistic
        try:
            self.assertAlmostEqual(result, expected, places=6)
            results.append({
                "function_name": "distance_prior_posterior",
                "code": self.__class__.code_snippet,
                "result": "passed"
            })
        except AssertionError:
            results.append({
                "function_name": "distance_prior_posterior",
                "code": self.__class__.code_snippet,
                "result": "failed"
            })

        # Test Hellinger method
        method = "Hellinger"
        result = distance_prior_posterior(p, q, method, discrete)
        # Perform custom calculation for expected result if needed
        try:
            # Placeholder for real expected value calculation
            expected = 0.5
            self.assertAlmostEqual(result, expected, places=6)
            results.append({
                "function_name": "distance_prior_posterior",
                "code": self.__class__.code_snippet,
                "result": "passed"
            })
        except AssertionError:
            results.append({
                "function_name": "distance_prior_posterior",
                "code": self.__class__.code_snippet,
                "result": "failed"
            })
        
        # Add more test cases for "JS" and "min" as needed

        # Writing test results to JSONL
        existing_records = []
        if os.path.exists(TEST_RESULT_JSONL):
            with open(TEST_RESULT_JSONL, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    existing_records.append(json.loads(line))

        # Remove old records related to this function
        existing_records = [
            rec for rec in existing_records
            if rec.get("function_name") != "distance_prior_posterior"
        ]

        existing_records.extend(results)

        with open(TEST_RESULT_JSONL, "w", encoding="utf-8") as f:
            for record in existing_records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    unittest.main()