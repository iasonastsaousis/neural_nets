"""Small independent comparisons against the original loop implementations."""

import unittest

import numpy as np
from threadpoolctl import threadpool_limits

from main import KNearestNeighbors, NearestClassCentroid
from optimized import OptimizedKNearestNeighbors, OptimizedNearestClassCentroid


class OptimizedTests(unittest.TestCase):
    def test_reference_predictions_and_batch_boundaries(self):
        rng = np.random.default_rng(27)
        points = {c: rng.random((7, 12), dtype=np.float32) for c in (4, 2, 8)}
        queries = rng.random((11, 12), dtype=np.float32)
        with threadpool_limits(limits=2):
            for size in (1, 4, 20):
                optimized = OptimizedKNearestNeighbors(batch_size=size).fit(points)
                predictions = optimized.predict_for_k(queries)
                for k in (1, 3):
                    original = KNearestNeighbors(k=k)
                    original.fit(points)
                    expected = [original.predict(q) for q in queries]
                    np.testing.assert_array_equal(predictions[k], expected)
                original_ncc = NearestClassCentroid()
                original_ncc.fit(points)
                fast_ncc = OptimizedNearestClassCentroid(batch_size=size).fit(points)
                np.testing.assert_array_equal(
                    fast_ncc.predict_many(queries),
                    [original_ncc.predict(q) for q in queries],
                )

    def test_duplicate_and_equidistant_points(self):
        points = {2: np.array([[1., 0.], [1., 0.]]),
                  0: np.array([[-1., 0.], [0., 1.]]),
                  1: np.array([[0., -1.]])}
        queries = np.array([[0., 0.], [1., 0.]])
        optimized = OptimizedKNearestNeighbors().fit(points)
        for k in (1, 3, 5):
            original = KNearestNeighbors(k=k)
            original.fit(points)
            np.testing.assert_array_equal(
                optimized.predict_for_k(queries, (k,))[k],
                [original.predict(q) for q in queries],
            )

    def test_ncc_preserves_insertion_order_on_ties(self):
        points = {2: np.array([[1., 0.]]), 0: np.array([[-1., 0.]])}
        self.assertEqual(OptimizedNearestClassCentroid().fit(points).predict([0., 0.]), 2)

    def test_empty_queries_and_invalid_inputs(self):
        model = OptimizedKNearestNeighbors(k=1)
        with self.assertRaises(ValueError):
            model.predict([0., 0.])
        model.fit({0: np.array([[0., 0.]])})
        self.assertEqual(model.predict_many(np.empty((0, 2))).shape, (0,))
        with self.assertRaises(ValueError):
            model.kneighbors(np.zeros((1, 3)))
        with self.assertRaises(ValueError):
            model.kneighbors(np.zeros((1, 2)), k=2)


if __name__ == "__main__":
    unittest.main()
