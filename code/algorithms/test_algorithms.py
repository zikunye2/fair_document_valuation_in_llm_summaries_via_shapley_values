"""Small exact games test mathematical invariants independently of paper data."""
import itertools
import unittest

import numpy as np

from run_baselines import exact_shapley, monte_carlo
from run_cluster import components, separated_clusters, quotient_game, distribute
import random


class AlgorithmTests(unittest.TestCase):
    def test_additive_game(self):
        weights = [1., -2., 3.]
        values = np.array([sum(w for i, w in enumerate(weights) if mask & (1 << i)) for mask in range(8)])
        np.testing.assert_allclose(exact_shapley(values), weights, atol=1e-12)
        estimate, _, _ = monte_carlo(values, 3, random.Random(8))
        np.testing.assert_allclose(estimate, weights, atol=1e-12)

    def test_exact_matches_all_permutations_with_interactions(self):
        values = np.array([0., 1., 2., 8., 4., 3., 9., 5.])
        expected = np.zeros(3)
        for permutation in itertools.permutations(range(3)):
            mask = 0
            for i in permutation:
                new_mask = mask | (1 << i)
                expected[i] += (values[new_mask] - values[mask]) / 6
                mask = new_mask
        np.testing.assert_allclose(exact_shapley(values), expected, atol=1e-12)
        self.assertAlmostEqual(sum(exact_shapley(values)), values[-1])

    def test_chaining_and_separation(self):
        distances = np.array([[0., .1, .3], [.1, 0., .1], [.3, .1, 0.]])
        np.testing.assert_array_equal(components(distances, .15), [0, 0, 0])
        labels, epsilon, _ = separated_clusters(distances, .15)
        self.assertEqual(len(set(labels)), 3)
        self.assertLess(epsilon, .1)

    def test_singleton_partition_is_exact_and_coarse_partition_efficient(self):
        values = np.array([0., 1., 2., 8., 4., 3., 9., 5.])
        reduced, clusters = quotient_game(values, np.arange(3))
        np.testing.assert_array_equal(reduced, values)
        np.testing.assert_allclose(distribute(exact_shapley(reduced), clusters, 3), exact_shapley(values))
        reduced, clusters = quotient_game(values, np.zeros(3, dtype=int))
        np.testing.assert_allclose(distribute(exact_shapley(reduced), clusters, 3), [5/3]*3)


if __name__ == "__main__":
    unittest.main()
