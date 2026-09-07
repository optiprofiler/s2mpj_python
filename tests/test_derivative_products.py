"""Check constraint products against explicit Jacobians and differences."""

from pathlib import Path
import sys
import unittest

import numpy as np
from scipy.sparse import csr_matrix

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from s2mpjlib import CUTEst_problem


class GroupedConstraints(CUTEst_problem):
    """c_i(x) = (a_i @ x - b_i)^2 / scale_i, using native group assembly."""
    def __init__(self, count=1):
        self.name = 'GROUPED_CONSTRAINT_FIXTURE'
        self.n, self.m = 2, count
        self.congrps = np.arange(count)
        self.conderlvl = np.full(count, 2)
        self.A = csr_matrix(np.array([[1., 2.], [-2., 1.]])[:count])
        self.gconst = np.array([0.5, -1.0])[:count]
        self.gscale = np.array([2., 3.])[:count]
        self.grftype = ['square'] * count

    @staticmethod
    def square(self, nargout, value, group):
        return (value ** 2, 2 * value, 2.0)[:nargout]


class DerivativeProductTests(unittest.TestCase):
    def test_group_jacobian_vector_product(self):
        p = GroupedConstraints()
        x, v = np.array([0.7, -0.2]), np.array([0.3, -0.8])
        h = 1e-6
        _, jac = p.cJx(x)
        actual = p.cJxv(x, v).ravel()
        np.testing.assert_allclose(actual, np.asarray(jac @ v).ravel())
        difference = (p.cx(x + h * v) - p.cx(x - h * v)).ravel() / (2 * h)
        np.testing.assert_allclose(actual, difference, rtol=1e-8, atol=1e-9)

    def test_group_transpose_jacobian_vector_product(self):
        p = GroupedConstraints()
        x, weights = np.array([0.7, -0.2]), np.array([1.3])
        _, jac = p.cJx(x)
        actual = p.cJtxv(x, weights).ravel()
        np.testing.assert_allclose(actual, np.asarray(jac.T @ weights).ravel())
        h = 1e-6
        difference = np.array([
            float(weights @ (p.cx(x + h * e) - p.cx(x - h * e)).ravel()) / (2 * h)
            for e in np.eye(2)
        ])
        np.testing.assert_allclose(actual, difference, rtol=1e-8, atol=1e-9)

    def test_multigroup_transpose_uses_each_constraint_weight(self):
        p = GroupedConstraints(count=2)
        x, weights = np.array([0.7, -0.2]), np.array([1.3, -0.4])
        _, jac = p.cJx(x)
        np.testing.assert_allclose(p.cJtxv(x, weights).ravel(), np.asarray(jac.T @ weights).ravel())
        h = 1e-6
        difference = np.array([
            float(weights @ (p.cx(x + h * e) - p.cx(x - h * e)).ravel()) / (2 * h)
            for e in np.eye(2)
        ])
        np.testing.assert_allclose(p.cJtxv(x, weights).ravel(), difference, rtol=1e-8, atol=1e-9)

    def test_selected_constraints_transpose_product(self):
        p = GroupedConstraints(count=2)
        x = np.array([0.7, -0.2])
        for indices, weights in [([1], np.array([1.3])), ([1, 0], np.array([1.3, -0.4]))]:
            with self.subTest(indices=indices):
                _, jac = p.cIJx(x, indices)
                actual = p.cIJtxv(x, weights, indices).ravel()
                np.testing.assert_allclose(actual, np.asarray(jac.T @ weights).ravel())

    def test_native_problem_products(self):
        import importlib
        sys.path.insert(0, str(ROOT / 'src' / 'python_problems'))
        for name in ('HS65', 'HS71'):
            with self.subTest(problem=name):
                p = getattr(importlib.import_module(name), name)()
                x = np.asarray(p.x0).ravel()
                _, jac = p.cJx(x)
                v = np.linspace(0.2, 0.8, p.n)
                weights = np.linspace(-0.7, 1.3, len(p.congrps))
                np.testing.assert_allclose(p.cJxv(x, v).ravel(), np.asarray(jac @ v).ravel())
                actual = p.cJtxv(x, weights).ravel()
                np.testing.assert_allclose(actual, np.asarray(jac.T @ weights).ravel(), atol=1e-12)
                h = 1e-6
                difference = np.array([
                    float(weights @ (p.cx(x + h * e) - p.cx(x - h * e)).ravel()) / (2 * h)
                    for e in np.eye(p.n)
                ])
                np.testing.assert_allclose(actual, difference, rtol=1e-7, atol=1e-7)


if __name__ == '__main__':
    unittest.main()
