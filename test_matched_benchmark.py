import unittest
import numpy as np
from matched_benchmark import cvar,solve

class MatchedTests(unittest.TestCase):
    def setUp(self):self.r=np.random.default_rng(4).normal(0,.02,(60,10))
    def test_solver_agreement(self):self.assertAlmostEqual(cvar(self.r,solve(self.r)),cvar(self.r,solve(self.r,'CVXPY')),places=6)
    def test_feasibility(self):
        w=solve(self.r);self.assertAlmostEqual(w.sum(),1);self.assertTrue(w.min()>=-1e-7);self.assertTrue(w.max()<=.300001)
    def test_cvar_hand_calculation(self):self.assertAlmostEqual(cvar(np.array([[-.1],[-.2],[-.3],[-.4]]),[1],.5),.35)
    def test_infeasible_cap(self):
        with self.assertRaises(ValueError):solve(self.r,cap=.01)
    def test_reproducibility(self):np.testing.assert_allclose(solve(self.r),solve(self.r))
    def test_not_worse_than_feasible_equal(self):self.assertLessEqual(cvar(self.r,solve(self.r)),cvar(self.r,np.ones(10)/10)+1e-8)

if __name__=='__main__':unittest.main()
