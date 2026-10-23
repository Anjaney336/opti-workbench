from pathlib import Path
import unittest
import numpy as np
from research_harness import evaluate_step,train_weights
class ResearchChecks(unittest.TestCase):
 def test_no_trade_no_cost(self):
  w=np.ones(10)/10;n,d,t=evaluate_step(w,w,np.ones(10)*.01);self.assertAlmostEqual(n,.01);self.assertEqual(t,0);self.assertAlmostEqual(d.sum(),1)
 def test_turnover_cost(self):
  w=np.array([1.,0]);p=np.array([0.,1]);n,d,t=evaluate_step(w,p,np.zeros(2),fee=.01);self.assertAlmostEqual(n,-.02);self.assertEqual(t,2)
 def test_drift(self):
  n,d,t=evaluate_step(np.array([.5,.5]),np.array([.5,.5]),np.array([.1,-.1]));self.assertTrue(np.allclose(d,[.55,.45]))
 def test_cost_direction(self):
  w=np.array([.8,.2]);p=np.array([.2,.8]);r=np.array([.1,-.1]);self.assertLess(evaluate_step(w,p,r,.01)[0],evaluate_step(w,p,r,0)[0])
if __name__=='__main__':unittest.main()

