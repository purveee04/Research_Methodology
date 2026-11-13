import tempfile,unittest
import numpy as np
from member3_m32 import spherical_vectors,footprint_pairs,components,minimally_repair,allocate_labelled,validate_split

class SplitTests(unittest.TestCase):
 def test_ra_wrap(self):
  p=footprint_pairs([359.999,0.001,90],[0,0,0],[0.262]*3,256)
  self.assertEqual([(a,b) for a,b,*_ in p],[(0,1)])
 def test_non_overlap(self):
  self.assertEqual(footprint_pairs([0,2,4],[0,0,0],[0.262]*3,256),[])
 def test_connected_groups(self):
  gid=components(5,[(0,1,1.,2.),(1,2,1.,2.),(3,4,1.,2.)])
  self.assertEqual(gid.tolist(),[0,0,0,3,3])
 def test_partition_repair(self):
  prior={'pool':np.array([0,2]),'val':np.array([1,3]),'test':np.array([4]),'lab':np.array([0]),'unl_pool':np.array([2])}
  labels=np.array([0,0,0,0,0]); gid=np.array([0,0,2,3,4])
  a=minimally_repair(prior,gid)
  self.assertEqual(a['assignment'][0],a['assignment'][1])
  a['lab'],a['unl_pool']=allocate_labelled(prior,a['pool'],labels)
  v=validate_split(a,gid,[(0,1,0.,100.)],labels)
  self.assertEqual(v['status'],'PASS_CONSERVATIVE_GEOMETRIC_GUARD')
 def test_deterministic(self):
  prior={'pool':np.array([0,2,5]),'val':np.array([1,3]),'test':np.array([4]),'lab':np.array([0]),'unl_pool':np.array([2,5])}
  g=np.array([0,0,2,3,4,5]);labels=np.array([0,0,0,0,0,0])
  r1=minimally_repair(prior,g);r2=minimally_repair(prior,g)
  for name in ['pool','val','test']:
   np.testing.assert_array_equal(r1[name],r2[name])
  l1,u1=allocate_labelled(prior,r1['pool'],labels,seed=16)
  l2,u2=allocate_labelled(prior,r2['pool'],labels,seed=16)
  np.testing.assert_array_equal(l1,l2);np.testing.assert_array_equal(u1,u2)
 def test_hash_guard(self):
  prior={'pool':np.array([0,2]),'val':np.array([1]),'test':np.array([3]),'lab':np.array([0]),'unl_pool':np.array([2])}
  g=np.array([0,1,2,3]);labels=np.array([0,0,0,0]);r=minimally_repair(prior,g)
  r['lab']=np.array([0]);r['unl_pool']=np.array([2])
  with self.assertRaises(AssertionError):validate_split(r,g,[],labels,expected_hash='a',actual_hash='b')
if __name__=='__main__':unittest.main()
