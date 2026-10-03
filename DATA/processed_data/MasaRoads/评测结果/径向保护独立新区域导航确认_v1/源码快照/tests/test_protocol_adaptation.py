import sys,unittest
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from io import BytesIO
from types import SimpleNamespace
import numpy as np
from PIL import Image
from agents.scaled_edge_navigator import scaled_policy_features
from train.protocol_adaptation import ProtocolWorld
from train.navigation_spatial import NavigationWorld
from data.protocol_fit_cache import virtual_patch
from data.scaled_masa import scaled_crop
from data.process_masa import preprocess_patch,MEAN,STD


class ProtocolTests(unittest.TestCase):
    def data(self,k):return {'fit':np.arange(k*k*512,dtype=np.float32).reshape(k*k,512)/8192,'held':np.ones((k*k,512),np.float32)}

    def test_five_protocol_matches_original_world_inputs_and_rewards(self):
        table=self.data(5);mean=np.arange(512,dtype=np.float32)/512
        old=NavigationWorld(SimpleNamespace(data=table),{'fit':np.zeros((25,4,768),np.float32)},['fit'],mean,np.zeros((4,768),np.float32),'Small256','NoTarget',4,73)
        new=ProtocolWorld(table,['fit'],mean,5,4,73)
        for step in range(30):
            np.testing.assert_array_equal(new.observe(),old.observe());np.testing.assert_array_equal(new.goal,old.goal)
            actions=np.array([step%4,(step+1)%4,(step+2)%4,(step+3)%4]);a,b=new.step(actions),old.step(actions)
            for key in ('done','success','wall','revisited','external','pbrs'):np.testing.assert_array_equal(a[key],b[key])
            new.reset(a['done']);old.reset(b['done'])

    def test_ten_inputs_match_public_deployment_mapping(self):
        table=self.data(10);mean=np.zeros(512,np.float32);w=ProtocolWorld(table,['fit'],mean,10,1,4)
        w.position[0]=36;w.steps[0]=4;w.visits[0]=0;visited=[24,25,35,35,36]
        for cell in visited:w.visits[0,cell]+=1
        x=scaled_policy_features(mean,table['fit'][36],(3,6),16,visited,10,20)
        np.testing.assert_array_equal(w.observe()[0],x)
        self.assertEqual(w.observe().shape,(1,1052))

    def test_true_goal_never_changes_public_no_target_input(self):
        w=ProtocolWorld(self.data(10),['fit'],np.ones(512,np.float32),10,1,5);before=w.observe().copy();w.goal[0]=(w.goal[0]+1)%100
        np.testing.assert_array_equal(before,w.observe());self.assertEqual(w.areas,['fit']);self.assertEqual(len(w.table),1)

    def test_budget_terminal_and_first_arrival(self):
        w=ProtocolWorld(self.data(10),['fit'],np.zeros(512,np.float32),10,1,5)
        w.position[0]=0;w.goal[0]=99;w.steps[0]=19;w.visits[0]=0;w.visits[0,0]=1
        info=w.step(np.array([1]));self.assertTrue(info['done'][0]);self.assertFalse(info['success'][0]);self.assertEqual(w.steps[0],20)
        w.position[0]=98;w.goal[0]=99;w.steps[0]=0;info=w.step(np.array([1]));self.assertTrue(info['success'][0]);self.assertTrue(info['done'][0])

    def test_pbrs_grid_diameter_and_terminal_zero(self):
        for k in (5,10):
            w=ProtocolWorld(self.data(k),['fit'],np.zeros(512,np.float32),k,1,7);w.position[0]=0;w.goal[0]=k*k-1;w.steps[0]=0
            result=w.step(np.array([1]));self.assertAlmostEqual(float(result['pbrs'][0]),.5*(.99*-(2*(k-1)-1)/(2*(k-1))+1),places=6)

    def test_virtual_jpeg_is_identical_to_s4_crop_bytes(self):
        rng=np.random.default_rng(3);source=Image.fromarray(rng.integers(0,256,(1500,1500,3),dtype=np.uint8))
        tensor,jpeg_sha,rgb_sha=virtual_patch(source,37);stream=BytesIO();scaled_crop(source,37).save(stream,format='JPEG',quality=75)
        import hashlib
        self.assertEqual(jpeg_sha,hashlib.sha256(stream.getvalue()).hexdigest())
        with Image.open(BytesIO(stream.getvalue())) as image:
            self.assertEqual(rgb_sha,hashlib.sha256(np.asarray(image.convert('RGB'),np.uint8).tobytes()).hexdigest())
            arr=np.asarray(image.resize((224,224),Image.Resampling.BICUBIC),np.float32)/255
        np.testing.assert_array_equal(tensor.numpy(),((arr-MEAN)/STD).transpose(2,0,1))


if __name__=='__main__':unittest.main()
