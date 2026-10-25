"""Selected-source access and independent center-coordinate boundaries."""
import copy
import unittest
from PIL import Image
from tools.capturepoint_label_review_v1 import select,center_from_stitch


class ReviewBoundaries(unittest.TestCase):
    def fixture(self):
        ss=[];rr=[]
        for i in range(14):
            sid=f's{i}';group=f'g{i}';role='fit' if i<9 else 'development'
            tiles=[{'path':f'DATA/raw_data/MM-GAG/{i}/sat_{j//9+1}_{j%9+1}.jpg'} for j in range(81)]
            ss.append({'source_id':sid,'group':group,'role':role,'ground':{'path':f'g{i}'},'tiles':tiles,'stitch':{'path':f'm{i}'}})
            rr.append({'source_id':sid,'group':group,'role':'fit_preparation' if i<9 else 'development','representative':True})
        return ss,rr

    def test_only_14_representatives_42_files(self):
        s,r=self.fixture();self.assertEqual(len(select(s,r)),14)
        for change in ('reserved','not_rep','group'):
            b=copy.deepcopy(r)
            if change=='reserved':b[0]['role']='confirmation_reserved'
            elif change=='not_rep':b[0]['representative']=False
            else:b[0]['group']='wrong'
            with self.assertRaises(ValueError):select(s,b)

    def test_proxy_and_duplicate_source_fail(self):
        s,r=self.fixture();s[0]['tiles'][40]['path']='wrong.jpg'
        with self.assertRaises(ValueError):select(s,r)
        s,r=self.fixture();s[1]=s[0]
        with self.assertRaises(ValueError):select(s,r)

    def test_center_is_cell5_5_in_11_grid(self):
        im=Image.new('RGB',(2816,2816));im.paste((254,7,11),(1280,1280,1536,1536))
        crop=center_from_stitch(im);self.assertEqual(crop.size,(256,256));self.assertEqual(crop.getpixel((255,255)),(254,7,11))
        with self.assertRaises(ValueError):center_from_stitch(Image.new('RGB',(2560,2560)))


if __name__=='__main__':unittest.main()
