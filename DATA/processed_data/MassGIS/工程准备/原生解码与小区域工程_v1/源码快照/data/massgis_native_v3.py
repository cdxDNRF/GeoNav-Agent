"""Container-independent OpenCV read of the unchanged original JP2 codestream."""
from pathlib import Path
from io import BytesIO
import sys,zipfile
import numpy as np
from PIL import Image
import cv2
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from data import massgis_native_v1 as base
from data import massgis_native_v2 as prior

OUT=base.OUT/'原始码流适配_v3'
ORIGINAL_DECODE=cv2.imdecode

def decoder(buffer,flags):
    data=buffer.tobytes()
    if len(data)>12 and data[4:8]==b'jP  ':
        data=prior.codestream(data)
    return ORIGINAL_DECODE(np.frombuffer(data,np.uint8),flags)

def parser(data):
    result=prior.parse(data)
    result['independent_cv2_input']='unchanged bare JP2 codestream (container not interpreted)'
    result['metadata_interpreter']='massgis_native_v3 + v2 geographic/container metadata'
    return result

def prepare():
    if OUT.exists():raise ValueError('V3 batch exists, no overwrite')
    OUT.mkdir()
    base.write(prior.OUT/'容器解码拒绝.json',dict(passed=False,source='241894',
        reason='OpenCV OpenJPEG decodeGrayscaleData unsupported conversion from4 components to4; original JP2 enum17',
        source_pixels_quality_not_rejudged=True,new_requests=0))
    names=['project/src/data/massgis_native_v3.py','project/src/data/massgis_native_v2.py',
           'project/src/data/massgis_native_v1.py']
    base.write(OUT/'预登记.json',dict(source_sha256={p:base.digest(base.ROOT/p) for p in names},
        original_registration_sha256=base.digest(base.QA/'预登记.json'),
        raw_sha256={base.relative(p):base.digest(p) for p in (base.RAW/'zip').glob('*.zip')},
        invariant='Strip JP2 wrapper only, preserve original jp2c bytes; PIL container/PIL bare/OpenCV bare all4components exact',
        same_sources=True,same_quality_thresholds=True,new_requests=0))

def verify():
    reg=base.read(OUT/'预登记.json')
    for k,v in {**reg['source_sha256'],**reg['raw_sha256']}.items():
        if base.digest(base.ROOT/k)!=v:raise ValueError('V3 source/input drift')
    cv2.setNumThreads(1);rows=[]
    for s in base.read(base.META/'工程区选择.json')['selected']['sources']:
        p=base.RAW/'zip'/(s['sheet_id']+'.zip');meta=parser(p.read_bytes()[:65536])
        print('Bare codestream comparison',s['sheet_id'],flush=True)
        with zipfile.ZipFile(p) as z:jp2=z.read(meta['jp2_name'])
        raw=prior.codestream(jp2)
        with Image.open(BytesIO(raw)) as im:im.load();a=np.array(im)
        with Image.open(BytesIO(jp2)) as im:im.load();b=np.array(im)
        c=decoder(np.frombuffer(jp2,np.uint8),cv2.IMREAD_UNCHANGED)
        if a.shape!=(8000,8000,4) or b.shape!=a.shape or c is None or c.shape!=a.shape:
            raise ValueError('All4 native components required')
        if not np.array_equal(a,b) or not np.array_equal(a,c[:,:,[2,1,0,3]]):
            raise ValueError('Original components differ after container stripping')
        rows.append(dict(source=s['sheet_id'],declared_colour_enum=meta['declared_colour_enum'],
            all_four_components_exact=True,original_codestream_sha256=base.old.sha256(raw).hexdigest(),
            rgb_channels_not_all_identical=bool(np.any(a[:,:,0]!=a[:,:,1]) or np.any(a[:,:,1]!=a[:,:,2]))))
        del a,b,c,jp2,raw
    base.write(OUT/'原始分量逐位核验.json',dict(passed=True,checks=rows,colour_conversion=False,original_files_modified=False,new_SR=False))

def continue_engineering():
    if not base.read(OUT/'原始分量逐位核验.json')['passed']:raise ValueError('Native component equality not verified')
    for k,v in base.read(OUT/'预登记.json')['source_sha256'].items():
        if base.digest(base.ROOT/k)!=v:raise ValueError('V3 drift')
    base.metadata=parser
    base.cv2.imdecode=decoder  # Explicit process-local backend; frozen files unchanged.
    base.decode();base.build();base.interface()

def audit():
    if not base.read(OUT/'原始分量逐位核验.json')['passed']:raise ValueError('Component proof required')
    from eval import audit_massgis_native_v1 as check
    check.cv2.imdecode=decoder
    check.main()

if __name__=='__main__':
    {'prepare':prepare,'verify':verify,'continue':continue_engineering,'audit':audit}[sys.argv[1]]()
