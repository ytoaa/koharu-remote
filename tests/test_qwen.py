"""Qwen transport, masked compositing, geometry and pinned Turbo call contracts.

Offline tests exercise the real handlers/backend with only the CUDA generator
and neural pipeline injected. They do not claim model quality or GPU compatibility.
"""
import copy
import hashlib
import io
import json
import os
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
import pytest
from PIL import Image, ImageDraw
from fastapi.testclient import TestClient
from qwen import backend as real
from qwen.app import create_app
from qwen.options import Options, MODEL
from qwen.model_files import REVISION, SAMPLE_SIGMAS
from qwen.healthcheck import require_ready


def uploads(source, mask):
    def png(image):
        out=io.BytesIO();image.save(out,'PNG');return out.getvalue()
    return {'image':('source.png',png(source),'image/png'),
            'mask':('mask.png',png(mask),'image/png')}


@pytest.fixture
def backend(monkeypatch):
    monkeypatch.delenv('ROREM_AUTH_REQUIRED',raising=False)
    seeds=[];calls=[]
    class Generator:
        def __init__(self,device): assert device=='cuda'
        def manual_seed(self,seed): seeds.append(seed);return self
    class Pipeline:
        config=SimpleNamespace(sample_sigmas=SAMPLE_SIGMAS)
        # Actual pinned QwenImage21Pipeline arguments. Unsupported RORem/legacy
        # Qwen arguments raise TypeError, rather than being silently swallowed.
        def __call__(self,*,prompt,image,width,height,output_resolution,
                     true_cfg_scale,use_kv_cache,generator):
            calls.append(dict(prompt=prompt,image=image,width=width,height=height,
                              output_resolution=output_resolution,true_cfg_scale=true_cfg_scale,
                              use_kv_cache=use_kv_cache))
            return SimpleNamespace(images=[Image.new('RGB',(width,height),'red')])
    b=real.QwenBackend.__new__(real.QwenBackend)
    b.model_loaded=True;b.device='cuda';b.gpu_name='test GPU'
    b.details={'warmup_complete':True,'model_revision':REVISION}
    b.pipe=Pipeline();b.torch=SimpleNamespace(Generator=Generator,inference_mode=nullcontext)
    return b,calls,seeds


def test_http_uses_reference_mask_saved_schedule_and_preserves_outside_mask(backend):
    b,calls,seeds=backend
    source=Image.new('RGB',(520,310),'white')
    mask=Image.new('L',source.size,0);ImageDraw.Draw(mask).rectangle((100,100,110,110),fill=255)
    with TestClient(create_app(lambda:b)) as client:
        health=client.get('/health').json();require_ready(health)
        assert health['service']=='qwen-inpaint' and health['model']==MODEL
        response=client.post('/v1/inpaint',files=uploads(source,mask),data={'options':json.dumps({
            'max_side':512,'mask_dilation':2,'seed':7,'use_kv_cache':False,'prompt':'Keep thin lines.'})})
    assert response.status_code==200 and response.headers['x-model']==MODEL
    result=Image.open(io.BytesIO(response.content))
    assert result.size==source.size
    assert result.getpixel((105,105))==(255,0,0)
    assert result.getpixel((99,105))==(255,255,255)  # dilation must not widen the final mask
    call=calls[0]
    assert call['width']==call['height']==call['output_resolution']==512
    assert [image.mode for image in call['image']]==['RGBA','RGBA']
    assert call['image'][1].getpixel((0,0))==(0,0,0,255)
    assert call['true_cfg_scale']==1.0 and call['use_kv_cache'] is False
    assert 'Image 2 is a binary selection mask' in call['prompt'] and 'Keep thin lines.' in call['prompt']
    assert seeds==[7]


def test_random_seed_fixed_zero_and_option_isolation(backend,monkeypatch):
    b,calls,seeds=backend
    draws=iter([41,42]);monkeypatch.setattr(real.secrets,'randbits',lambda n:next(draws) if n==31 else -999)
    image=Image.new('RGB',(8,8),'white');mask=Image.new('L',image.size,255)
    with TestClient(create_app(lambda:b)) as client:
        for options in ({'seed':0},{'seed':-1,'use_kv_cache':False},{'seed':-1},{}):
            assert client.post('/v1/inpaint',files=uploads(image,mask),data={'options':json.dumps(options)}).status_code==200
    assert seeds==[0,41,42,0]
    assert [c['use_kv_cache'] for c in calls]==[True,False,True,True]


@pytest.mark.parametrize('options',[{'max_side':4097},{'max_side':255},{'seed':-2},{'seed':True},
    {'mask_dilation':33},{'use_kv_cache':1},{'steps':4},{'strength':0.9},
    {'negative_prompt':'text'},{'guidance_scale':2},{'model':'Qwen-Image-Edit-2511'}])
def test_incompatible_options_rejected_before_cuda(backend,options):
    b,calls,seeds=backend
    image=Image.new('RGB',(8,8),'white');mask=Image.new('L',image.size,255)
    with TestClient(create_app(lambda:b)) as client:
        assert client.post('/v1/inpaint',files=uploads(image,mask),data={'options':json.dumps(options)}).status_code==422
    assert not calls and not seeds


@pytest.mark.parametrize('limit',[256,513,1024,2048,4096])
def test_geometry_has_aligned_canvas_bounded_long_side_and_surviving_thin_mask(limit):
    source=Image.new('RGB',(5100,330),'white')
    mask=Image.new('L',source.size,0);mask.putpixel((123,12),255)
    image,reference,size=real.prepare(source,mask,limit,0)
    assert image.size==reference.size
    assert image.width==image.height<=limit and image.width%32==0
    assert reference.convert('L').getbbox() is not None
    assert max(size)<=limit


def test_qwen_4096_is_accepted_and_rorrem_rollback_defaults_remain():
    from rorem.options import Options as RoremOptions
    assert Options(max_side=4096).max_side==4096 and Options().max_side==2048
    assert RoremOptions().max_side==1024


def test_preload_failure_stays_unready_and_exposes_no_success(backend):
    def fail(): raise RuntimeError('model failed')
    with TestClient(create_app(fail)) as client:
        health=client.get('/health')
        assert health.status_code==503 and not health.json()['ready']
        with pytest.raises(RuntimeError): require_ready(health.json())


def test_qwen_startup_loads_once_warms_all_saved_steps_and_never_reloads(monkeypatch,backend,tmp_path):
    import sys
    b,calls,seeds=backend
    events=[]
    class Pipeline(type(b.pipe)):
        @classmethod
        def from_pretrained(cls,path,**kwargs):
            events.append(('load',path,kwargs));return cls()
        def to(self,device): events.append(('to',device));return self
        def set_progress_bar_config(self,**kwargs): pass
    torch=SimpleNamespace(Generator=b.torch.Generator,inference_mode=nullcontext,
        bfloat16='bf16',__version__='test',version=SimpleNamespace(cuda='test'),
        cuda=SimpleNamespace(is_available=lambda:True,get_device_name=lambda:'test GPU'))
    monkeypatch.setitem(sys.modules,'torch',torch)
    monkeypatch.setitem(sys.modules,'diffusers',SimpleNamespace(QwenImage21Pipeline=Pipeline,__version__='test'))
    monkeypatch.setattr(real,'validate_models',lambda p:{'path':p})
    monkeypatch.setenv('QWEN_MODEL_DIR',str(tmp_path))
    loaded=real.QwenBackend()
    assert loaded.model_loaded and loaded.details['warmup_complete']
    assert events==[('load',str(tmp_path),{'dtype':'bf16','local_files_only':True}),('to','cuda')]
    loaded.infer(Image.new('RGB',(8,8),'white'),Image.new('L',(8,8),255),Options())
    assert len([e for e in events if e[0]=='load'])==1 and len(calls)==2
    Pipeline.config=SimpleNamespace(sample_sigmas=[1.0,0.5])
    with pytest.raises(RuntimeError,match='sampling schedule'): real.QwenBackend()


def test_manifest_detects_truncated_and_tampered_checkpoint(tmp_path,monkeypatch):
    from qwen import model_files as module
    contents={'model_index.json':json.dumps({'_class_name':'QwenImage21Pipeline','sample_sigmas':SAMPLE_SIGMAS}).encode(),
              'weights.safetensors':b'test-only-weight-content'}
    manifest={'files':{name:{'size':len(data),'sha256':hashlib.sha256(data).hexdigest()} for name,data in contents.items()}}
    monkeypatch.setattr(module,'MANIFEST',manifest)
    for name,data in contents.items(): (tmp_path/name).write_bytes(data)
    assert module.validate_models(tmp_path)['revision']==REVISION
    (tmp_path/'weights.safetensors').write_bytes(b'test-only-weight-contEnt')
    with pytest.raises(ValueError): module.validate_models(tmp_path)


def test_downloaded_real_pinned_metadata_matches_manifest():
    from qwen.model_files import MANIFEST
    folder=os.environ.get('QWEN_METADATA_FIXTURE')
    if not folder: pytest.skip('Optional pinned upstream metadata not supplied')
    for name in ['model_index.json','transformer/config.json','vae/config.json',
                 'text_encoder/config.json','processor/processor_config.json','scheduler/scheduler_config.json']:
        data=(Path(folder)/name).read_bytes();expected=MANIFEST['files'][name]
        assert len(data)==expected['size']
        assert hashlib.sha1(f'blob {len(data)}\0'.encode()+data).hexdigest()==expected['git_blob_sha1']
