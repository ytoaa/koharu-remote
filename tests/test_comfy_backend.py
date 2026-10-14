import json
from io import BytesIO
from PIL import Image
import numpy as np
import pytest
from qwen.comfy_backend import ComfyBackend, DEFAULT_MODELS, build_graph, encode_upload
from qwen.options import Options
from rorem.strategy import preserve_outside_mask

def test_actual_sampling_contract():
    g=build_graph('crop.png',Options(),DEFAULT_MODELS,7,0)
    assert g['5']['inputs']['images.image_1']==['1',0]
    assert 'images.image_2' not in g['5']['inputs']
    assert g['3']['inputs']['type']=='qwen_image'
    assert g['9']['inputs']['latent_image']==['5',2]
    assert g['9']['inputs']['noise_seed']==7
    assert [float(v) for v in g['8']['inputs']['sigmas'].split(',')]==[1.0,0.978453,0.95418,0.926626,0.89508,0.845148,0.704534,0.414568,0.0]
    assert g['6']['inputs']['device']=='auto'
    assert build_graph('crop.png',Options(use_kv_cache=False),DEFAULT_MODELS,7,0)['6']['inputs']['device']=='off'

class FakeComfy(ComfyBackend):
    def __init__(self):
        self.model_loaded=False;self.selected=dict(DEFAULT_MODELS);self.details={'warmup_complete':False}
        self.name='Qwen-Image-2.1-Turbo';self.device='cuda';self.gpu_name='Fake GPU';self.timeout=30;self.graph=None;self.runs=0
        self.configure(self.selected,force=True)
    def json(self,path,data=None):
        if path=='/object_info':return {kind:{'input':{'required':{field:[[value,'alternative.safetensors']]}}} for kind,field,value in [('UNETLoader','unet_name',DEFAULT_MODELS['model']),('CLIPLoader','clip_name',DEFAULT_MODELS['text_encoder']),('VAELoader','vae_name',DEFAULT_MODELS['vae'])]}
        if path=='/prompt':self.graph=data['prompt'];self.runs+=1;return {'prompt_id':'our-job'}
        if path=='/history/our-job':return {'our-job':{'status':{'status_str':'success'},'outputs':{'11':{'images':[{'filename':'output.png','subfolder':'','type':'output'}]}}}}
        raise AssertionError(path)
    def request(self,path,data=None,content_type=None):
        if path=='/upload/image':return json.dumps({'name':'unique.png','subfolder':''}).encode()
        if path.startswith('/view?'):
            b=BytesIO();Image.new('RGB',(320,192),'red').save(b,format='PNG');return b.getvalue()
        raise AssertionError(path)

def test_preload_select_infer_and_exact_mask_composite():
    b=FakeComfy();assert b.model_loaded and b.details['warmup_complete'];assert b.runs==1
    b.configure({'model':DEFAULT_MODELS['model']});assert b.runs==1
    b.configure({'model':'alternative.safetensors'});assert b.runs==2 and b.selected['model']=='alternative.safetensors'
    source=Image.new('RGB',(363,466),'white');mask=Image.new('L',source.size,0);mask.putpixel((50,60),255)
    out=b.infer(source,mask,Options(model='alternative.safetensors',max_side=320))
    assert out.size==source.size
    merged=preserve_outside_mask(out,source,mask)
    a=np.asarray(merged);assert tuple(a[60,50])==(255,0,0);assert np.all(a[0]==255)
    with pytest.raises(ValueError,match='Select and preload'):b.infer(source,mask,Options(model='not-selected'))
    with pytest.raises(ValueError,match='not installed'):b.configure({'model':'missing.safetensors'})
    assert b.selected['model']=='alternative.safetensors'

def test_upload_and_no_reference_mask_dilation():
    raw,typ=encode_upload(Image.new('RGB',(32,32)))
    assert 'multipart/form-data' in typ and b'name="image"' in raw and b'\x89PNG' in raw
    b=FakeComfy()
    with pytest.raises(ValueError,match='dilation'):b.infer(Image.new('RGB',(32,32)),Image.new('L',(32,32)),Options(mask_dilation=1))

def test_existing_api_health_selection_and_masked_image(monkeypatch):
    from fastapi.testclient import TestClient
    from qwen.app import create_app
    from qwen.healthcheck import require_ready
    monkeypatch.setenv('QWEN_BACKEND','comfyui')
    b=FakeComfy()
    with TestClient(create_app(lambda:b)) as client:
        h=client.get('/health').json();require_ready(h)
        assert h['capabilities']['configurable_models'] is True
        response=client.post('/v1/models/select',json={'model':'alternative.safetensors'})
        assert response.status_code==200 and response.json()['ready'] is True
        assert response.json()['models']['model']=='alternative.safetensors'
        assert client.post('/v1/models/select',json={'model':'missing.safetensors'}).status_code==422
        assert client.post('/v1/models/select',json={'model':'alternative.safetensors'}).status_code==200
        def png(im):
            stream=BytesIO();im.save(stream,format='PNG');return stream.getvalue()
        source=Image.new('RGB',(64,64),'white');mask=Image.new('L',source.size,0);mask.putpixel((10,10),255)
        response=client.post('/v1/inpaint',files={'image':('crop.png',png(source),'image/png'),'mask':('mask.png',png(mask),'image/png')},data={'options':json.dumps({'model':'alternative.safetensors'})})
        assert response.status_code==200
        im=Image.open(BytesIO(response.content));assert im.size==source.size
        assert im.getpixel((10,10))==(255,0,0) and im.getpixel((0,0))==(255,255,255)
