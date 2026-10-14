"""Contract tests use a fake Comfy server; no GPU or model downloads."""
import asyncio, importlib.util, json, sys, types
from io import BytesIO
from pathlib import Path
import numpy as np
import pytest
from PIL import Image
ROOT = Path(__file__).resolve().parents[1]

@pytest.fixture
def extension(monkeypatch, tmp_path):
    package = types.ModuleType('test_koharu_nodes')
    package.__path__ = [str(ROOT / 'comfyui/koharu_nodes')]
    monkeypatch.setitem(sys.modules, package.__name__, package)
    class Routes:
        def get(self, path): return lambda f: f
        def post(self, path): return lambda f: f
    server = types.ModuleType('server')
    server.PromptServer = types.SimpleNamespace(instance=types.SimpleNamespace(routes=Routes()))
    monkeypatch.setitem(sys.modules, 'server', server)
    monkeypatch.setitem(sys.modules, 'torch', types.SimpleNamespace(from_numpy=lambda value:value))
    monkeypatch.setitem(sys.modules, 'nodes', types.SimpleNamespace(NODE_CLASS_MAPPINGS={}))
    loaded = []
    for name in ('runtime','nodes','routes'):
        full = package.__name__ + '.' + name
        spec = importlib.util.spec_from_file_location(full, ROOT / f'comfyui/koharu_nodes/{name}.py')
        module = importlib.util.module_from_spec(spec)
        monkeypatch.setitem(sys.modules, full, module)
        spec.loader.exec_module(module)
        loaded.append(module)
    monkeypatch.setenv('KOHARU_COMFY_CONFIG', str(tmp_path / 'config.json'))
    monkeypatch.setenv('KOHARU_WORKFLOW_DIR', str(tmp_path / 'profiles'))
    return loaded

class Tensor:
    shape = (1, 3, 4, 3)
    def __getitem__(self, _): return self
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return np.ones((3,4,3), dtype=np.float32)

class ArrayTensor:
    def __init__(self, array):
        self.array = np.asarray(array, dtype=np.float32)
        self.shape = self.array.shape
        self.device = 'cpu'
        self.dtype = np.float32
    def __getitem__(self, key): return ArrayTensor(self.array[key])
    def detach(self): return self
    def cpu(self): return self
    def numpy(self): return self.array


def test_final_rgb_is_delivered_without_hidden_compositing_or_resize(extension):
    runtime,nodes,_=extension
    job=runtime.Job(Image.new('RGB',(4,3),(12,34,56)),Image.new('L',(4,3)),{})
    runtime.jobs['a']=job
    final=ArrayTensor(np.full((1,3,4,3),0.4,np.float32))
    result=nodes.KoharuInpaintOutput().finish('a',final)
    output=Image.open(BytesIO(job.result))
    assert output.mode=='RGB' and output.size==(4,3)
    assert output.getpixel((2,1))==(102,102,102)
    assert result['result'][0] is final and job.finished.is_set()


def test_output_rejects_implicit_resize(extension):
    runtime,nodes,_=extension
    job=runtime.Job(Image.new('RGB',(93,67)),Image.new('L',(93,67)),{})
    runtime.jobs['a']=job
    with pytest.raises(ValueError,match='request dimensions'):
        nodes.KoharuInpaintOutput().finish('a',Tensor())
    assert job.error and job.finished.is_set()


def test_fill_holes_preserves_border_connected_regions_and_large_holes(extension, monkeypatch):
    _, nodes, _ = extension
    class Result(ArrayTensor):
        def to(self, **kwargs): return self
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(from_numpy=Result))
    mask = np.ones((1,10,10),np.float32)
    mask[0,2,2]=0
    mask[0,4:7,4:7]=0  # Nine pixels exceeds configured hole cap.
    mask[0,0:3,8]=0  # Connected to exterior, never filled.
    output = nodes.KoharuMaskFillHoles().fill(ArrayTensor(mask),max_hole_area=4)[0].numpy()
    assert output[0,2,2]==1
    assert np.all(output[0,4:7,4:7]==0)
    assert np.all(output[0,0:3,8]==0)
    tensor=ArrayTensor(mask)
    assert nodes.KoharuMaskFillHoles().fill(tensor,enabled=False)[0] is tensor


def test_local_color_node_returns_aligned_image_and_rejects_shape_mismatch(extension,monkeypatch):
    _,nodes,_=extension
    class Result(ArrayTensor):
        def to(self,**kwargs):return self
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(from_numpy=Result))
    node=nodes.NODE_CLASS_MAPPINGS['KoharuLocalColorMatch']()
    target=ArrayTensor(np.full((1,16,16,3),0.48,np.float32))
    reference=ArrayTensor(np.full((1,16,16,3),0.5,np.float32))
    mask=ArrayTensor(np.zeros((1,16,16),np.float32))
    output=node.match(target,reference,mask)[0]
    assert output.shape==target.shape and np.allclose(output.numpy(),0.5)
    with pytest.raises(ValueError,match='Align'):
        node.match(target,reference,ArrayTensor(np.zeros((1,8,8),np.float32)))


def test_graph_has_owned_terminal_and_exact_turbo_sigmas(extension):
    runtime, _, routes = extension
    from dgx.qwen.options import Options
    job = runtime.Job(Image.new('RGB',(256,256)),options=Options(seed=17))
    graph = routes.graph('inpainting','owned',job,runtime.config())
    assert graph['1']['class_type'] == 'KoharuInput'
    assert graph['11']['inputs']['request_id'] == 'owned'
    assert graph['9']['inputs']['noise_seed'] == 0  # Saved workflow owns seed by default
    assert len(graph['8']['inputs']['sigmas'].split(',')) == 9
    assert not any(n['class_type'] == 'SaveImage' for n in graph.values())

@pytest.mark.parametrize('value',[{'text':float('nan')},{'panel':True},{'other':0.5},{'text':1.1}])
def test_invalid_detection_options(extension,value):
    with pytest.raises(ValueError): extension[2].options('detection',value)

def test_model_mismatch_rejected(extension):
    with pytest.raises(ValueError): extension[2].options('inpainting',{'model':'missing.safetensors'})

def test_health_not_ready_before_warmup(extension):
    response = asyncio.run(extension[2].vision_health(types.SimpleNamespace(headers={})))
    assert not json.loads(response.text)['ready']

def test_bounded_queue(extension):
    runtime,_,routes=extension
    runtime.jobs.update({str(i):object() for i in range(4)})
    from aiohttp import web
    with pytest.raises(web.HTTPTooManyRequests): asyncio.run(routes.execute('ocr',runtime.Job(None)))

def test_timeout_does_not_cancel_owned_task(extension,monkeypatch):
    _,_,routes=extension
    finished=[]
    async def execute(stage,job):
        await asyncio.sleep(.025)
        finished.append(stage)
        return {'text':'done'}
    monkeypatch.setattr(routes,'execute',execute)
    monkeypatch.setattr(routes,'config',lambda:{'timeout':.001})
    async def scenario():
        from aiohttp import web
        with pytest.raises(web.HTTPGatewayTimeout): await routes.submit('ocr',None)
        assert routes.active_tasks
        await asyncio.gather(*routes.active_tasks)
        assert finished == ['ocr']
        assert not routes.active_tasks
    asyncio.run(scenario())

def test_model_config_change_invalidates_readiness(extension,monkeypatch,tmp_path):
    runtime,_,_=extension
    runtime.ready.add('inpainting')
    runtime.loaded_models.update(runtime.config()['models'])
    assert runtime.is_ready('inpainting')
    path=tmp_path/'changed.json'
    path.write_text(json.dumps({'models':{'model':'different-turbo.safetensors'}}))
    monkeypatch.setenv('KOHARU_COMFY_CONFIG',str(path))
    assert not runtime.is_ready('inpainting')

def test_queue_validation_failure_releases_owned_job(extension,monkeypatch):
    runtime,_,routes=extension
    class Response:
        status=400
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def json(self): return {'error':'invalid prompt'}
    class Session:
        def __init__(self,**kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        def post(self,*args,**kwargs): return Response()
    monkeypatch.setattr(routes.aiohttp,'ClientSession',Session)
    job=runtime.Job(Image.new('RGB',(32,32)),options={})
    with pytest.raises(RuntimeError,match='validation'): asyncio.run(routes.execute('detection',job))
    assert not runtime.jobs

def test_queue_completion_uses_only_owned_history(extension,monkeypatch):
    runtime,_,routes=extension
    seen=[]
    job=runtime.Job(Image.new('RGB',(32,32)),options={})
    class Response:
        status=200
        def __init__(self,body): self.body=body
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def json(self): return self.body
    class Session:
        def __init__(self,**kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        def post(self,url,json):
            seen.append(url)
            assert json['prompt']['1']['inputs']['request_id'] in runtime.jobs
            job.result={'protocol':1,'detections':[]};job.finished.set()
            return Response({'prompt_id':'our-job'})
        def get(self,url):
            seen.append(url)
            return Response({'our-job':{'status':{'status_str':'success'}}})
    monkeypatch.setattr(routes.aiohttp,'ClientSession',Session)
    assert asyncio.run(routes.execute('detection',job)) == job.result
    assert seen == ['http://127.0.0.1:8188/prompt','http://127.0.0.1:8188/history/our-job']
    assert not runtime.jobs

def test_removed_pending_job_releases_ownership(extension,monkeypatch):
    runtime,_,routes=extension
    class Response:
        status=200
        def __init__(self,body): self.body=body
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        async def json(self): return self.body
    class Session:
        def __init__(self,**kwargs): pass
        async def __aenter__(self): return self
        async def __aexit__(self,*args): pass
        def post(self,*args,**kwargs): return Response({'prompt_id':'deleted'})
        def get(self,url): return Response({} if '/history/' in url else {'queue_running':[],'queue_pending':[]})
    monkeypatch.setattr(routes.aiohttp,'ClientSession',Session)
    job=runtime.Job(Image.new('RGB',(32,32)),options={})
    with pytest.raises(RuntimeError,match='disappeared'): asyncio.run(routes.execute('detection',job))
    assert not runtime.jobs


def test_saved_graph_is_reloaded_and_preserves_workflow_options(extension,tmp_path):
    runtime,_,routes=extension
    from importlib import import_module
    workflows=import_module('test_koharu_nodes.workflows')
    from dgx.qwen.options import Options
    job=runtime.Job(Image.new('RGB',(256,256)),options=Options(seed=91,prompt='client prompt'))
    graph=routes.graph('inpainting','first',job,runtime.config())
    graph['1']['inputs']['image']='sample.png'
    graph['5']['inputs']['prompt']='workflow prompt'
    graph['9']['inputs']['cfg']=2.5
    graph['9']['inputs']['noise_seed']=42
    workflows.save_profile('inpainting',graph,{'nodes':[]})
    actual=routes.graph('inpainting','next',job,runtime.config())
    assert actual['5']['inputs']['prompt']=='workflow prompt'
    assert actual['9']['inputs']['cfg']==2.5 and actual['9']['inputs']['noise_seed']==42
    assert actual['1']['inputs']['request_id']=='next'
    assert 'image' not in actual['1']['inputs']
    graph['9']['inputs']['cfg']=3.5
    workflows.save_profile('inpainting',graph,{'nodes':[]})
    assert routes.graph('inpainting','again',job,runtime.config())['9']['inputs']['cfg']==3.5


def test_invalid_profile_does_not_replace_saved_graph(extension):
    runtime,_,routes=extension
    from importlib import import_module
    workflows=import_module('test_koharu_nodes.workflows')
    from dgx.qwen.options import Options
    job=runtime.Job(Image.new('RGB',(256,256)),options=Options())
    graph=routes.graph('inpainting','valid',job,runtime.config())
    workflows.save_profile('inpainting',graph,{'nodes':[]})
    before=workflows.profile_path('inpainting').read_bytes()
    graph['11']['inputs']['images']=['missing',0]
    with pytest.raises(ValueError,match='missing'):workflows.save_profile('inpainting',graph,{'nodes':[]})
    assert workflows.profile_path('inpainting').read_bytes()==before


def test_unrelated_output_nodes_are_not_submitted(extension):
    runtime,_,routes=extension
    from importlib import import_module
    workflows=import_module('test_koharu_nodes.workflows')
    from dgx.qwen.options import Options
    job=runtime.Job(Image.new('RGB',(256,256)),options=Options())
    graph=routes.graph('inpainting','valid',job,runtime.config())
    graph['unrelated']={'class_type':'SaveImage','inputs':{'images':['10',0]}}
    assert 'unrelated' not in workflows.bind(graph,'inpainting','owned',0)


def test_modular_ocr_accepts_postprocessed_text(extension):
    runtime,nodes,_=extension
    job=runtime.Job(Image.new('RGB',(32,32)))
    runtime.jobs['ocr']=job
    nodes.KoharuOCROutput().finish('ocr','edited',metadata={'model':'hayai','revision':'pinned','text':'raw'})
    assert job.result['text']=='edited' and job.result['model']=='hayai'
    assert job.finished.is_set()


def test_detection_output_rejects_wrong_coordinates(extension):
    runtime,nodes,_=extension
    runtime.jobs['d']=runtime.Job(Image.new('RGB',(100,200)))
    with pytest.raises(ValueError,match='coordinates'):
        nodes.KoharuDetectionOutput().finish('d',{'protocol':1,'image_width':50,'image_height':100,'detections':[]})


def test_explicit_image_mask_and_final_previews_are_kept(extension):
    runtime,_,routes=extension
    from importlib import import_module
    workflows=import_module('test_koharu_nodes.workflows')
    from dgx.qwen.options import Options
    job=runtime.Job(Image.new('RGB',(256,256)),options=Options())
    graph=routes.graph('inpainting','valid',job,runtime.config())
    graph.update({
        'source_preview':{'class_type':'PreviewImage','inputs':{'images':['1',0]}},
        'mask_image':{'class_type':'MaskToImage','inputs':{'mask':['1',1]}},
        'mask_preview':{'class_type':'PreviewImage','inputs':{'images':['mask_image',0]}},
        'final_preview':{'class_type':'PreviewImage','inputs':{'images':['11',0]}},
        'unrelated':{'class_type':'LoadImage','inputs':{'image':'other.png'}},
        'unrelated_preview':{'class_type':'PreviewImage','inputs':{'images':['unrelated',0]}}})
    bound=workflows.bind(graph,'inpainting','owned',0)
    assert {'source_preview','mask_image','mask_preview','final_preview'} <= set(bound)
    assert 'unrelated_preview' not in bound


def test_saved_ui_snapshot_is_bound_to_each_job(extension):
    runtime,_,routes=extension
    from importlib import import_module
    workflows=import_module('test_koharu_nodes.workflows')
    from dgx.qwen.options import Options
    job=runtime.Job(Image.new('RGB',(256,256)),options=Options())
    graph=routes.graph('inpainting','valid',job,runtime.config())
    ui={'nodes':[{'id':1}], 'version':0.4}
    workflows.save_profile('inpainting',graph,ui)
    routes.graph('inpainting','owned',job,runtime.config())
    assert job.workflow==ui


def test_browser_follow_uses_connected_session_and_expires(extension,monkeypatch):
    _,_,routes=extension
    server=routes.PromptServer.instance
    monkeypatch.setattr(server,'sockets',{'browser':object()},raising=False)
    now=routes.time.monotonic()
    routes.watchers.update(browser=now+90,disconnected=now+90,expired=now-1)
    assert routes.browser_client('owned')=='browser'
    assert set(routes.watchers)=={'browser'}
    routes.watchers['browser']=now-1
    assert routes.browser_client('owned')=='koharu-owned'


def test_custom_mask_preview_output_is_kept(extension,monkeypatch):
    runtime,_,routes=extension
    from importlib import import_module
    workflows=import_module('test_koharu_nodes.workflows')
    from dgx.qwen.options import Options
    sys.modules['nodes'].NODE_CLASS_MAPPINGS['MaskPreview']=types.SimpleNamespace(OUTPUT_NODE=True)
    job=runtime.Job(Image.new('RGB',(256,256)),options=Options())
    graph=routes.graph('inpainting','valid',job,runtime.config())
    graph['mask_preview']={'class_type':'MaskPreview','inputs':{'mask':['1',1]}}
    assert 'mask_preview' in workflows.bind(graph,'inpainting','owned',0)


def test_live_follow_filters_unselected_stages(extension,monkeypatch):
    _,_,routes=extension
    monkeypatch.setattr(routes.PromptServer.instance,'sockets',{'browser':object()},raising=False)
    routes.watchers['browser']=routes.time.monotonic()+90
    routes.watch_stages['browser']=('inpainting',)
    assert routes.browser_client('a','inpainting')=='browser'
    assert routes.browser_client('b','ocr')=='koharu-b'
    assert routes.browser_client('c','detection')=='koharu-c'


def test_api_binding_prunes_manual_loader_and_exposes_original_scope(extension):
    runtime,_,routes=extension
    from dgx.qwen.options import Options
    job=runtime.Job(Image.new('RGB',(64,64)),options=Options())
    graph=routes.graph('inpainting','api-job',job,runtime.config())
    assert not any(n['class_type']=='LoadImage' for n in graph.values())
    assert 'image' not in graph['1']['inputs'] and 'mask' not in graph['1']['inputs']
    assert graph['11']['inputs']['images']==['15',0]
    assert graph['15']['inputs']['mask']==['1',1]
    assert graph['15']['inputs']['resize_source'] is False
    assert graph['15']['inputs']['source']==['29',0]
    assert graph['29']['class_type']=='KoharuLocalColorMatch'
    assert graph['29']['inputs']['reference']==['1',0]
    assert graph['29']['inputs']['target']==['35',0]
    assert graph['35']['inputs']['resize_source'] is True
    assert not any(n['class_type']=='Sleep' for n in graph.values())


def test_installed_client_without_new_headers_can_receive_rgb(extension,monkeypatch):
    runtime,_,routes=extension
    monkeypatch.setattr(routes,'is_ready',lambda stage:True)
    def png(image):
        buffer=BytesIO();image.save(buffer,format='PNG');return buffer.getvalue()
    async def form(request):
        return {'image':png(Image.new('RGB',(3,4),(10,20,30))), 'mask':png(Image.new('L',(3,4)))}
    monkeypatch.setattr(routes,'form',form)
    response=asyncio.run(routes.infer(types.SimpleNamespace(headers={}), 'inpainting'))
    output=Image.open(BytesIO(response.body))
    assert response.status==200 and output.mode=='RGB' and output.size==(3,4)
    assert 'X-Koharu-Inpaint-Result' not in response.headers


def test_gui_input_uses_standard_load_image_without_resizing(extension,monkeypatch):
    _,nodes,_=extension
    image=ArrayTensor(np.ones((1,3,4,3)))
    mask=ArrayTensor(np.ones((1,3,4)))
    output=nodes.KoharuInput().load('',image,mask)
    assert output[0] is image and output[1] is mask
    with pytest.raises(ValueError,match='LoadImage'): nodes.KoharuInput().load('')


def test_ocr_repetition_retry_is_bounded_and_uses_visible_controls(extension,monkeypatch):
    _,nodes,_=extension
    from dgx.vision.hayai import HayaiOutputError
    calls=[]
    class Backend:
        def ocr(self,image,options,**kwargs):
            calls.append(options.model_dump())
            if len(calls)==1: raise HayaiOutputError('ocr_repetition_loop')
            return {'protocol':1,'text':'read','model':'test','revision':'test'}
    result=nodes.KoharuOCR().ocr(Tensor(),model=Backend(),repetition_penalty=1.0,retry_repetition_penalty=1.15)
    assert result[0]=='read' and len(calls)==2
    assert calls[0]['repetition_penalty']==1.0 and calls[1]['repetition_penalty']==1.15
    assert calls[0]['max_new_tokens']==calls[1]['max_new_tokens']
    assert result[1]['recovery']['attempts']==2


def test_ocr_repetition_at_token_limit_enters_existing_retry(extension):
    _,nodes,_=extension
    from dgx.vision.hayai import validate_result
    calls=[]
    class Backend:
        def ocr(self,image,options,**kwargs):
            calls.append(options.repetition_penalty)
            text='abc'*100 if len(calls)==1 else 'read'
            count=options.max_new_tokens if len(calls)==1 else 1
            return {'protocol':1,'text':validate_result(text,count,options.max_new_tokens)}
    result=nodes.KoharuOCR().ocr(Tensor(),model=Backend())
    assert result[0]=='read' and calls==[1.0,1.1]


@pytest.mark.parametrize('retry,penalty,code,attempts',[
    (True,1.1,'ocr_repetition_loop',2),(False,1.1,'ocr_repetition_loop',1),
    (True,1.0,'ocr_repetition_loop',1),(True,1.1,'ocr_token_limit',1)])
def test_ocr_retry_does_not_return_invalid_text(extension,monkeypatch,retry,penalty,code,attempts):
    _,nodes,_=extension
    from dgx.vision.hayai import HayaiOutputError
    calls=[]
    class Backend:
        def ocr(self,image,options,**kwargs):
            calls.append(options)
            raise HayaiOutputError(code)
    with pytest.raises(HayaiOutputError):
        nodes.KoharuOCR().ocr(Tensor(),model=Backend(),retry_on_repetition=retry,retry_repetition_penalty=penalty)
    assert len(calls)==attempts


def test_typed_ocr_history_error_maps_to_422_without_generic_500(extension,monkeypatch):
    _,_,routes=extension
    status={'messages':[['execution_error',{'exception_type':'HayaiOutputError',
            'exception_message':'ocr_repetition_loop: Hayai output contains an excessive repetition loop\n'}]]}
    error=routes.workflow_error(status)
    assert error.code=='ocr_repetition_loop'
    monkeypatch.setattr(routes,'is_ready',lambda stage:True)
    def png(image):
        buffer=BytesIO();image.save(buffer,format='PNG');return buffer.getvalue()
    async def form(request): return {'image':png(Image.new('RGB',(3,4)))}
    async def submit(*args): raise error
    monkeypatch.setattr(routes,'form',form);monkeypatch.setattr(routes,'submit',submit)
    response=asyncio.run(routes.infer(types.SimpleNamespace(headers={}), 'ocr'))
    assert response.status==422
    assert json.loads(response.text)['detail']['code']=='ocr_repetition_loop'
    status['messages'][0][1]['exception_type']='RuntimeError'
    assert isinstance(routes.workflow_error(status),RuntimeError)


def detection_fixture():
    from dgx.vision.model import rle
    items=[]
    for label,label_id,x,y in [('text',0,2,4),('onomatopoeia',1,17,12),('bubble',2,9,2)]:
        pixels=np.ones((5,5),np.uint8);pixels[1:4,1:4]=0
        items.append({'label':label,'label_id':label_id,'score':0.9,'bbox':[x,y,x+5,y+5],
            'mask':{'x':x,'y':y,'width':5,'height':5,'runs':rle(pixels)}})
    return {'protocol':1,'image_width':32,'image_height':24,'detections':items}


def test_detection_masks_fill_and_grow_roundtrip_original_coordinates(extension,monkeypatch):
    _,nodes,_=extension
    class ResultTensor(ArrayTensor):
        def to(self,**kwargs): return self
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(from_numpy=ResultTensor))
    from scipy import ndimage
    from test_koharu_nodes.detection_masks import decode_mask
    result=detection_fixture();unchanged=json.dumps(result)
    extract=nodes.NODE_CLASS_MAPPINGS['KoharuDetectionMasks']()
    apply=nodes.NODE_CLASS_MAPPINGS['KoharuDetectionMasksApply']()
    masks,context=extract.extract(result,padding=2)
    assert len(masks)==2 and masks[0].shape==(1,9,9)
    processed=[]
    for mask in masks:
        filled=nodes.KoharuMaskFillHoles().fill(mask,max_hole_area=16)[0]
        grown=ndimage.binary_dilation(filled.numpy()[0]>0.5)
        processed.append(ArrayTensor(grown.astype(np.float32)[None]))
    output=apply.apply(processed,[context],[0.5],[True])[0]
    assert output['detections'][2]==result['detections'][2]
    assert json.dumps(result)==unchanged
    for index in (0,1):
        item=output['detections'][index];original=result['detections'][index]
        mask,(x,y,w,h)=decode_mask(item['mask'],(32,24))
        assert (x,y)==(original['mask']['x']-1,original['mask']['y']-1)
        assert (w,h)==(7,7)
        assert mask[3,3]==1  # Interior missing from the original mask is now returned to Koharu.
        assert item['bbox']==[x,y,x+w,y+h]
        assert item['label']==original['label'] and item['score']==original['score']
    assert extract.OUTPUT_IS_LIST==(True,False) and apply.INPUT_IS_LIST


def test_detection_masks_empty_selection_and_no_detection(extension,monkeypatch):
    _,nodes,_=extension
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(from_numpy=ArrayTensor))
    extract=nodes.NODE_CLASS_MAPPINGS['KoharuDetectionMasks']()
    apply=nodes.NODE_CLASS_MAPPINGS['KoharuDetectionMasksApply']()
    result=detection_fixture()
    masks,context=extract.extract(result,labels='panel')
    assert len(masks)==1 and masks[0].shape==(1,1,1)
    assert apply.apply(masks,[context],[0.5],[True])[0]==result
    result['detections']=[]
    masks,context=extract.extract(result)
    assert apply.apply(masks,[context],[0.5],[True])[0]==result


def test_detection_masks_reject_bad_rle_and_changed_shape(extension,monkeypatch):
    _,nodes,_=extension
    monkeypatch.setitem(sys.modules,'torch',types.SimpleNamespace(from_numpy=ArrayTensor))
    extract=nodes.NODE_CLASS_MAPPINGS['KoharuDetectionMasks']()
    apply=nodes.NODE_CLASS_MAPPINGS['KoharuDetectionMasksApply']()
    result=detection_fixture();result['detections'][0]['mask']['runs']=[26]
    with pytest.raises(ValueError,match='RLE'):extract.extract(result)
    masks,context=extract.extract(detection_fixture(),padding=2)
    with pytest.raises(ValueError,match='count'):apply.apply(masks[:1],[context],[0.5],[True])
    masks[0]=ArrayTensor(np.zeros((1,1,1)))
    with pytest.raises(ValueError,match='dimensions'):apply.apply(masks,[context],[0.5],[True])
    from test_koharu_nodes import detection_masks
    monkeypatch.setattr(detection_masks,'MAX_PIXELS',16)
    with pytest.raises(ValueError,match='memory'):extract.extract(detection_fixture())


def test_ocr_patch_budget_is_workflow_owned_and_reused_on_retry(extension,monkeypatch):
    _,nodes,_=extension
    from dgx.vision.hayai import HayaiOutputError
    calls=[]
    class Backend:
        def ocr(self,image,options,max_patches):
            calls.append(max_patches)
            if len(calls)==1:raise HayaiOutputError('ocr_repetition_loop')
            return {'protocol':1,'text':'read'}
    assert nodes.KoharuOCR().ocr(Tensor(),model=Backend(),max_patches=1024)[0]=='read'
    assert calls==[1024,1024]
    with pytest.raises(ValueError,match='max_patches'):nodes.KoharuOCR().ocr(Tensor(),model=Backend(),max_patches=8192)


def line_image():
    pixels=np.full((1,140,800,3),0.25,np.float32)
    for y in (15,55,95):
        for x in range(30,750,20):pixels[0,y:y+20,x:x+5,:]=1
    return ArrayTensor(pixels)


def test_ocr_line_split_covers_source_and_join_preserves_reading_order(extension):
    _,nodes,_=extension
    split=nodes.NODE_CLASS_MAPPINGS['KoharuOCRLineCrops']()
    join=nodes.NODE_CLASS_MAPPINGS['KoharuOCRJoin']()
    image=line_image()
    crops,layout=split.split(image)
    assert len(crops)==3 and layout['direction']=='horizontal'
    coverage=np.zeros(image.shape[1],bool)
    for crop,box in zip(crops,layout['boxes']):
        assert crop.shape[2]==800
        coverage[box[1]:box[3]]=True
    assert coverage.all()
    text=['first','second','third']
    metadata=[{'protocol':1,'text':value,'model':'hayai','revision':'v2'} for value in text]
    merged,result=join.join(text,metadata,[layout])
    assert merged=='first\nsecond\nthird' and result['line_layout']==layout


def test_ocr_line_split_preserves_vertical_right_to_left_and_pass_through(extension):
    _,nodes,_=extension
    split=nodes.NODE_CLASS_MAPPINGS['KoharuOCRLineCrops']()
    vertical=ArrayTensor(line_image().numpy().transpose(0,2,1,3))
    crops,layout=split.split(vertical)
    assert len(crops)==3 and layout['direction']=='vertical_rtl'
    assert layout['boxes'][0][0]>layout['boxes'][-1][0]
    assert split.split(vertical,enabled=False)[0][0] is vertical
    blank=ArrayTensor(np.ones((1,80,300,3)))
    assert split.split(blank)[0][0] is blank
    square=ArrayTensor(np.ones((1,80,80,3)))
    assert split.split(square)[0][0] is square
    # Too many candidate bands fall back intact, never drop content.
    assert split.split(line_image(),max_lines=2)[0][0].shape==line_image().shape


def test_near_square_ocr_auto_detects_rows_and_columns_without_dropping_pixels(extension):
    _,nodes,_=extension
    split=nodes.NODE_CLASS_MAPPINGS['KoharuOCRLineCrops']()
    pixels=np.full((1,140,160,3),0.25,np.float32)
    for y in (15,55,95):
        for x in range(15,145,15):pixels[0,y:y+20,x:x+5,:]=1
    for image,direction,axis in [(ArrayTensor(pixels),'horizontal',1),
            (ArrayTensor(pixels.transpose(0,2,1,3)),'vertical_rtl',0)]:
        crops,layout=split.split(image)
        assert len(crops)==3 and layout['direction']==direction
        covered=np.zeros(image.shape[2-axis],bool)
        for box in layout['boxes']:covered[box[axis]:box[axis+2]]=True
        assert covered.all()


def test_near_square_ocr_auto_preserves_ambiguous_grid_and_logs_reason(extension,caplog):
    _,nodes,_=extension
    split=nodes.NODE_CLASS_MAPPINGS['KoharuOCRLineCrops']()
    pixels=np.full((1,140,140,3),0.25,np.float32)
    for y in (15,55,95):
        for x in (15,55,95):pixels[0,y:y+20,x:x+20,:]=1
    image=ArrayTensor(pixels)
    with caplog.at_level('INFO'):
        crops,layout=split.split(image,polarity='bright')
    assert crops[0] is image and layout['direction']=='unchanged'
    assert 'reason=ambiguous_rows_and_columns' in caplog.text


def test_ocr_line_join_rejects_missing_or_mismatched_results(extension):
    _,nodes,_=extension
    join=nodes.NODE_CLASS_MAPPINGS['KoharuOCRJoin']()
    layout={'boxes':[[0,0,20,10],[0,10,20,20]]}
    a={'protocol':1,'text':'a','model':'one','revision':'v2'}
    b={'protocol':1,'text':'b','model':'two','revision':'v2'}
    with pytest.raises(ValueError,match='layout'):join.join(['a'],[a],[layout])
    with pytest.raises(ValueError,match='identities'):join.join(['a','b'],[a,b],[layout])
    b={**a,'text':'wrong'}
    with pytest.raises(ValueError,match='metadata'):join.join(['a','b'],[a,b],[layout])


@pytest.fixture
def model_folders(extension,monkeypatch,tmp_path):
    roots={}
    def add(category,path):
        paths=roots.setdefault(category,[])
        if path not in paths:paths.append(path)
    folder_paths=types.SimpleNamespace(models_dir=str(tmp_path/'ComfyUI/models'),
        add_model_folder_path=add,get_folder_paths=lambda category:list(roots[category]))
    monkeypatch.setitem(sys.modules,'folder_paths',folder_paths)
    def bundle(stage,name,base=None):
        root=Path(base or folder_paths.models_dir)/'koharu'/stage/name
        root.mkdir(parents=True,exist_ok=True)
        markers={'detection':'load_model.py','ocr':'modeling_hayai.py','text_masks':'inference.py'}
        for file in ('model.safetensors','inference_config.json' if stage=='detection' else 'config.json',markers[stage]):
            (root/file).write_text('fixture')
        return root
    return folder_paths,bundle


def test_model_loaders_catalog_bundles_symlinks_and_extra_paths(extension,model_folders,tmp_path):
    _,nodes,_=extension
    from test_koharu_nodes import loaders
    folder_paths,bundle=model_folders
    detected=bundle('detection','detector-a')
    ocr=bundle('ocr','hayai-a')
    alias=Path(folder_paths.models_dir)/'koharu/detection/alias'
    alias.symlink_to(detected,target_is_directory=True)
    # Cyclic symlinks do not cause traversal loops.
    (alias.parent/'cycle').symlink_to(alias.parent,target_is_directory=True)
    external=bundle('ocr','external',base=tmp_path/'outside')
    folder_paths.add_model_folder_path('koharu_ocr',str(external.parent))
    assert loaders.resolve_model('detection','alias')==detected.resolve()
    assert loaders.resolve_model('detection','detector-a')==detected.resolve()
    assert loaders.resolve_model('ocr','external')==external.resolve()
    assert 'hayai-a' in nodes.NODE_CLASS_MAPPINGS['KoharuOCRModelLoader'].INPUT_TYPES()['required']['model_name'][0]
    assert all(value==detected.resolve() for value in loaders.model_catalog('detection').values())
    with pytest.raises(ValueError,match='Select'):loaders.resolve_model('ocr','../hayai-a')
    with pytest.raises(ValueError,match='not found'):loaders.resolve_model('ocr','missing')


def test_detector_catalog_requires_detector_inference_config(extension,model_folders):
    from test_koharu_nodes import loaders
    _,bundle=model_folders
    detected=bundle('detection','detector')
    assert not (detected/'config.json').exists()
    assert loaders.model_catalog('detection')=={'detector':detected.resolve()}
    (detected/'inference_config.json').unlink()
    (detected/'config.json').write_text('{}')
    assert loaders.model_catalog('detection')=={}


def test_model_loader_reuses_resident_backend_and_execution_has_no_path_widget(extension,model_folders,monkeypatch):
    runtime,nodes,_=extension
    _,bundle=model_folders
    a=bundle('ocr','a');b=bundle('ocr','b')
    from dgx.vision import hayai
    calls=[]
    class FakeBackend:
        def __init__(self,path):
            calls.append(str(path));self.model_name='fixture';self.path=path
        def ocr(self,image,options,max_patches):return {'protocol':1,'text':'read'}
    monkeypatch.setattr(hayai,'Backend',FakeBackend)
    loader=nodes.NODE_CLASS_MAPPINGS['KoharuOCRModelLoader']()
    model=loader.load('a')[0]
    assert loader.load('a')[0] is model and len(calls)==1
    assert nodes.KoharuOCR().ocr(Tensor(),model)[0]=='read'
    assert 'model_directory' not in nodes.KoharuOCR.INPUT_TYPES()['required']
    assert nodes.KoharuOCR.INPUT_TYPES()['required']['model'][0]=='KOHARU_OCR_MODEL'
    loader.load('b')
    assert len(calls)==2 and ('ocr',str(a.resolve())) not in runtime.backends
    assert ('ocr',str(b.resolve())) in runtime.backends


def test_vision_readiness_tracks_selected_loader_without_loading_gpu(extension,model_folders):
    runtime,nodes,_=extension
    _,bundle=model_folders
    a=bundle('detection','a');b=bundle('detection','b')
    from test_koharu_nodes.workflows import profile_path
    # An already-applied RF-DETR-only profile remains supported after factory
    # templates switch to SAM. Reading health must not introduce a SAM model.
    graph={'1':{'class_type':'KoharuInput','inputs':{'request_id':''}},
           '11':{'class_type':'KoharuDetectorModelLoader','inputs':{'model_name':'a'}},
           '2':{'class_type':'KoharuDetector','inputs':{'images':['1',0],'model':['11',0]}},
           '3':{'class_type':'KoharuDetectionOutput','inputs':{'request_id':'','result':['2',0]}}}
    profile=profile_path('detection');profile.parent.mkdir(parents=True,exist_ok=True);profile.write_text(json.dumps(graph))
    runtime.ready.add('detection');runtime.backends[('detection',str(a.resolve()))]=object()
    assert runtime.is_ready('detection')
    graph['11']['inputs']['model_name']='b';profile.write_text(json.dumps(graph))
    assert not runtime.is_ready('detection') and ('detection',str(b.resolve())) not in runtime.backends
    graph['2']['inputs'].pop('model');profile.write_text(json.dumps(graph))
    assert not runtime.is_ready('detection')


def test_default_vision_workflows_connect_explicit_model_loaders(extension):
    runtime,_,routes=extension
    for stage,kind in [('detection','KoharuDetectorModelLoader'),('ocr','KoharuOCRModelLoader')]:
        job=runtime.Job(Image.new('RGB',(32,32)),options={})
        graph=routes.graph(stage,'job',job,runtime.config())
        link=graph['2']['inputs']['model']
        assert graph[link[0]]['class_type']==kind and link[1]==0
        assert 'model_directory' not in graph['2']['inputs']


def test_sam_model_inventory_precision_cache_and_auxiliary_readiness(extension,model_folders,monkeypatch):
    runtime,nodes,_=extension
    _,bundle=model_folders
    detector=bundle('detection','detector');sam=bundle('text_masks','sam')
    from test_koharu_nodes import loaders
    assert loaders.resolve_model('text_mask','sam')==sam.resolve()
    from dgx.vision import text_sam
    calls=[]
    class FakeBackend:
        def __init__(self,path,precision):calls.append((path,precision));self.model_name='sam'
    monkeypatch.setattr(text_sam,'Backend',FakeBackend)
    loader=nodes.NODE_CLASS_MAPPINGS['KoharuTextMaskModelLoader']()
    model=loader.load('sam','bf16')[0]
    assert loader.load('sam','bf16')[0] is model and len(calls)==1
    from test_koharu_nodes.workflows import profile_path
    graph=json.loads((ROOT/'comfyui/workflows/detection-sam-ts-l.api.json').read_text())
    graph['11']['inputs']['model_name']='detector';graph['12']['inputs']['model_name']='sam'
    profile=profile_path('detection');profile.parent.mkdir(parents=True,exist_ok=True);profile.write_text(json.dumps(graph))
    runtime.ready.add('detection');runtime.backends[('detection',str(detector.resolve()))]=object()
    assert runtime.is_ready('detection')
    graph['12']['inputs']['precision']='fp32';profile.write_text(json.dumps(graph))
    assert not runtime.is_ready('detection')
    loader.load('sam','fp32')
    assert runtime.is_ready('detection') and len(calls)==2
    assert ('text_mask',str(sam.resolve()),'bf16') not in runtime.backends


def test_sam_apply_replaces_only_text_masks_and_preserves_page_coordinates(extension):
    from test_koharu_nodes.text_masks import apply_text_mask,page_mask
    from test_koharu_nodes.detection_masks import decode_mask
    result=detection_fixture()
    mask=np.zeros((24,32),np.float32);mask[5:7,3:6]=1;mask[13:16,18:20]=1
    before=json.loads(json.dumps(result))
    refined=apply_text_mask(result,mask,region_padding=0)
    assert result==before and refined['detections'][2]==result['detections'][2]
    text,(x,y,w,h)=decode_mask(refined['detections'][0]['mask'],(32,24))
    assert (x,y)==(3,5) and text.shape==(2,3) and text.all()
    assert np.array_equal(page_mask(refined),mask)
    union=apply_text_mask(result,mask,mode='union',region_padding=0)
    assert np.all(page_mask(union)>=page_mask(result))


def test_sam_empty_fallback_switch_and_disabled_apply(extension):
    _,nodes,_=extension
    from test_koharu_nodes.text_masks import apply_text_mask,page_mask
    result=detection_fixture();mask=np.zeros((24,32),np.float32)
    assert apply_text_mask(result,mask)['detections']==result['detections']
    assert not page_mask(apply_text_mask(result,mask,empty_fallback=False)).any()
    assert nodes.NODE_CLASS_MAPPINGS['KoharuTextMaskApply']().apply(result,None,enabled=False)[0] is result


def test_sam_recovers_only_uncovered_text_components_and_respects_client_limit(extension):
    from test_koharu_nodes.text_masks import apply_text_mask,page_mask
    result=detection_fixture();mask=page_mask(result)
    mask[20:22,26:28]=1
    refined=apply_text_mask(result,mask,region_padding=0,recover_missed=True,grouping_radius=0,min_component_area=2)
    assert len(refined['detections'])==4 and refined['text_mask_refinement']['recovered']==1
    recovered=refined['detections'][-1]
    assert recovered['label']=='text' and recovered['score_kind']=='unscored_mask_component'
    assert np.array_equal(page_mask(refined),mask)
    mask[20:22,2:4]=1
    with pytest.raises(ValueError,match='Too many'):
        apply_text_mask(result,mask,region_padding=0,recover_missed=True,grouping_radius=0,min_component_area=2,max_new_regions=1)


def test_sam_apply_rejects_mismatched_page_mask(extension):
    from test_koharu_nodes.text_masks import apply_text_mask
    with pytest.raises(ValueError,match='page dimensions'):
        apply_text_mask(detection_fixture(),np.zeros((12,16),np.float32))


def test_sam_candidate_workflow_binds_to_existing_detection_contract_and_keeps_previews(extension):
    runtime,_,routes=extension
    graph_path=ROOT/'comfyui/workflows/detection-sam-ts-l.api.json'
    job=runtime.Job(Image.new('RGB',(32,24)),options={})
    graph=routes.graph('detection','owned',job,{'workflows':{'detection':str(graph_path)},'models':runtime.config()['models']})
    assert '4' not in graph
    assert graph['3']['inputs']['result']==['28',0]
    assert graph['28']['inputs']['result']==['29',0]
    assert graph['29']['class_type']=='KoharuDetectionPolicy'
    assert graph['29']['inputs']['result']==['8',0]
    assert graph['20']['inputs']['result']==['28',0]
    assert graph['5']['inputs']['result']==['14',0]
    assert graph['14']['inputs']['result']==['2',0]
    assert graph['14']['inputs']['mask']==['13',0]
    assert graph['13']['inputs']['model']==['12',0]
    assert graph['3']['inputs']['request_id']=='owned'
    assert all(str(i) in graph for i in (17,19,22,26))


def nms_audit_fixture():
    from dgx.vision.model import rle
    # Equal boxes, but a low-score instance contains an additional footer line.
    first=np.zeros((8,8),bool);first[1:3,1:5]=True
    footer=np.zeros_like(first);footer[6,2:6]=True
    return {'protocol':1,'image_width':16,'image_height':16,'detections':[
        {'label_id':0,'label':'text','score':score,'bbox':[2.,2.,10.,10.],
         'mask':{'x':2,'y':2,'width':8,'height':8,'runs':rle(mask)}}
        for score,mask in [(0.9,first),(0.8,footer)]]}


def test_client_nms_audit_detects_unique_footer_loss_without_mutation(extension):
    from test_koharu_nodes.client_mask_audit import audit,nms_result
    from test_koharu_nodes.text_masks import page_mask
    result=nms_audit_fixture();before=json.loads(json.dumps(result))
    report=audit(result)
    assert report['regions']==2 and report['kept']==1
    assert report['dropped'][0]['index']==1 and report['dropped'][0]['kept_index']==0
    assert report['lost_text_pixels']==4 and report['lost_text_bounds']==[4,8,8,9]
    assert page_mask(result,'text')[8,4] and not page_mask(nms_result(result),'text')[8,4]
    assert result==before


def test_client_nms_is_class_specific_and_includes_iou_half_boundary(extension):
    from test_koharu_nodes.client_mask_audit import client_nms
    result=nms_audit_fixture()
    result['detections'][0]['bbox']=[0.,0.,6.,3.]
    result['detections'][1]['bbox']=[2.,0.,8.,3.]
    kept,dropped=client_nms(result)
    assert kept==[0] and dropped[0]['iou']==0.5
    result['detections'][1]['label']='onomatopoeia';result['detections'][1]['label_id']=1
    assert client_nms(result)==([0,1],[])
    result['detections'][1]['label']='text';result['detections'][1]['bbox'][0]=2.01
    assert client_nms(result)==([0,1],[])


def test_client_nms_tie_order_matches_top_then_rightmost_and_stable_order(extension):
    from test_koharu_nodes.client_mask_audit import client_nms
    result=nms_audit_fixture()
    result['detections'][1]['score']=0.9
    result['detections'][1]['bbox']=[3.,2.,11.,10.]
    assert client_nms(result)[0]==[1]
    result['detections'][1]['bbox']=[3.,3.,11.,11.]
    assert client_nms(result)[0]==[0]
    result['detections'][1]['bbox']=result['detections'][0]['bbox'][:]
    assert client_nms(result)[0]==[0]
    result['detections'][0]['score']=-0.0;result['detections'][1]['score']=0.0
    assert client_nms(result)[0]==[1]  # Rust f32::total_cmp distinguishes signed zero.


def test_client_nms_duplicate_pixels_are_not_reported_as_lost(extension):
    from test_koharu_nodes.client_mask_audit import audit
    result=nms_audit_fixture()
    result['detections'][1]['mask']=json.loads(json.dumps(result['detections'][0]['mask']))
    report=audit(result)
    assert report['dropped'] and report['lost_text_pixels']==0 and report['lost_text_bounds'] is None


def test_client_nms_large_page_skips_pixel_allocation_and_retains_box_report(extension):
    from test_koharu_nodes.client_mask_audit import audit
    result=nms_audit_fixture();result['image_width']=result['image_height']=10000
    report=audit(result)
    assert report['dropped'] and report['lost_text_pixels'] is None
    assert report['pixel_audit']=='skipped_page_pixel_limit'


def test_client_nms_preview_default_preserves_mask_and_opt_in_shows_loss(extension):
    _,nodes,_=extension
    node=nodes.NODE_CLASS_MAPPINGS['KoharuDetectionMaskPage']()
    result=nms_audit_fixture();before=json.loads(json.dumps(result))
    assert node.assemble(result,labels='text')[0][0,8,4]==1
    assert node.assemble(result,labels='text',client_nms=True)[0][0,8,4]==0
    assert result==before
    with pytest.raises(ValueError,match='switch'):
        node.assemble(result,client_nms='true')


def test_detection_terminal_audit_logs_loss_but_delivers_original_result(extension,caplog):
    import logging
    runtime,nodes,_=extension
    runtime.jobs['audit']=runtime.Job(Image.new('RGB',(16,16)))
    result=nms_audit_fixture()
    with caplog.at_level(logging.INFO):
        nodes.KoharuDetectionOutput().finish('audit',result)
    assert runtime.jobs['audit'].result is result and len(result['detections'])==2
    assert '"lost_text_pixels":4' in caplog.text and 'request=audit' in caplog.text


def test_graph_identity_excludes_bound_request_and_reports_actual_profile(extension,caplog):
    import logging
    runtime,_,routes=extension
    path=ROOT/'comfyui/workflows/detection-sam-ts-l.api.json'
    settings={'workflows':{'detection':str(path)},'models':runtime.config()['models']}
    with caplog.at_level(logging.INFO):
        routes.graph('detection','first',runtime.Job(Image.new('RGB',(16,16))),settings)
        routes.graph('detection','second',runtime.Job(Image.new('RGB',(16,16))),settings)
    messages=[r.getMessage() for r in caplog.records if r.getMessage().startswith('Koharu graph:')]
    identities=[m.split('api_sha256=')[1].split()[0] for m in messages]
    assert len(identities)==2 and identities[0]==identities[1]
    assert all(str(path) in m and 'KoharuTextMaskApply' in m for m in messages)


def test_mask_preserving_nms_recovers_suppressed_footer_and_keeps_winner_metadata(extension):
    from test_koharu_nodes.detection_merge import merge_overlaps
    from test_koharu_nodes.client_mask_audit import client_nms,audit
    from test_koharu_nodes.text_masks import page_mask
    result=nms_audit_fixture();before=json.loads(json.dumps(result))
    merged=merge_overlaps(result)
    assert len(merged['detections'])==1
    assert merged['detections'][0]['label']=='text' and merged['detections'][0]['score']==0.9
    assert np.array_equal(page_mask(merged,'text'),page_mask(result,'text'))
    assert merged['mask_preserving_nms']['merged']==1
    assert client_nms(merged)[1]==[] and audit(merged)['lost_text_pixels']==0
    assert result==before


def test_mask_preserving_nms_rechecks_overlaps_created_by_box_expansion(extension):
    from test_koharu_nodes.detection_merge import merge_overlaps
    from test_koharu_nodes.client_mask_audit import client_nms
    from test_koharu_nodes.text_masks import page_mask
    result=nms_audit_fixture();result['image_width']=32;result['image_height']=24
    def point(x,y):return {'x':x,'y':y,'width':1,'height':1,'runs':[0,1]}
    result['detections'][0].update(bbox=[0.,0.,6.,6.],mask=point(4,4))
    result['detections'][1].update(bbox=[1.,0.,7.,6.],mask=point(15,4))
    result['detections'].append({'label_id':0,'label':'text','score':0.7,
                                'bbox':[5.,0.,16.,6.],'mask':point(14,5)})
    # The third box survives the first NMS, but overlaps the expanded winner.
    assert client_nms(result)[0]==[0,2]
    merged=merge_overlaps(result)
    assert len(merged['detections'])==1 and merged['mask_preserving_nms']['rounds']==2
    assert merged['detections'][0]['bbox']==[0.,0.,16.,6.]
    assert np.array_equal(page_mask(merged),page_mask(result)) and not client_nms(merged)[1]


def test_mask_preserving_nms_keeps_other_labels_and_supports_disabled_comparison(extension):
    _,nodes,_=extension
    from test_koharu_nodes.detection_merge import merge_overlaps
    result=nms_audit_fixture()
    for item in result['detections']:item['label']='onomatopoeia';item['label_id']=1
    assert merge_overlaps(result)['detections']==result['detections']
    assert len(merge_overlaps(result,labels='text + onomatopoeia')['detections'])==1
    node=nodes.NODE_CLASS_MAPPINGS['KoharuDetectionMergeOverlaps']()
    assert node.merge(result,enabled=False)[0] is result


def test_mask_preserving_nms_expands_ocr_box_but_never_invents_mask_pixels(extension):
    from test_koharu_nodes.detection_merge import merge_overlaps
    from test_koharu_nodes.text_masks import page_mask
    result=nms_audit_fixture()
    result['detections'][1]['mask']={'x':12,'y':13,'width':2,'height':1,'runs':[0,2]}
    merged=merge_overlaps(result)
    assert merged['detections'][0]['bbox']==[2.,2.,14.,14.]
    assert np.array_equal(page_mask(merged),page_mask(result))
    assert merge_overlaps(result,expand_bbox=False)['detections'][0]['bbox']==[2.,2.,10.,10.]


def test_mask_preserving_nms_bounds_allocation_and_rejects_bad_switch(extension):
    from test_koharu_nodes.detection_merge import merge_overlaps
    result=nms_audit_fixture();result['image_width']=result['image_height']=10000
    result['detections'][1]['mask']={'x':9999,'y':9999,'width':1,'height':1,'runs':[0,1]}
    with pytest.raises(ValueError,match='extent'):
        merge_overlaps(result)
    _,nodes,_=extension
    with pytest.raises(ValueError,match='switch'):
        nodes.NODE_CLASS_MAPPINGS['KoharuDetectionMergeOverlaps']().merge(result,enabled='true')


def test_default_detection_template_preserves_masks_before_client_nms(extension):
    runtime,_,routes=extension
    graph=routes.graph('detection','owned',runtime.Job(Image.new('RGB',(16,16))),runtime.config())
    assert graph['3']['inputs']['result']==['28',0]
    assert graph['28']['class_type']=='KoharuDetectionMergeOverlaps'
    assert graph['28']['inputs']['result']==['29',0]
    assert graph['29']['inputs']['result']==['8',0]
    assert graph['13']['class_type']=='KoharuTextMask'


def test_detection_policy_controls_classes_without_changing_masks(extension):
    from test_koharu_nodes.detection_policy import apply_policy
    result=detection_fixture()
    result['detections'].append({**result['detections'][2],'label':'panel','label_id':3})
    original=json.dumps(result)
    kept=apply_policy(result)
    assert kept['detections']==result['detections']
    promoted=apply_policy(result,include_text=False,effects='as_text',include_bubbles=False,include_panels=False)
    assert len(promoted['detections'])==1
    item=promoted['detections'][0];effect=result['detections'][1]
    assert item['label']=='text' and item['label_id']==0
    assert item['original_label']=='onomatopoeia' and item['original_label_id']==1
    assert item['mask'] is effect['mask'] and item['bbox'] is effect['bbox']
    assert item['score']==effect['score']
    assert apply_policy(result,effects='exclude')['detections']==[result['detections'][i] for i in (0,2,3)]
    assert json.dumps(result)==original
    with pytest.raises(ValueError):apply_policy(result,effects='erase_only')
    with pytest.raises(ValueError):apply_policy(result,include_text=1)


def test_effects_promoted_before_merge_preserve_both_masks_at_client_boundary(extension):
    from test_koharu_nodes.detection_policy import apply_policy
    from test_koharu_nodes.detection_merge import merge_overlaps
    from test_koharu_nodes.client_mask_audit import client_nms
    from test_koharu_nodes.text_masks import page_mask
    result=nms_audit_fixture()
    result['detections'][1].update(label='onomatopoeia',label_id=1)
    selected=apply_policy(result,effects='as_text')
    merged=merge_overlaps(selected)
    assert len(merged['detections'])==1
    assert np.array_equal(page_mask(merged),page_mask(selected))
    assert len(client_nms(merged)[1])==0


def test_korean_metadata_preserves_saved_input_keys_and_enum_values(extension,model_folders):
    _,nodes,_=extension
    from test_koharu_nodes.ui_metadata import DEFINITIONS
    assert set(DEFINITIONS)==set(nodes.NODE_CLASS_MAPPINGS)
    for kind,cls in nodes.NODE_CLASS_MAPPINGS.items():
        definition=DEFINITIONS[kind]
        schema=cls.INPUT_TYPES()
        fields={**schema.get('required',{}),**schema.get('optional',{})}
        assert set(fields)==set(definition['inputs']),kind
        assert set(definition['outputs'])=={str(i) for i in range(len(cls.RETURN_TYPES))},kind
        for key,spec in fields.items():
            assert spec[1]['tooltip']==definition['inputs'][key]['tooltip'],(kind,key)
            localized_options=definition['inputs'][key].get('options',{})
            if localized_options:
                assert isinstance(spec[0],list),(kind,key)
                assert all(str(value) in {str(v) for v in spec[0]} for value in localized_options),(kind,key)
        assert cls.DESCRIPTION==definition['description']
        assert nodes.NODE_DISPLAY_NAME_MAPPINGS[kind]==definition['display_name']
    assert nodes.KoharuDetector.INPUT_TYPES()['required']['text'][1]['default']==0.25
    assert nodes.KoharuOCR.INPUT_TYPES()['required']['max_new_tokens'][1]['max']==2048


def workflow_response(routes,stage,template=True):
    request=types.SimpleNamespace(match_info={'stage':stage},query={'template':'1'} if template else {},headers={})
    return json.loads(asyncio.run(routes.workflow_get(request)).text)


def test_fresh_templates_keep_api_and_ui_model_selections_in_sync(extension,model_folders,monkeypatch):
    runtime,_,routes=extension
    _,bundle=model_folders
    bundle('detection','my-detector');bundle('text_masks','my-sam');bundle('ocr','my-ocr')
    configured={**runtime.config(),'models':{'model':'chosen-unet.safetensors','text_encoder':'chosen-clip.safetensors','vae':'chosen-vae.safetensors'}}
    monkeypatch.setattr(routes,'config',lambda:configured)
    for stage,selections in [('detection',{'11':('model_name','my-detector'),'12':('model_name','my-sam')}),
                             ('inpainting',{'2':('unet_name','chosen-unet.safetensors'),'3':('clip_name','chosen-clip.safetensors'),'4':('vae_name','chosen-vae.safetensors')})]:
        data=workflow_response(routes,stage)
        assert data['workflow']['nodes']
        ui={str(n['id']):n for n in data['workflow']['nodes']}
        for node_id,(field,value) in selections.items():
            assert data['prompt'][node_id]['inputs'][field]==value
            assert ui[node_id]['widgets_values_named'][field]==value
            assert ui[node_id]['widgets_values'][list(ui[node_id]['widgets_values_named']).index(field)]==value
    ocr=workflow_response(routes,'ocr')
    assert ocr['prompt']['8']['inputs']['model_name']=='my-ocr'
    assert not runtime.backends  # Opening a template never loads a model.


def test_factory_template_does_not_replace_applied_profile(extension,model_folders):
    runtime,_,routes=extension
    from test_koharu_nodes.workflows import profile_path
    saved={'prompt':{'unchanged':{'class_type':'KoharuInput','inputs':{}}},'workflow':{'nodes':[{'id':123}]}}
    path=profile_path('detection');path.parent.mkdir(parents=True);path.write_text(json.dumps(saved))
    before=path.read_bytes()
    assert workflow_response(routes,'detection',template=False)==saved
    fresh=workflow_response(routes,'detection')
    assert fresh['prompt']['29']['class_type']=='KoharuDetectionPolicy'
    assert fresh['prompt']['13']['class_type']=='KoharuTextMask'
    assert path.read_bytes()==before


@pytest.mark.parametrize('stage,companion',[('detection','detection-sam-ts-l.json'),('inpainting','qwen21-turbo-local-color.json')])
def test_factory_ui_and_api_connections_match_provided_workflow(extension,stage,companion):
    from test_koharu_nodes.workflows import snapshot
    prompt,ui=snapshot(ROOT/f'comfyui/koharu_nodes/workflows/{stage}.api.json')
    assert ui==json.loads((ROOT/'comfyui/workflows'/companion).read_text())
    nodes={str(n['id']):n for n in ui['nodes']}
    links={link[0]:link for link in ui['links']}
    for node_id,node in prompt.items():
        assert node['class_type']==nodes[node_id]['type']
        for slot,field in enumerate(nodes[node_id].get('inputs',[])):
            if field.get('link') is None:continue
            link=links[field['link']]
            assert link[3:5]==[int(node_id),slot]
            assert node['inputs'][field['name']]==[str(link[1]),link[2]]
            assert link[0] in nodes[str(link[1])]['outputs'][link[2]]['links']
        for field,value in nodes[node_id].get('widgets_values_named',{}).items():
            if field in node['inputs'] and not isinstance(node['inputs'][field],list):
                assert node['inputs'][field]==value,(node_id,field)
