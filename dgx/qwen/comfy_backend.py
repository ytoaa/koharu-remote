"""Existing ComfyUI adapter; does not change packages or unload other models."""
from io import BytesIO
import json, logging, math, os, secrets, time
from types import SimpleNamespace
import urllib.parse, urllib.request
from PIL import Image
from .model_files import SAMPLE_SIGMAS
from .options import MODEL, Options, DEFAULT_PROMPT
LOG=logging.getLogger('uvicorn.error')
DEFAULT_MODELS={'model':'qwen_image_2.1_turbo_int8_convrot.safetensors','text_encoder':'qwen3vl_8b_int8_convrot.safetensors','vae':'qwen_image_2.1_vae_bf16.safetensors'}

def build_graph(image,opts,selected,seed,resolution):
    def n(kind,**inputs): return {'class_type':kind,'inputs':inputs}
    return {
      '1':n('LoadImage',image=image),
      '2':n('UNETLoader',unet_name=selected['model'],weight_dtype='default'),
      '3':n('CLIPLoader',clip_name=selected['text_encoder'],type='qwen_image',device='default'),
      '4':n('VAELoader',vae_name=selected['vae']),
      '5':n('TextEncodeQwenImage21',clip=['3',0],vae=['4',0],**{'images.image_1':['1',0]},prompt=opts.prompt,negative_prompt='',resolution=resolution),
      '6':n('QwenImage21Cache',model=['2',0],device='auto' if opts.use_kv_cache else 'off',dtype='default'),
      '7':n('KSamplerSelect',sampler_name='euler'),
      '8':n('ManualSigmas',sigmas=', '.join(map(str,SAMPLE_SIGMAS+[0.0]))),
      '9':n('SamplerCustom',model=['6',0],add_noise=True,noise_seed=seed,cfg=1.0,positive=['5',0],negative=['5',1],sampler=['7',0],sigmas=['8',0],latent_image=['5',2]),
      '10':n('VAEDecode',samples=['9',0],vae=['4',0]),
      '11':n('SaveImage',images=['10',0],filename_prefix='koharu_bridge')}

def encode_upload(image):
    data=BytesIO();image.save(data,format='PNG');b='koharu-'+secrets.token_hex(16)
    body=(f'--{b}\r\nContent-Disposition: form-data; name="image"; filename="crop.png"\r\nContent-Type: image/png\r\n\r\n').encode()+data.getvalue()
    body+=f'\r\n--{b}\r\nContent-Disposition: form-data; name="type"\r\n\r\ninput\r\n--{b}--\r\n'.encode()
    return body,'multipart/form-data; boundary='+b

class ComfyBackend:
    def __init__(self):
        self.model_loaded=False;self.url=os.environ.get('QWEN_COMFY_URL','http://127.0.0.1:8188').rstrip('/')
        p=urllib.parse.urlsplit(self.url)
        if p.scheme not in ('http','https') or not p.hostname or p.username or p.password or p.query or p.fragment: raise ValueError('Invalid ComfyUI URL')
        self.timeout=int(os.environ.get('QWEN_COMFY_TIMEOUT','1800'))
        if not 30<=self.timeout<=7200: raise ValueError('Comfy timeout must be 30..7200')
        self.opener=urllib.request.build_opener(urllib.request.ProxyHandler({}))
        self.info=self.json('/object_info')
        kinds=['LoadImage','UNETLoader','CLIPLoader','VAELoader','TextEncodeQwenImage21','QwenImage21Cache','KSamplerSelect','ManualSigmas','SamplerCustom','VAEDecode','SaveImage']
        if any(k not in self.info for k in kinds): raise ValueError('ComfyUI Qwen 2.1 nodes missing')
        gpu=next((d for d in self.json('/system_stats').get('devices',[]) if d.get('type')=='cuda'),None)
        if not gpu: raise ValueError('ComfyUI CUDA device unavailable')
        self.device='cuda';self.gpu_name=gpu.get('name','CUDA');self.strategy=SimpleNamespace(name='comfy-crop-mask-composite')
        self.selected={k:os.environ.get('QWEN_COMFY_'+k.upper(),v) for k,v in DEFAULT_MODELS.items()};self.name=MODEL
        self.details={'backend':'comfyui','steps':8,'cfg':1.0,'warmup_complete':False,'models':dict(self.selected)}
        self.configure(self.selected,force=True)
    def request(self,path,data=None,content_type='application/json'):
        req=urllib.request.Request(self.url+path,data=data,headers={'Content-Type':content_type} if data is not None else {})
        with self.opener.open(req,timeout=30) as r:
            raw=r.read(134217729)
            if len(raw)>134217728: raise ValueError('Comfy response too large')
            return raw
    def json(self,path,data=None): return json.loads(self.request(path,None if data is None else json.dumps(data).encode()))
    def configure(self,values,force=False):
        if set(values)-set(DEFAULT_MODELS): raise ValueError('Unknown model setting')
        selected={**self.selected,**{k:v for k,v in values.items() if v and not(k=='model' and v==MODEL)}}
        self.info=self.json('/object_info')
        for key,kind,field in [('model','UNETLoader','unet_name'),('text_encoder','CLIPLoader','clip_name'),('vae','VAELoader','vae_name')]:
            if selected[key] not in self.info[kind]['input']['required'][field][0]: raise ValueError('Comfy model not installed: '+selected[key])
        if selected==self.selected and not force and self.model_loaded:return
        old=dict(self.selected);start=time.monotonic();self.model_loaded=False;self.details['warmup_complete']=False;self.selected=selected
        try:
            fixture=Image.new('RGB',(256,256),'white');self.infer(fixture,Image.new('L',fixture.size,255),Options(max_side=256, prompt='Remove all text from <image1>. Preserve the artwork.'))
        except Exception:
            self.selected=old;raise
        self.name=MODEL;self.model_loaded=True
        self.details.update(models=dict(selected),warmup_complete=True,warmup_seconds=round(time.monotonic()-start,3))
    def infer(self,source,mask,opts):
        if opts.prompt == 'Remove the text inside the white mask and reconstruct the original background. Preserve the illustration style, line art and screentone.':
            opts=opts.model_copy(update={'prompt':DEFAULT_PROMPT})
        if opts.mask_dilation:
            raise ValueError("Comfy instruction edit does not use a reference mask; set mask dilation to 0")
        for key,v in {'model':opts.model,'text_encoder':opts.text_encoder,'vae':opts.vae}.items():
            if v and not(key=='model' and v==MODEL) and v!=self.selected[key]:raise ValueError('Select and preload models before inference')
        scale=min(1.0,opts.max_side/max(source.size));size=tuple(max(32,math.floor(v*scale/32)*32) for v in source.size)
        crop=source.resize(size,Image.Resampling.LANCZOS).convert('RGB');body,typ=encode_upload(crop)
        upload=json.loads(self.request('/upload/image',body,typ));image=(upload.get('subfolder','').strip('/')+'/' if upload.get('subfolder') else '')+upload['name']
        seed=secrets.randbits(31) if opts.seed==-1 else opts.seed;start=time.monotonic()
        result=self.json('/prompt',{'prompt':build_graph(image,opts,self.selected,seed,0),'client_id':'koharu-'+secrets.token_hex(16)})
        if result.get('node_errors') or not result.get('prompt_id'):raise ValueError('Comfy rejected workflow: '+json.dumps(result))
        pid=result['prompt_id'];deadline=start+self.timeout;timed_out=False
        while True:
            try:
                history=self.json('/history/'+urllib.parse.quote(pid,safe='')).get(pid)
            except (OSError,ValueError):
                LOG.warning('Comfy history unavailable for %s; keep gate until completion is known',pid)
                time.sleep(2);continue
            if history:
                if timed_out:raise TimeoutError('Comfy deadline exceeded; submitted job has finished')
                status=history.get('status',{})
                if status.get('status_str')=='error':raise RuntimeError('Comfy inference failed: '+json.dumps(status.get('messages',[])))
                images=history.get('outputs',{}).get('11',{}).get('images',[])
                if images:
                    query=urllib.parse.urlencode({k:images[0].get(k,'') for k in ('filename','subfolder','type')});raw=self.request('/view?'+query)
                    with Image.open(BytesIO(raw)) as im:im.load();out=im.convert('RGB').resize(source.size,Image.Resampling.LANCZOS)
                    LOG.info('Comfy inpaint: prompt_id=%s seed=%d crop=%dx%d %.2fs',pid,seed,*size,time.monotonic()-start);return out
            if time.monotonic()>deadline and not timed_out:
                timed_out=True;LOG.error('Comfy job %s timed out; gate retained until its own job finishes',pid)
            # Do not globally interrupt another user's ComfyUI tasks or release a running job's gate.
            time.sleep(0.5 if not timed_out else 2)
