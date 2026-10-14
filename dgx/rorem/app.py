from contextlib import asynccontextmanager
from io import BytesIO
import asyncio, json, os, secrets, logging, time
from PIL import Image, UnidentifiedImageError
from fastapi import FastAPI, UploadFile, File, Form, Header, HTTPException
from fastapi.responses import Response, JSONResponse
from pydantic import ValidationError
from .options import Options
from .limits import read_max_pixels
from .strategy import preserve_outside_mask
from .upload_limits import read_max_upload_bytes

MAX_PIXELS=read_max_pixels()
Image.MAX_IMAGE_PIXELS=MAX_PIXELS
log=logging.getLogger('uvicorn.error')

def inference_error(error):
    oom=type(error).__name__=='OutOfMemoryError' or 'cuda out of memory' in str(error).lower()
    return {'code':'cuda_out_of_memory' if oom else 'inference_failed','type':type(error).__name__}

def decode(data, mode):
    try:
        with Image.open(BytesIO(data)) as im:
            if im.width*im.height > MAX_PIXELS:
                raise HTTPException(422, {'code':'image_too_large','width':im.width,'height':im.height,
                                          'pixels':im.width*im.height,'max_pixels':MAX_PIXELS})
            if im.width < 8 or im.height < 8:
                raise ValueError('image sides must be >=8px')
            im.load(); return im.convert(mode)
    except (UnidentifiedImageError,OSError,ValueError,Image.DecompressionBombError) as e:
        raise HTTPException(422,str(e))

def create_app(factory=None, token=None, *, options_type=Options,
               service_name='rorem-inpaint', model_name='RORem-4S',
               service_version='0.2.3', extra_capabilities=None, strategy_name='full-page'):
    max_bytes=read_max_upload_bytes()
    state={'ready':False,'backend':None,'error':None,'busy':False,'task':None,'last_inference_error':None}
    gate=asyncio.Lock()
    @asynccontextmanager
    async def lifespan(app):
        state['token']=token or os.environ.get('ROREM_API_TOKEN','')
        mode=os.environ.get('ROREM_AUTH_REQUIRED','false').strip().lower()
        if mode not in ('','false','0','no','true','1','yes'): raise RuntimeError('ROREM_AUTH_REQUIRED must be true or false')
        state['auth_required']=mode in ('true','1','yes')
        if state['auth_required'] and len(state['token'])<24: raise RuntimeError('ROREM_API_TOKEN must contain at least 24 characters when authentication is enabled')
        try:
            if factory is None:
                from .backend import RORemBackend
                state['backend']=await asyncio.to_thread(RORemBackend)
            else: state['backend']=await asyncio.to_thread(factory)
            if not loaded(): raise RuntimeError('Backend constructed but model is not loaded')
            state['ready']=True
        except Exception as e:
            state['error']=type(e).__name__+': '+str(e)
            log.exception('%s preload failed; service stays unavailable', model_name)
        yield
        if state['task']: await asyncio.gather(state['task'],return_exceptions=True)
        state['ready']=False
    def loaded():
        b=state['backend']
        return b is not None and getattr(b,'model_loaded',hasattr(b,'infer')) is True
    app=FastAPI(title='Koharu Remote '+model_name,version=service_version,lifespan=lifespan)
    def auth(authorization):
        if not state.get('auth_required',False): return
        expected='Bearer '+state.get('token','')
        if not authorization or not secrets.compare_digest(authorization.encode(),expected.encode()): raise HTTPException(401,'Invalid service token')
    @app.get('/health')
    def health(authorization: str|None=Header(default=None)):
        auth(authorization)
        b=state['backend']
        ready=state['ready'] and loaded()
        return JSONResponse({'protocol':1,'service':service_name,'ready':ready,
                             'model':getattr(b,'name',model_name),'model_loaded':loaded(),
                             'auth_required':state.get('auth_required',False),'auth_revision':'optional-1',
                             'busy':state['busy'],'device':getattr(b,'device',None),
                             'gpu_name':getattr(b,'gpu_name',None),'strategy':strategy_name,
                             'runtime':getattr(b,'details',None),'error':state['error'],
                             'service_version':service_version,'max_pixels':MAX_PIXELS,'max_upload_bytes':max_bytes,
                             'last_inference_error':state['last_inference_error'],
                             'capabilities':{'inpainting_options':True,'random_seed':True,
                                             **(extra_capabilities or {})}},status_code=200 if ready else 503)
    async def run(source, mask, opts):
        started=time.monotonic()
        try:
            out=await asyncio.to_thread(state['backend'].infer,source,mask,opts)
            # Even a backend that paints the full image cannot modify unmasked pixels.
            out=preserve_outside_mask(out,source,mask)
            buf=BytesIO(); out.save(buf,format='PNG')
            log.info('Inpaint completed: %dx%d, %.2fs',*source.size,time.monotonic()-started)
            state['last_inference_error']=None
            return buf.getvalue()
        except Exception as e:
            state['last_inference_error']=inference_error(e)
            log.exception('%s inference failed', model_name)
            raise
        finally:
            state['busy']=False; gate.release()
    @app.post('/v1/models/select')
    async def select_models(values: dict, authorization: str|None=Header(default=None)):
        auth(authorization)
        backend=state['backend']
        if backend is None or not hasattr(backend,'configure'):
            raise HTTPException(409,'Server does not support configurable models')
        if gate.locked(): raise HTTPException(429,'Inference busy')
        await gate.acquire();state['busy']=True;state['ready']=False
        async def select():
            try:
                await asyncio.to_thread(backend.configure,values)
                state['ready']=loaded();state['error']=None
            except Exception as error:
                state['error']=str(error);raise
            finally:
                state['busy']=False;gate.release()
        task=asyncio.create_task(select());state['task']=task
        try: await asyncio.shield(task)
        except asyncio.CancelledError: raise
        except ValueError as error: raise HTTPException(422,str(error))
        except Exception as error: raise HTTPException(503,str(error))
        return {'protocol':1,'models':backend.selected,'ready':state['ready']}

    @app.post('/v1/inpaint')
    async def inpaint(mask: UploadFile=File(),image: UploadFile|None=File(default=None),
                      source: UploadFile|None=File(default=None),options: str=Form('{}'),
                      steps: int|None=Form(default=None),strength: float|None=Form(default=None),
                      guidance_scale: float|None=Form(default=None),prompt: str|None=Form(default=None),
                      negative_prompt: str|None=Form(default=None),authorization: str|None=Header(default=None)):
        auth(authorization)
        if not state['ready'] or not loaded(): raise HTTPException(503,'Model is not ready')
        if (image is None)==(source is None): raise HTTPException(422,'Provide exactly one image (preferred) or source (v0.2.1 alias)')
        try:
            values=json.loads(options)
            if not isinstance(values,dict): raise ValueError('options must be a JSON object')
            fields={'steps':steps,'strength':strength,'guidance_scale':guidance_scale,
                    'prompt':prompt,'negative_prompt':negative_prompt}
            for name,value in fields.items():
                if value is not None:
                    if name in values: raise ValueError('Option supplied twice: '+name)
                    values[name]=value
            opts=options_type.model_validate(values)
            if hasattr(state["backend"],"validate_options"):
                state["backend"].validate_options(opts)
        except (ValidationError,ValueError) as e: raise HTTPException(422,str(e))
        upload=image if image is not None else source
        async def read_upload(file, field):
            data=await file.read(max_bytes+1)
            if len(data)>max_bytes:
                log.warning('Inpaint upload rejected: %s exceeds %d bytes',field,max_bytes)
                raise HTTPException(413,{'code':'upload_too_large','field':field,
                                         'max_upload_bytes':max_bytes})
            return data
        s=await read_upload(upload,'image' if image is not None else 'source')
        m=await read_upload(mask,'mask')
        src=decode(s,'RGB'); raw=decode(m,'L')
        if src.size!=raw.size: raise HTTPException(422,'source/mask dimensions mismatch')
        binary=raw.point(lambda v:255 if v>=127 else 0)
        if binary.getbbox() is None: raise HTTPException(422,'empty mask')
        if gate.locked(): raise HTTPException(429,'Inference busy; retry explicitly after current job finishes')
        await gate.acquire(); state['busy']=True
        task=asyncio.create_task(run(src,binary,opts)); state['task']=task
        # A disconnected client must not release the GPU gate while its CUDA work is running.
        try: result=await asyncio.shield(task)
        except asyncio.CancelledError: raise
        except Exception as e: raise HTTPException(500,inference_error(e))
        return Response(result,media_type='image/png',headers={'X-Model':model_name,'X-Protocol':'1','X-Inference-Strategy':strategy_name})
    return app

app=create_app()
