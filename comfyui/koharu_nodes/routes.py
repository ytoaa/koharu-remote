"""Compatibility API hosted by ComfyUI, with submission to its normal queue."""
import asyncio, hashlib, json, logging, math, os, secrets, time
from io import BytesIO
import aiohttp
from aiohttp import web
from PIL import Image, UnidentifiedImageError
from server import PromptServer
from dgx.qwen.options import Options, MODEL
from dgx.vision.options import OcrOptions
from dgx.vision.hayai import HayaiOutputError
from dgx.vision.model import DEFAULTS, REPO as DETECTOR_REPO
from dgx.vision.hayai_files import REPO as OCR_REPO
from .runtime import Job, jobs, lock, config, ready, backends, is_ready

LOG = logging.getLogger(__name__)
routes = PromptServer.instance.routes
active_tasks = set()
watchers = {}
watch_stages = {}
follow_pending = {}
MAX_BYTES = 32 * 1024 * 1024

def browser_client(request_id, stage=None):
    now = time.monotonic()
    sockets = getattr(PromptServer.instance, 'sockets', {})
    for sid, expiry in list(watchers.items()):
        if expiry < now or sid not in sockets:
            watchers.pop(sid, None)
            watch_stages.pop(sid, None)
    return next((sid for sid in reversed(watchers) if stage is None or stage in watch_stages.get(sid, ('inpainting','detection','ocr'))), 'koharu-' + request_id)


def authorize(request):
    token = os.environ.get('KOHARU_COMFY_TOKEN', '')
    if token and not secrets.compare_digest(request.headers.get('Authorization', ''), 'Bearer ' + token):
        raise web.HTTPUnauthorized(text='Invalid Koharu token')

def decode(data, mode):
    if not data or len(data) > MAX_BYTES:
        raise ValueError('Invalid image size')
    with Image.open(BytesIO(data)) as image:
        if image.width * image.height > 32 * 1024 * 1024:
            raise ValueError('Image exceeds pixel limit')
        image.load()
        return image.convert(mode)

async def form(request):
    reader = await request.multipart()
    fields = {}
    while part := await reader.next():
        if part.name not in ('image', 'mask', 'options') or part.name in fields:
            raise ValueError('Unexpected or duplicate multipart field')
        chunks = bytearray()
        while chunk := await part.read_chunk():
            chunks.extend(chunk)
            if len(chunks) > (65536 if part.name == 'options' else MAX_BYTES):
                raise ValueError('Multipart field too large')
        fields[part.name] = bytes(chunks)
    return fields

def options(stage, values):
    if stage == 'inpainting':
        result = Options.model_validate(values)
        selected = config()['models']
        for key in selected:
            name = getattr(result, key)
            if name and not (key == 'model' and name == MODEL) and name != selected[key]:
                raise ValueError('Model differs from server config; edit config and warm up')
        return result
    if stage == 'ocr':
        result = OcrOptions.model_validate(values)
        identity = next((b.model_name for k,b in backends.items() if k[0] == stage), OCR_REPO)
        if result.model and result.model != identity:
            raise ValueError('Selected OCR model is not loaded')
        return result
    values = dict(values)
    model = values.pop('model', '')
    identity = next((b.model_name for k,b in backends.items() if k[0] == stage), DETECTOR_REPO)
    if model and model != identity:
        raise ValueError('Selected detector is not loaded')
    if set(values) - set(DEFAULTS):
        raise ValueError('Unknown detection option')
    thresholds = {**DEFAULTS, **values}
    if any(type(v) not in (int,float) or not math.isfinite(v) or not 0 <= v <= 1 for v in thresholds.values()):
        raise ValueError('Invalid threshold')
    return thresholds

def graph(stage, request_id, job, settings):
    from pathlib import Path
    path = settings.get('workflows', {}).get(stage) or Path(__file__).with_name('workflows') / (stage+'.api.json')
    from .workflows import snapshot, bind, models
    workflow, job.workflow = snapshot(path)
    job.selected_models = models(workflow, settings['models'])
    # Hash the unbound API graph: request IDs must not change its identity.
    digest=hashlib.sha256(json.dumps(workflow,sort_keys=True,separators=(',',':')).encode()).hexdigest()[:16]
    bound=bind(workflow, stage, request_id)
    kinds=sorted({node['class_type'] for node in bound.values()})
    LOG.info('Koharu graph: stage=%s request=%s profile=%s api_sha256=%s nodes=%s',
             stage,request_id,path,digest,','.join(kinds))
    return bound

def workflow_error(status):
    # Map only our typed output-validation failures, never arbitrary Comfy errors.
    for event, details in status.get('messages', []):
        if event != 'execution_error' or not isinstance(details, dict):
            continue
        kind = details.get('exception_type', '')
        code = details.get('exception_message', '').split(':', 1)[0]
        if (kind == 'HayaiOutputError' or kind.endswith('.HayaiOutputError')) and code in HayaiOutputError.MESSAGES:
            return HayaiOutputError(code)
    return RuntimeError('Comfy workflow failed: ' + json.dumps(status))


async def execute(stage, job):
    settings = config()
    request_id = secrets.token_hex(16)
    with lock:
        if len(jobs) >= 4:
            raise web.HTTPTooManyRequests(text='Koharu queue full')
        jobs[request_id] = job
    try:
        workflow = graph(stage, request_id, job, settings)
        client_id = browser_client(request_id,stage)
        if client_id in watchers:
            event = asyncio.Event()
            follow_pending[request_id] = (client_id, event)
            try:
                PromptServer.instance.send_sync('koharu_follow', {'stage':stage,'workflow':job.workflow,'prompt':workflow,'request_id':request_id},client_id)
                # Load the matching graph before native execution/preview events arrive.
                await asyncio.wait_for(event.wait(),5)
            except asyncio.TimeoutError:
                LOG.warning('Browser follow did not acknowledge graph load; inference continues')
            finally:
                follow_pending.pop(request_id,None)
        async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=30), trust_env=False) as session:
            async with session.post(settings['api_url'].rstrip('/') + '/prompt', json={
                    'prompt': workflow, 'client_id': client_id,
                    'extra_data': {'extra_pnginfo': {'workflow': job.workflow}} if job.workflow else {}}) as response:
                result = await response.json()
            if response.status != 200 or result.get('node_errors') or not result.get('prompt_id'):
                raise RuntimeError('Workflow validation failed: ' + json.dumps(result))
            pid = result['prompt_id']
            LOG.info('Koharu submitted stage=%s prompt_id=%s', stage, pid)
            absent = 0
            while True:
                try:
                    async with session.get(settings['api_url'].rstrip('/') + '/history/' + pid) as response:
                        history = (await response.json()).get(pid)
                    if history:
                        if history.get('status', {}).get('status_str') == 'error':
                            raise workflow_error(history.get('status', {}))
                        if not job.finished.is_set() or job.result is None:
                            raise RuntimeError('Workflow completed without Koharu output')
                        if job.error:
                            raise RuntimeError(job.error)
                        LOG.info('Koharu completed stage=%s prompt_id=%s total=%.3fs', stage,pid,time.monotonic()-job.created)
                        return job.result
                    async with session.get(settings['api_url'].rstrip('/') + '/queue') as response:
                        queue = await response.json()
                    owned = any(len(item) > 1 and item[1] == pid
                                for key in ('queue_running','queue_pending') for item in queue.get(key, []))
                    absent = 0 if owned else absent + 1
                    if absent >= 3:
                        raise RuntimeError('Owned workflow disappeared from queue without a result')
                except (aiohttp.ClientError, asyncio.TimeoutError):
                    # Keep ownership while status is unknown. Never interrupt GUI jobs.
                    absent = 0
                    LOG.warning('Koharu completion status unavailable: %s', pid)
                await asyncio.sleep(0.25)
    finally:
        # No timeout/cancellation of execute() after submission: route awaits shield.
        with lock:
            jobs.pop(request_id, None)

async def submit(stage, job):
    task = asyncio.create_task(execute(stage, job))
    active_tasks.add(task)
    def done(t):
        active_tasks.discard(t)
        if not t.cancelled():
            t.exception()  # Retrieve exceptions even when the HTTP caller disconnected.
    task.add_done_callback(done)
    try:
        return await asyncio.wait_for(asyncio.shield(task), config()['timeout'])
    except asyncio.TimeoutError:
        raise web.HTTPGatewayTimeout(text='Request deadline exceeded; queued job still owned until completion')

async def infer(request, stage):
    authorize(request)
    if not is_ready(stage):
        raise web.HTTPServiceUnavailable(text='Run /koharu/warmup before inference')
    try:
        fields = await form(request)
        image = decode(fields['image'], 'RGB')
        values = json.loads(fields.get('options', b'{}'))
        if not isinstance(values, dict):
            raise ValueError('Options must be an object')
        opts = options(stage, values)
        mask = decode(fields['mask'], 'L') if stage == 'inpainting' else None
        if mask is not None and mask.size != image.size:
            raise ValueError('Mask dimensions differ from image')
        job = Job(image, mask, opts)
        if stage == 'inpainting':
            if not mask.getbbox():
                buffer = BytesIO(); image.save(buffer, format='PNG')
                return web.Response(body=buffer.getvalue(), content_type='image/png')
        result = await submit(stage, job)
        return (web.Response(body=result, content_type='image/png') if stage == 'inpainting'
                else web.json_response(result))
    except web.HTTPException:
        raise
    except HayaiOutputError as exc:
        LOG.warning('Koharu OCR result rejected: %s', exc.code)
        return web.json_response({'detail': {'code': exc.code, 'message': str(exc)}}, status=422)
    except (ValueError, KeyError, TypeError, UnidentifiedImageError, Image.DecompressionBombError) as exc:
        raise web.HTTPUnprocessableEntity(text=str(exc))
    except Exception:
        LOG.exception('Koharu inference failed: %s', stage)
        raise web.HTTPInternalServerError(text='Koharu inference failed; inspect ComfyUI logs')

@routes.post('/koharu/inpaint/v1/inpaint')
async def inpaint(request):
    return await infer(request, 'inpainting')

@routes.post('/koharu/vision/v1/detect')
async def detect(request):
    return await infer(request, 'detection')

@routes.post('/koharu/vision/v1/ocr')
async def ocr(request):
    return await infer(request, 'ocr')

@routes.get('/koharu/inpaint/health')
async def inpaint_health(request):
    authorize(request)
    loaded = is_ready('inpainting')
    return web.json_response({'protocol':1,'service':'qwen-inpaint','ready':loaded,'model_loaded':loaded,
        'model':MODEL,'device':'cuda','busy':bool(jobs),
        'capabilities':{'configurable_models':True,'inpainting_options':True,'random_seed':True},
        'runtime':{'models':config()['models']}})

@routes.get('/koharu/vision/health')
async def vision_health(request):
    authorize(request)
    loaded = is_ready('detection') and is_ready('ocr')
    identities = {k[0]:b.model_name for k,b in backends.items()}
    return web.json_response({'protocol':1,'service':'koharu-vision','ready':loaded,'model_loaded':loaded,
        'device':'cuda','model':identities.get('detection', DETECTOR_REPO),
        'ocr_model':identities.get('ocr', OCR_REPO),'ocr_model_loaded':is_ready('ocr'),'busy':bool(jobs),
        'capabilities':{'hayai_ocr_options':True,'detection_thresholds':True}})

@routes.post('/koharu/warmup')
async def warmup(request):
    authorize(request)
    try:
        values = await request.json() if request.can_read_body else {}
        stages = values.get('stages', ['inpainting','detection','ocr'])
        if not isinstance(stages,list) or not stages or any(s not in ('inpainting','detection','ocr') for s in stages):
            raise web.HTTPUnprocessableEntity(text='Invalid warmup stages')
        for stage in stages:
            image = Image.new('RGB',(256,256),'white')
            await submit(stage, Job(image, Image.new('L',image.size,255), options(stage, {})))
        return web.json_response({'ready':True,'models':config()['models']})
    except web.HTTPException:
        raise
    except Exception:
        LOG.exception('Koharu warmup failed')
        raise web.HTTPInternalServerError(text='Warmup failed; inspect ComfyUI logs')

@routes.post('/koharu/inpaint/v1/models/select')
async def select(request):
    authorize(request)
    values = await request.json()
    selected = config()['models']
    if not isinstance(values,dict) or set(values)-set(selected) or any(selected[k]!=v for k,v in values.items()):
        raise web.HTTPConflict(text='Set model filenames in server config, then warm up')
    image = Image.new('RGB',(256,256),'white')
    await submit('inpainting', Job(image,Image.new('L',image.size,255),Options(max_side=256)))
    return web.json_response({'ready':True,'models':selected})

@routes.get('/koharu/workflows/{stage}')
async def workflow_get(request):
    authorize(request)
    from pathlib import Path
    from .workflows import profile_path, snapshot
    stage = request.match_info['stage']
    try:
        saved = profile_path(stage)
        template = getattr(request, 'query', {}).get('template') == '1'
        if saved.exists() and not template:
            return web.json_response(json.loads(saved.read_text()))
        selected = None if template else config().get('workflows',{}).get(stage)
        path = Path(selected) if selected else Path(__file__).with_name('workflows') / (stage+'.api.json')
        prompt, workflow = snapshot(path)
        def select_widget(kind, field, value):
            for node in prompt.values():
                if node['class_type'] == kind:
                    node['inputs'][field] = value
            if workflow:
                for node in workflow.get('nodes',[]):
                    if node.get('type') != kind:
                        continue
                    named=node.get('widgets_values_named',{})
                    if field in named:
                        index=list(named).index(field)
                        if index < len(node.get('widgets_values',[])):
                            node['widgets_values'][index]=value
                        named[field]=value
        # Keep Qwen selections and show available loader bundles in fresh templates.
        settings = config()
        if template and stage == 'inpainting':
            for kind, field, key in [('UNETLoader','unet_name','model'),('CLIPLoader','clip_name','text_encoder'),('VAELoader','vae_name','vae')]:
                select_widget(kind,field,settings['models'][key])
        from .loaders import model_catalog, KINDS
        for model_stage in (['detection','text_mask'] if stage=='detection' else ['ocr'] if stage=='ocr' else []):
            names=list(model_catalog(model_stage))
            loader_kind=KINDS[model_stage][3]
            if names and any(node['class_type']==loader_kind and node['inputs'].get('model_name') not in names for node in prompt.values()):
                select_widget(loader_kind,'model_name',names[0])
        return web.json_response({'prompt':prompt,'workflow':workflow})
    except (ValueError,OSError) as exc:
        raise web.HTTPUnprocessableEntity(text=str(exc))

@routes.post('/koharu/workflows/{stage}')
async def workflow_save(request):
    authorize(request)
    from .workflows import save_profile
    import nodes as comfy_nodes
    stage = request.match_info['stage']
    try:
        raw = await request.content.read(4*1024*1024+1)
        if len(raw)>4*1024*1024:
            raise ValueError('Workflow exceeds 4 MiB')
        values=json.loads(raw)
        prompt=values['prompt']
        if not isinstance(prompt,dict):
            raise ValueError('Invalid API graph')
        for node in prompt.values():
            if node.get('class_type') not in comfy_nodes.NODE_CLASS_MAPPINGS:
                raise ValueError('Missing ComfyUI node: '+str(node.get('class_type')))
        save_profile(stage,prompt,values['workflow'])
        return web.json_response({'saved':True,'stage':stage,'next_request':True})
    except (ValueError,KeyError,TypeError,OSError) as exc:
        raise web.HTTPUnprocessableEntity(text=str(exc))


@routes.post('/koharu/watch')
async def watch(request):
    authorize(request)
    try:
        values = await request.json()
        sid = values['client_id']
        if not isinstance(sid,str) or not sid or len(sid)>128:
            raise ValueError('Invalid browser session')
        if values.get('enabled') is False:
            watchers.pop(sid,None)
            watch_stages.pop(sid,None)
            return web.json_response({'watching':False})
        stages = values.get('stages',['inpainting','detection','ocr'])
        if not isinstance(stages,list) or any(s not in ('inpainting','detection','ocr') for s in stages):
            raise ValueError('Invalid watched stages')
        browser_client('cleanup')
        if sid not in getattr(PromptServer.instance,'sockets',{}):
            raise ValueError('ComfyUI WebSocket session is not connected')
        if sid not in watchers and len(watchers)>=16:
            raise ValueError('Too many watching browsers')
        watchers.pop(sid,None)
        watchers[sid] = time.monotonic()+90
        watch_stages[sid] = tuple(stages)
        return web.json_response({'watching':True})
    except (ValueError,KeyError,TypeError) as exc:
        raise web.HTTPUnprocessableEntity(text=str(exc))


@routes.post('/koharu/watch/ready')
async def watch_ready(request):
    authorize(request)
    try:
        values=await request.json()
        pending=follow_pending.get(values['request_id'])
        if pending and values.get('client_id')==pending[0]:
            pending[1].set()
        return web.json_response({'acknowledged':bool(pending and values.get('client_id')==pending[0])})
    except (ValueError,KeyError,TypeError) as exc:
        raise web.HTTPUnprocessableEntity(text=str(exc))
