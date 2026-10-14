"""Replace RORem with Qwen Turbo on port 8765; retain the stopped rollback container."""
import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tempfile
import time
import urllib.error
import urllib.request
from binding import address, health_url, set_address
from vision.auth import validate_token, auth_required
from rorem.limits import read_max_pixels
from rorem.upload_limits import read_max_upload_bytes
from qwen.model_files import validate_models, MANIFEST
from qwen.healthcheck import require_ready
import manage_rorem

ROOT = Path(__file__).resolve().parent
PROJECT = 'koharu-qwen'
IMAGE = 'local/koharu-qwen:0.6.0'


def env_path():
    return ROOT / '.env.qwen'


def config():
    values = {}
    for line in env_path().read_text().splitlines():
        if line and not line.startswith('#'):
            key, raw = line.split('=',1)
            value = json.loads(raw)
            if not isinstance(value,str):
                raise ValueError('Qwen configuration values must be quoted strings')
            values[key] = value.replace('$$','$')
    return values


def active():
    return env_path().exists() and config().get('QWEN_ACTIVE','false') == 'true'


def set_active(enabled):
    c=config();c['QWEN_ACTIVE']='true' if enabled else 'false'
    write_config(c)


def write_config(c):
    if env_path().is_symlink():
        raise ValueError('Refusing symlink configuration')
    fd,path=tempfile.mkstemp(prefix='.qwen-',dir=ROOT)
    try:
        with os.fdopen(fd,'w') as stream:
            stream.write('\n'.join(k+'='+json.dumps(v.replace('$','$$')) for k,v in c.items())+'\n')
            stream.flush();os.fsync(stream.fileno())
        os.replace(path,env_path())
    finally:
        if os.path.exists(path): os.unlink(path)


def validate_config(c):
    address(c['QWEN_BIND_ADDRESS'])
    required = auth_required(c.get('QWEN_AUTH_REQUIRED','false'))
    validate_token(c.get('QWEN_API_TOKEN',''), required)
    if required and len(c.get('QWEN_API_TOKEN','')) < 24:
        raise ValueError('Authentication token must contain at least 24 characters')
    read_max_pixels(c.get('QWEN_MAX_PIXELS','64000000'))
    read_max_upload_bytes(c.get('QWEN_MAX_UPLOAD_BYTES','134217728'))
    paths = [Path(c[k]).resolve() for k in ('QWEN_MODEL_DIR','QWEN_CACHE_DIR')]
    checkout = ROOT.parent.resolve()
    protected = []
    if manage_rorem.env_path().exists():
        previous = manage_rorem.config()
        protected = [Path(previous[k]).resolve() for k in ('ROREM_PIPELINE_DIR','ROREM_LORA_DIR')]
    for path in paths:
        if path.is_relative_to(checkout) or checkout.is_relative_to(path):
            raise ValueError('Qwen model/cache folders must be outside the checkout')
        for existing in protected:
            if path.is_relative_to(existing) or existing.is_relative_to(path):
                raise ValueError('Qwen folders overlap preserved RORem models')
    if paths[0].is_relative_to(paths[1]) or paths[1].is_relative_to(paths[0]):
        raise ValueError('Qwen model and cache folders must be separate')


def init(model_dir, cache_dir, bind=None):
    if env_path().exists():
        raise ValueError('.env.qwen already exists; use qwen bind to change its address')
    previous = manage_rorem.config() if manage_rorem.env_path().exists() else {}
    c = {'QWEN_ACTIVE':'false','QWEN_MODEL_DIR':str(Path(model_dir).expanduser().resolve()),
         'QWEN_CACHE_DIR':str(Path(cache_dir).expanduser().resolve()),
         'QWEN_BIND_ADDRESS':bind or previous.get('BIND_ADDRESS','127.0.0.1'),
         'QWEN_AUTH_REQUIRED':previous.get('ROREM_AUTH_REQUIRED','false'),
         'QWEN_API_TOKEN':previous.get('ROREM_API_TOKEN',''),
         'QWEN_BASE_IMAGE':previous.get('ROREM_BASE_IMAGE','local/koharu-rorem:0.2.3')}
    validate_config(c)
    # Validate all destinations before creating anything; existing folders/weights are retained.
    for key in ('QWEN_MODEL_DIR','QWEN_CACHE_DIR'):
        Path(c[key]).mkdir(parents=True,exist_ok=True)
    manage_rorem.private_write(env_path(), '\n'.join(k+'='+json.dumps(v.replace('$','$$')) for k,v in c.items())+'\n')
    print('QWEN_CONFIGURED: separate model/cache paths; existing services unchanged')


def compose(*args, capture=False):
    env = os.environ.copy()
    env.update(config())
    result = subprocess.run(['docker','compose','--project-name',PROJECT,
        '--project-directory',str(ROOT),'--file',str(ROOT/('compose.comfy.yaml' if config().get('QWEN_BACKEND')=='comfyui' else 'compose.qwen.yaml')),
        '--env-file',str(env_path()),*args],check=True,text=True,env=env,
        stdout=subprocess.PIPE if capture else None)
    return result.stdout if capture else None


def check(model=True):
    c = config()
    validate_config(c)
    for key in ('QWEN_MODEL_DIR','QWEN_CACHE_DIR'):
        if not Path(c[key]).is_dir():
            raise ValueError('Qwen model/cache folder missing')
    if c.get('QWEN_BACKEND') != 'comfyui':
        manage_rorem.image_info(c['QWEN_BASE_IMAGE'])
    compose('config','--quiet')
    if model and c.get('QWEN_BACKEND') != 'comfyui':
        validate_models(c['QWEN_MODEL_DIR'])
    print('QWEN_CONFIG_OK_GPU_NOT_TESTED')
    return c


def source_digest():
    digest = hashlib.sha256()
    paths = [ROOT/'Dockerfile.comfy',ROOT/'compose.comfy.yaml',ROOT/'Dockerfile.qwen',ROOT/'requirements.qwen.txt',ROOT/'protect_torch.py']
    paths += [p for folder in ('qwen','rorem') for p in (ROOT/folder).rglob('*')
              if p.is_file() and p.suffix in ('.py','.json')]
    for path in sorted(paths):
        digest.update(path.relative_to(ROOT).as_posix().encode()+b'\0'+path.read_bytes())
    return digest.hexdigest()


def ready_image():
    values = json.loads(manage_rorem.docker('image','inspect','local/koharu-qwen-comfy:0.6.0' if config().get('QWEN_BACKEND')=='comfyui' else IMAGE,capture=True))
    if len(values) != 1 or values[0].get('Architecture') != 'arm64':
        raise ValueError('Expected the built ARM64 Qwen image')
    labels = values[0].get('Config',{}).get('Labels',{})
    if labels.get('org.koharu.qwen.version') != '0.6.0' or labels.get('org.koharu.qwen.source') != source_digest():
        raise ValueError('Qwen image source differs; run manage.py qwen build')


def build():
    check(model=False)
    compose('build','--build-arg','QWEN_SOURCE_SHA='+source_digest(),'qwen')


def download():
    c = check(model=False)
    if c.get('QWEN_BACKEND')=='comfyui':
        raise ValueError('Download ComfyUI files with hf into ComfyUI/models; bridge has no model downloads')
    ready_image()
    if compose('ps','--status','running','-q','qwen',capture=True).split():
        raise ValueError('Stop Qwen before writing its model files')
    # No GPU and no running service required. Only the new model/cache folders are writable.
    manage_rorem.docker('run','--rm','--entrypoint','python',
        '--mount','type=bind,src='+c['QWEN_MODEL_DIR']+',dst=/download',
        '--mount','type=bind,src='+c['QWEN_CACHE_DIR']+',dst=/cache',
        '-e','HF_HOME=/cache/huggingface','-e','HF_HUB_OFFLINE=0',
        '-e','TRANSFORMERS_OFFLINE=0','-e','HF_HUB_DISABLE_TELEMETRY=1',IMAGE,'-m','qwen.download')


def request_health():
    c = config()
    token = c.get('QWEN_API_TOKEN','')
    request = urllib.request.Request(health_url(c['QWEN_BIND_ADDRESS'],8765),
                                     headers={'Authorization':'Bearer '+token} if token else {})
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(request,timeout=10) as response:
        return require_ready(json.load(response))


def idle():
    identifiers = compose('ps','--status','running','-q','qwen',capture=True).split()
    if not identifiers:
        return
    try:
        if request_health().get('busy'):
            raise ValueError('Qwen is busy; no service replacement performed')
    except urllib.error.HTTPError as error:
        if error.code != 503:
            raise
    except urllib.error.URLError as error:
        # During lifespan preload no requests are accepted. Refuse other network failures.
        if not isinstance(error.reason, ConnectionRefusedError):
            raise


def up():
    check()
    ready_image()
    idle()
    previous = manage_rorem.current() if manage_rorem.env_path().exists() else None
    stopped = False
    if previous and previous['State']['Status'] == 'running':
        manage_rorem.idle(previous)
        latest = manage_rorem.current()
        if not latest or latest['Id'] != previous['Id']:
            raise ValueError('RORem changed during replacement; no service stopped')
        record = ROOT/'state/qwen/replaced-rorem.json'
        if not record.exists():
            manage_rorem.private_write(record,json.dumps({'container_id':previous['Id'],
                'image_id':previous['Image']},indent=2)+'\n')
        manage_rorem.docker('stop',previous['Id'])
        stopped = True
    try:
        compose('up','--no-build','--no-deps','-d','qwen')
    except Exception:
        if stopped:
            # Release a partially created Qwen port binding before restoring service.
            compose('stop','qwen')
            manage_rorem.docker('start',previous['Id'])
        raise
    set_active(True)


def wait_ready(timeout=900):
    deadline=time.monotonic()+timeout
    while True:
        try:
            data=request_health()
            print(json.dumps(data,indent=2));return data
        except (OSError,ValueError,RuntimeError,urllib.error.URLError):
            if time.monotonic()>=deadline:
                raise ValueError('Qwen not ready; run manage.py qwen logs or qwen rollback') from None
            print('Waiting for Qwen preload and eight-step warmup...',flush=True)
            time.sleep(10)


def stop():
    idle()
    compose('stop','qwen')


def rollback(timeout=900):
    # No checkout rollback or model deletion: use the already retained RORem service.
    manage_rorem.check()
    container = manage_rorem.current()
    if not container:
        raise ValueError('Retained RORem container missing; restore RORem before rollback')
    record_path=ROOT/'state/qwen/replaced-rorem.json'
    if record_path.exists():
        record=json.loads(record_path.read_text())
        if container['Id']!=record['container_id'] or container['Image']!=record['image_id']:
            raise ValueError('RORem rollback container identity changed; no services stopped')
    manage_rorem.idle(container)
    stop()
    if container['State']['Status'] != 'running':
        manage_rorem.docker('start',container['Id'])
    manage_rorem.wait_ready(timeout)
    set_active(False)
    print('ROREM_RESTORED_ON_8765: reopen the retained previous Windows app; Qwen app requires Qwen')


def backend(mode, url=None):
    # Stop only our current inpainter before changing Compose variants; keep rollback config.
    idle()
    c=config()
    compose('stop','qwen')
    c['QWEN_BACKEND']=mode
    if url: c['QWEN_COMFY_URL']=url
    validate_config(c)
    write_config(c)
    print('Backend configured; run qwen build once, then qwen up and qwen wait')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest='command',required=True)
    p = sub.add_parser('init')
    p.add_argument('--model-dir',required=True)
    p.add_argument('--cache-dir',required=True)
    p.add_argument('--bind-address')
    p = sub.add_parser('bind');p.add_argument('address')
    p = sub.add_parser('backend');p.add_argument('mode',choices=['diffusers','comfyui']);p.add_argument('--url')
    for name in ('check','build','download','up','health','logs','stop'):
        sub.add_parser(name)
    for name in ('rollback','wait'):
        p = sub.add_parser(name);p.add_argument('--timeout',type=int,default=900)
    args = parser.parse_args(argv)
    if args.command == 'init':
        init(args.model_dir,args.cache_dir,args.bind_address)
    elif args.command == 'backend':
        backend(args.mode,args.url)
    elif args.command == 'bind':
        set_address(env_path(),'QWEN_BIND_ADDRESS',args.address,quoted=True)
    elif args.command == 'health':
        print(json.dumps(request_health(),indent=2))
    elif args.command == 'logs':
        compose('logs','--tail','100','qwen')
    elif args.command == 'rollback':
        rollback(args.timeout)
    elif args.command == 'wait':
        wait_ready(args.timeout)
    else:
        globals()[args.command]()


if __name__ == '__main__':
    main()
