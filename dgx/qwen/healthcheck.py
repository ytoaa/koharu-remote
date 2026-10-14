import json
import os
import urllib.request
from .model_files import MODEL, REVISION


def require_ready(data):
    if (not isinstance(data, dict) or type(data.get('protocol')) is not int
            or data.get('protocol') != 1 or data.get('service') != 'qwen-inpaint'
            or (not data.get('capabilities', {}).get('configurable_models') and data.get('model') != MODEL) or data.get('ready') is not True
            or data.get('model_loaded') is not True or data.get('device') != 'cuda'
            or data.get('capabilities', {}).get('model_revision') != REVISION
            or data.get('runtime', {}).get('warmup_complete') is not True):
        raise RuntimeError('Qwen Turbo service is unavailable or incompatible')
    return data


def main():
    token = os.environ.get('ROREM_API_TOKEN', '')
    req = urllib.request.Request('http://127.0.0.1:8765/health',
                                 headers={'Authorization':'Bearer '+token} if token else {})
    with urllib.request.urlopen(req, timeout=5) as response:
        require_ready(json.load(response))


if __name__ == '__main__':
    main()
