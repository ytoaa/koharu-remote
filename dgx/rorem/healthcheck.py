"""Shared readiness contract; the check never loads a second model."""
import json
import os
import urllib.request

def require_ready(data):
    if not isinstance(data,dict): raise RuntimeError('Health response must be a JSON object')
    if (type(data.get('protocol')) is not int or data.get('protocol') != 1 or data.get('ready') is not True or
        data.get('model_loaded') is not True or data.get('service') != 'rorem-inpaint' or
        data.get('model') != 'RORem-4S' or data.get('device') != 'cuda'):
        raise RuntimeError('Unexpected or unready service: ' + json.dumps(data))
    return data

def main():
    request=urllib.request.Request('http://127.0.0.1:8765/health',
                                  headers={'Authorization':'Bearer '+os.environ['ROREM_API_TOKEN']} if os.environ.get('ROREM_API_TOKEN') else {})
    with urllib.request.urlopen(request,timeout=5) as response:
        require_ready(json.load(response))

if __name__=='__main__': main()
