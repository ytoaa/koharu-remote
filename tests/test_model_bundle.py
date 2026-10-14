import hashlib,json
import pytest
from vision.model_bundle import validate_bundle,identity

def test_compatible_weights_can_change_without_loader_changes(tmp_path):
    files={'model.safetensors':b'old weights','loader.py':b'fixed loader'}
    pinned={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
    files['model.safetensors']=b'new weights'
    for k,v in files.items():(tmp_path/k).write_bytes(v)
    hashes={k:hashlib.sha256(v).hexdigest() for k,v in files.items()}
    (tmp_path/'model-files.json').write_text(json.dumps({'repository':'local/new-model','revision':'local-v2','files':hashes}))
    assert validate_bundle(tmp_path,'old/model','old-revision',pinned,('loader.py',))==tmp_path
    assert identity(tmp_path,'old/model','old-revision')==('local/new-model','local-v2')
    files['loader.py']=b'incompatible new code';(tmp_path/'loader.py').write_bytes(files['loader.py'])
    hashes['loader.py']=hashlib.sha256(files['loader.py']).hexdigest()
    (tmp_path/'model-files.json').write_text(json.dumps({'repository':'local/new-model','revision':'local-v2','files':hashes}))
    with pytest.raises(ValueError,match='incompatible'):validate_bundle(tmp_path,'old/model','old-revision',pinned,('loader.py',))
