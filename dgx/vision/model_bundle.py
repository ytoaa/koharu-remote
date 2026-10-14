"""Runtime model identities for compatible local weights; immutable adapter code."""
import hashlib,json
from pathlib import Path

def identity(folder,repo,revision):
    path=Path(folder)/'model-files.json'
    if not path.exists():return repo,revision
    data=json.loads(path.read_text())
    return data.get('repository',repo),data.get('revision',revision)

def validate_bundle(folder,repo,revision,files,code_files=()):
    folder=Path(folder);name,rev=identity(folder,repo,revision)
    if (name,rev)==(repo,revision): expected=files
    else:
        manifest=json.loads((folder/'model-files.json').read_text())
        expected=manifest.get('files',{})
        if not isinstance(name,str) or not name.strip() or not isinstance(rev,str) or not rev.strip():raise ValueError('Model identity missing')
        if set(expected)!=set(files):raise ValueError('Compatible model manifest file set mismatch')
        for file in code_files:
            if expected[file]!=files[file]:raise ValueError('Model loader code is incompatible; code changes need a rebuild')
    for name,digest in expected.items():
        if not isinstance(digest,str) or len(digest)!=64:raise ValueError('Model SHA256 missing')
        h=hashlib.sha256()
        with (folder/name).open('rb') as stream:
            for chunk in iter(lambda:stream.read(8*1024*1024),b''):h.update(chunk)
        if h.hexdigest()!=digest:raise ValueError('Model pinned file hash mismatch: '+name)
    return folder
