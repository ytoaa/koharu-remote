"""Model fingerprints written outside the immutable model tree."""
from pathlib import Path
from .model_files import sha256, LORA_NAME

def entries(pipeline, lora):
    pipeline, lora = Path(pipeline).resolve(), Path(lora).resolve()
    result = []
    for p in sorted(pipeline.rglob('*')):
        if p.is_file():
            name = 'pipeline/' + p.relative_to(pipeline).as_posix()
            if '\n' in name or '\r' in name:
                raise ValueError('Manifest does not support newline filenames')
            before = p.stat()
            digest = sha256(p)
            after = p.stat()
            if (before.st_size,before.st_mtime_ns,before.st_ino) != (after.st_size,after.st_mtime_ns,after.st_ino):
                raise ValueError('Model file changed during hashing: ' + name)
            result.append((name,digest))
    if not result: raise ValueError('Empty pipeline')
    result.append(('lora/' + LORA_NAME,sha256(lora / LORA_NAME)))
    return result

def ensure_outside(target, pipeline, lora):
    target = Path(target).resolve()
    roots=[Path(pipeline).resolve(),Path(lora).resolve()]
    if roots[0].parent==roots[1].parent: roots.append(roots[0].parent)
    if any(target.is_relative_to(root) for root in roots):
        raise ValueError('Generated metadata must be outside model directories')

def create(target, pipeline, lora):
    ensure_outside(target,pipeline,lora)
    # Never overwrite an earlier baseline. Hashing only reads the model files.
    content = ''.join(f'{digest}  {name}\n' for name,digest in entries(pipeline,lora))
    target = Path(target)
    target.parent.mkdir(parents=True,exist_ok=True)
    with target.open('x') as stream: stream.write(content)
    return len(content.splitlines())

def verify(target, pipeline, lora):
    ensure_outside(target,pipeline,lora)
    expected = {}
    for line in Path(target).read_text().splitlines():
        digest,name = line.split('  ',1)
        if name in expected or len(digest)!=64 or any(c not in '0123456789abcdef' for c in digest):
            raise ValueError('Invalid model manifest')
        expected[name]=digest
    actual = dict(entries(pipeline,lora))
    missing = sorted(set(expected)-set(actual))
    added = sorted(set(actual)-set(expected))
    changed = sorted(k for k in expected.keys() & actual.keys() if expected[k]!=actual[k])
    if missing or added or changed:
        raise ValueError(f'Model manifest mismatch: missing={missing}, added={added}, changed={changed}')
    return len(actual)
