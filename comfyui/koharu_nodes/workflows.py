"""Read a complete API graph once per submission, never cache file contents."""
from pathlib import Path
import json, os, tempfile

def snapshot(path):
    path = Path(path)
    if path.stat().st_size > 4 * 1024 * 1024:
        raise ValueError('Workflow exceeds 4 MiB')
    graph = json.loads(path.read_text())
    workflow = graph.get('workflow') if isinstance(graph,dict) and 'prompt' in graph else None
    if isinstance(graph, dict) and 'prompt' in graph:
        graph = graph['prompt']
    if not isinstance(graph, dict) or not graph or any(
        not isinstance(node, dict) or not isinstance(node.get('class_type'), str)
        or not isinstance(node.get('inputs'), dict) for node in graph.values()
    ):
        raise ValueError('Export workflow in ComfyUI API format, not UI nodes/links format')
    return graph, workflow

def load(path):
    return snapshot(path)[0]

def bind(graph, stage, request_id, seed=None):
    input_nodes = [n for n in graph.values() if n['class_type'] == 'KoharuInput']
    expected = {'inpainting':'KoharuInpaintOutput', 'detection':'KoharuDetectionOutput', 'ocr':'KoharuOCROutput'}[stage]
    output_nodes = [n for n in graph.values() if n['class_type'] == expected]
    if len(input_nodes) != 1 or len(output_nodes) != 1:
        raise ValueError(f'Workflow requires one KoharuInput and one {expected}')
    input_nodes[0]['inputs'].pop('image', None)
    input_nodes[0]['inputs'].pop('mask', None)
    for n in graph.values():
        if n['class_type'].startswith('Koharu') and 'request_id' in n['inputs']:
            n['inputs']['request_id'] = request_id
    field = {'inpainting':'images', 'detection':'result', 'ocr':'text'}[stage]
    if not isinstance(output_nodes[0]['inputs'].get(field), list):
        raise ValueError('Koharu output must be connected to generated results')
    # Keep explicitly connected PreviewImage branches, but exclude unrelated save side effects.
    reachable = set()
    visiting = set()
    def visit(node_id):
        if node_id in visiting:
            raise ValueError("Workflow contains a cycle")
        if node_id in reachable:
            return
        if node_id not in graph:
            raise ValueError('Workflow references a missing node')
        visiting.add(node_id)
        for value in graph[node_id]['inputs'].values():
            if isinstance(value,list) and len(value)==2 and isinstance(value[1],int) and isinstance(value[0],(str,int)):
                visit(str(value[0]))
        visiting.remove(node_id)
        reachable.add(node_id)
    output_id = next(k for k,v in graph.items() if v is output_nodes[0])
    visit(output_id)
    input_id = next(k for k,v in graph.items() if v is input_nodes[0])
    if input_id not in reachable:
        raise ValueError('Koharu input must be connected to the output path')
    def uses_input(node_id, seen):
        if node_id == input_id:
            return True
        if node_id in seen:
            return False
        if node_id not in graph:
            raise ValueError('Preview references a missing node')
        seen.add(node_id)
        return any(uses_input(str(v[0]), seen) for v in graph[node_id]['inputs'].values()
                   if isinstance(v,list) and len(v)==2 and isinstance(v[1],int) and isinstance(v[0],(str,int)))
    import nodes as comfy_nodes
    registry = getattr(comfy_nodes,'NODE_CLASS_MAPPINGS',{})
    for node_id, node in graph.items():
        kind = node['class_type']
        implementation = registry.get(kind)
        preview = kind == 'PreviewImage' or (
            'preview' in kind.lower() and getattr(implementation,'OUTPUT_NODE',False))
        if preview and uses_input(node_id,set()):
            visit(node_id)
    return {k:v for k,v in graph.items() if k in reachable}


def profile_path(stage):
    if stage not in ('inpainting','detection','ocr'):
        raise ValueError('Invalid workflow stage')
    directory = Path(os.environ.get('KOHARU_WORKFLOW_DIR', Path(__file__).with_name('profiles')))
    return directory / (stage + '.json')


def save_profile(stage, prompt, workflow):
    # Validate ownership and connectivity before replacing the active snapshot.
    bind(json.loads(json.dumps(prompt)), stage, 'validation', 0)
    if not isinstance(workflow,dict) or not isinstance(workflow.get('nodes'),list):
        raise ValueError('UI workflow is missing')
    path = profile_path(stage)
    path.parent.mkdir(parents=True,exist_ok=True)
    fd, name = tempfile.mkstemp(prefix='.'+stage+'-', dir=path.parent)
    try:
        with os.fdopen(fd,'w') as stream:
            json.dump({'prompt':prompt,'workflow':workflow},stream,ensure_ascii=False)
            stream.flush(); os.fsync(stream.fileno())
        os.replace(name,path)
    finally:
        if os.path.exists(name): os.unlink(name)
    return path

def models(graph, defaults):
    result = dict(defaults)
    for kind, field, key in [('UNETLoader','unet_name','model'),('CLIPLoader','clip_name','text_encoder'),('VAELoader','vae_name','vae')]:
        found = [n['inputs'].get(field) for n in graph.values() if n['class_type'] == kind]
        if len(found) == 1 and isinstance(found[0], str):
            result[key] = found[0]
    return result
