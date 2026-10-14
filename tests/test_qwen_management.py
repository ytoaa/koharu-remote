"""Replace/rollback a single active inpainter without restarting unrelated services."""
import json
import subprocess
import sys
import pytest
import manage_qwen as qwen
import manage_rorem as rorem
import manage as entry


def test_host_management_requires_only_standard_library():
    result = subprocess.run(
        [sys.executable, '-S', str(qwen.ROOT / 'manage.py'), 'qwen', '--help'],
        capture_output=True, text=True,
    )
    assert result.returncode == 0, result.stderr
    assert 'init' in result.stdout and 'download' in result.stdout


@pytest.fixture
def replacement(tmp_path,monkeypatch):
    root=tmp_path/'checkout/dgx';root.mkdir(parents=True)
    monkeypatch.setattr(qwen,'ROOT',root)
    previous={'Id':'old-rorem','Image':'sha256:retained-rorem','State':{'Status':'running'}}
    events=[]
    monkeypatch.setattr(qwen,'check',lambda **k:events.append('qwen-check'))
    monkeypatch.setattr(qwen,'ready_image',lambda:events.append('qwen-image'))
    monkeypatch.setattr(qwen,'idle',lambda:events.append('qwen-idle'))
    monkeypatch.setattr(qwen,'set_active',lambda enabled:events.append(('active',enabled)))
    monkeypatch.setattr(rorem,'env_path',lambda:root/'.env.rorem')
    (root/'.env.rorem').write_text('retained private configuration')
    monkeypatch.setattr(rorem,'current',lambda:previous)
    monkeypatch.setattr(rorem,'idle',lambda p:events.append('rorem-idle'))
    monkeypatch.setattr(rorem,'docker',lambda *a,**k:events.append(a))
    monkeypatch.setattr(qwen,'compose',lambda *a,**k:events.append(a))
    return root,previous,events


def test_replace_stops_only_rorem_retains_rollback_identity_and_uses_same_port(replacement):
    root,previous,events=replacement
    qwen.up()
    assert events.index(('stop','old-rorem'))<events.index(('up','--no-build','--no-deps','-d','qwen'))
    assert events[-1]==('active',True)
    assert not any(isinstance(e,tuple) and e[0] in ('rm','pull','prune','build') for e in events)
    record=json.loads((root/'state/qwen/replaced-rorem.json').read_text())
    assert record=={'container_id':previous['Id'],'image_id':previous['Image']}
    assert (root/'state/qwen/replaced-rorem.json').stat().st_mode&0o777==0o600
    assert (root/'.env.rorem').read_text()=='retained private configuration'


def test_busy_rorem_is_not_stopped_or_replaced(replacement,monkeypatch):
    root,previous,events=replacement
    def busy(p): raise ValueError('busy')
    monkeypatch.setattr(rorem,'idle',busy)
    with pytest.raises(ValueError,match='busy'): qwen.up()
    assert ('stop','old-rorem') not in events and ('active',True) not in events


def test_start_failure_restores_retained_container(replacement,monkeypatch):
    root,previous,events=replacement
    def compose(*args,**kw):
        events.append(args)
        if args[0]=='up': raise RuntimeError('failed')
    monkeypatch.setattr(qwen,'compose',compose)
    with pytest.raises(RuntimeError,match='failed'): qwen.up()
    assert events[-2:]==[('stop','qwen'),('start','old-rorem')]
    assert ('active',True) not in events


def test_rollback_stops_qwen_starts_original_and_switches_management(replacement,monkeypatch):
    root,previous,events=replacement
    qwen.up();events.clear();previous['State']['Status']='exited'
    monkeypatch.setattr(rorem,'check',lambda:None)
    monkeypatch.setattr(rorem,'wait_ready',lambda timeout:events.append(('rorem-ready',timeout)))
    qwen.rollback(600)
    assert events==['rorem-idle','qwen-idle',('stop','qwen'),('start','old-rorem'),('rorem-ready',600),('active',False)]
    assert not any('vision' in e or 'vllm' in e for e in events if isinstance(e,tuple))
    events.clear();previous['Image']='sha256:changed'
    with pytest.raises(ValueError,match='identity changed'):qwen.rollback()
    assert not events


def test_all_management_uses_only_active_inpainter(monkeypatch):
    import manage_vision
    events=[]
    monkeypatch.setattr(qwen,'active',lambda:True)
    monkeypatch.setattr(qwen,'check',lambda:events.append('qwen-check'))
    monkeypatch.setattr(qwen,'main',lambda args:events.append(('qwen',args)))
    monkeypatch.setattr(rorem,'main',lambda args:pytest.fail('Must not start RORem alongside Qwen'))
    monkeypatch.setattr(manage_vision,'check',lambda:events.append('vision-check'))
    monkeypatch.setattr(manage_vision,'main',lambda args:events.append(('vision',args)))
    entry.main(['all','up'])
    assert events==['vision-check','qwen-check',('vision',['up']),('qwen',['up'])]


def test_init_does_not_activate_or_overwrite_model_paths(tmp_path,monkeypatch):
    root=tmp_path/'checkout/dgx';root.mkdir(parents=True)
    monkeypatch.setattr(qwen,'ROOT',root)
    monkeypatch.setattr(rorem,'env_path',lambda:root/'.env.rorem')
    models=tmp_path/'models/qwen';models.mkdir(parents=True)
    (models/'existing').write_bytes(b'preserved')
    cache=tmp_path/'cache/qwen'
    qwen.init(models,cache,'0.0.0.0')
    assert not qwen.active() and qwen.config()['QWEN_BIND_ADDRESS']=='0.0.0.0'
    assert (models/'existing').read_bytes()==b'preserved'
    assert qwen.env_path().stat().st_mode&0o777==0o600
    qwen.set_active(True);assert qwen.active()
    qwen.set_active(False);assert not qwen.active()
    with pytest.raises(ValueError,match='already exists'):qwen.init(models,cache)

def test_backend_switch_updates_existing_config_without_rebuilding(monkeypatch,tmp_path):
    import manage_qwen as qwen
    monkeypatch.setattr(qwen,'ROOT',tmp_path)
    path=tmp_path/'.env.qwen'
    path.write_text('QWEN_ACTIVE="true"\nQWEN_BACKEND="diffusers"\nQWEN_API_TOKEN="retain-me"\n')
    events=[]
    monkeypatch.setattr(qwen,'idle',lambda:events.append('idle'))
    monkeypatch.setattr(qwen,'compose',lambda *args,**kwargs:events.append(args))
    monkeypatch.setattr(qwen,'validate_config',lambda c:None)
    qwen.backend('comfyui','http://host.docker.internal:8188')
    data=qwen.config();assert data['QWEN_BACKEND']=='comfyui'
    assert data['QWEN_API_TOKEN']=='retain-me' and data['QWEN_ACTIVE']=='true'
    assert events==['idle',('stop','qwen')]
