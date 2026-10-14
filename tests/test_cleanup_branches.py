from urllib.error import HTTPError
from urllib.parse import unquote
import json
import pytest
from tools.cleanup_branches import cleanup, main

REPO='example/koharu-remote'


def pr(branch,sha='old',owner=REPO,base='main'):
    return {'head':{'ref':branch,'sha':sha,'repo':{'full_name':owner}},'base':{'ref':base}}


def api(closed,open_prs=(),branches=None,ancestors=('old',),moved=False):
    branches=branches or {}
    mutations=[]
    def request(path,method='GET'):
        if method=='DELETE':mutations.append(unquote(path));return None
        if path=='':return {'default_branch':'main'}
        if path.startswith('/pulls?'):return list(open_prs if 'state=open' in path else closed)
        if path.startswith('/branches/'):
            branch=unquote(path.removeprefix('/branches/'))
            if branch=='main':return {'commit':{'sha':'newer' if moved else 'main-sha'}}
            if branch not in branches:raise HTTPError(path,404,'not found',{},None)
            return branches[branch]
        if path.startswith('/compare/'):
            sha=path.removeprefix('/compare/').split('...')[0]
            return {'merge_base_commit':{'sha':sha if sha in ancestors else 'another'},
                    'behind_by':0 if sha in ancestors else 1}
        raise AssertionError(path)
    return request,mutations


def test_only_closed_owned_ancestor_tips_are_deleted():
    names=['merged/work','discarded/contained','discarded/unique','moved','protected',
           'open/head','open/base','fork','feat/comfyui-sidebar','main','gone']
    closed=[pr(n,sha='unique' if n=='discarded/unique' else 'old',owner='other/repo' if n=='fork' else REPO) for n in names]
    branches={n:{'commit':{'sha':'unique' if n=='discarded/unique' else 'different' if n=='moved' else 'old'},
                 'protected':n=='protected'} for n in names if n!='gone'}
    request,mutations=api(closed,[pr('open/head',base='open/base')],branches)
    assert cleanup(request,REPO,'main-sha')==['discarded/contained','merged/work']
    assert mutations==['/git/refs/heads/discarded/contained','/git/refs/heads/merged/work']


@pytest.mark.parametrize('race',['tip','open','protected','main'])
def test_branch_cleanup_rechecks_identity_and_open_prs_before_mutation(race):
    request,mutations=api([pr('old-branch')],branches={'old-branch':{'commit':{'sha':'old'}}})
    counts={}
    def racing(path,method='GET'):
        counts[path]=counts.get(path,0)+1
        if counts[path]==2:
            if race=='tip' and path=='/branches/old-branch':return {'commit':{'sha':'changed'}}
            if race=='protected' and path=='/branches/old-branch':return {'commit':{'sha':'old'},'protected':True}
            if race=='main' and path=='/branches/main':return {'commit':{'sha':'changed'}}
            if race=='open' and path.startswith('/pulls?state=open'):return [pr('old-branch')]
        return request(path,method)
    assert cleanup(racing,REPO,'main-sha')==[] and mutations==[]


def test_stale_main_run_cannot_start_cleanup():
    request,mutations=api([pr('old-branch')],moved=True)
    assert cleanup(request,REPO,'main-sha')==[] and mutations==[]


def test_cleanup_refuses_pull_request_or_other_branch_execution(monkeypatch):
    monkeypatch.setenv('GITHUB_EVENT_NAME','pull_request')
    monkeypatch.setenv('GITHUB_REF','refs/pull/10/merge')
    with pytest.raises(RuntimeError,match='only after'):main()


@pytest.mark.parametrize('change',[{'name':'Other workflow'},{'event':'pull_request'},
    {'head_branch':'feature'},{'conclusion':'failure'},{'head_repository':{'full_name':'other/repo'}}])
def test_workflow_run_payload_must_prove_successful_owned_main_checks(monkeypatch,tmp_path,change):
    payload={'workflow_run':{'name':'Koharu CI and source','event':'push','head_branch':'main',
        'conclusion':'success','head_repository':{'full_name':REPO},'head_sha':'main-sha',**change}}
    path=tmp_path/'event.json';path.write_text(json.dumps(payload))
    monkeypatch.setenv('GITHUB_EVENT_NAME','workflow_run')
    monkeypatch.setenv('GITHUB_REF','refs/heads/main')
    monkeypatch.setenv('GITHUB_EVENT_PATH',str(path))
    monkeypatch.setenv('GITHUB_REPOSITORY',REPO)
    with pytest.raises(RuntimeError,match='requires successful'):main()
