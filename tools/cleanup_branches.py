"""Delete only closed-PR branch tips fully preserved in the current main commit."""
import json
import os
from pathlib import Path
import urllib.error
import urllib.parse
import urllib.request

KEEP = {'main', 'feat/comfyui-sidebar'}  # DGX continues pulling the active branch.


def cleanup(request, repository, target_sha):
    metadata = request('')
    default = metadata['default_branch']
    if request('/branches/'+urllib.parse.quote(default,safe=''))['commit']['sha'] != target_sha:
        print('Cleanup skipped: main advanced beyond this workflow')
        return []

    def pulls(state):
        result = []
        for page in range(1,101):
            items=request(f'/pulls?state={state}&per_page=100&page={page}')
            result.extend(items)
            if len(items)<100:return result
        raise RuntimeError('Too many PR pages; no branches deleted')

    active = pulls('open')
    protected = KEEP | {default} | {p[side]['ref'] for p in active for side in ('head','base')}
    candidates = {}
    for pr in pulls('closed'):
        head=pr['head']
        if (head.get('repo') or {}).get('full_name')==repository and head['ref'] not in protected:
            candidates.setdefault(head['ref'],set()).add(head['sha'])
    deleted=[]
    for branch,tips in sorted(candidates.items()):
        path='/branches/'+urllib.parse.quote(branch,safe='')
        try:
            current=request(path)
        except urllib.error.HTTPError as error:
            if error.code==404:continue
            raise
        sha=current['commit']['sha']
        if current.get('protected') or sha not in tips:
            continue
        # An unmerged/discarded branch with unique commits is retained too.
        comparison=request('/compare/'+sha+'...'+target_sha)
        if comparison.get('merge_base_commit',{}).get('sha')!=sha or comparison.get('behind_by')!=0:
            continue
        # Recheck open PR use and branch identity immediately before deleting.
        if any(p[side]['ref']==branch for p in pulls('open') for side in ('head','base')):
            continue
        if request('/branches/'+urllib.parse.quote(default,safe=''))['commit']['sha']!=target_sha:
            break
        latest=request(path)
        if latest.get('protected') or latest['commit']['sha']!=sha:
            continue
        request('/git/refs/heads/'+urllib.parse.quote(branch,safe=''),method='DELETE')
        deleted.append(branch)
        print('Deleted closed PR branch already preserved in main:',branch)
    return deleted


def main():
    if os.environ.get('GITHUB_EVENT_NAME')!='workflow_run' or os.environ.get('GITHUB_REF')!='refs/heads/main':
        raise RuntimeError('Branch cleanup runs only after successful main push checks')
    event=json.loads(Path(os.environ['GITHUB_EVENT_PATH']).read_text())
    run=event.get('workflow_run',{})
    if (run.get('name')!='Koharu CI and source' or run.get('event')!='push'
            or run.get('head_branch')!='main' or run.get('conclusion')!='success'
            or run.get('head_repository',{}).get('full_name')!=os.environ['GITHUB_REPOSITORY']):
        raise RuntimeError('Branch cleanup requires successful owned main push checks')
    repository=os.environ['GITHUB_REPOSITORY']
    base='https://api.github.com/repos/'+repository
    def request(path,method='GET'):
        req=urllib.request.Request(base+path,method=method,headers={
            'Authorization':'Bearer '+os.environ['GH_TOKEN'],
            'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
        with urllib.request.urlopen(req,timeout=30) as response:
            data=response.read()
        return json.loads(data) if data else None
    cleanup(request,repository,run['head_sha'])


if __name__=='__main__':
    main()
