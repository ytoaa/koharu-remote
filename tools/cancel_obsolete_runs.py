"""Cancel only older builds of this workflow and branch (migration to concurrency)."""
import json
import os
import time
import urllib.parse
import urllib.request
import urllib.error


def obsolete(current, candidate):
    return (candidate['id'] != current['id']
            and candidate['workflow_id'] == current['workflow_id']
            and candidate['head_branch'] == current['head_branch']
            and candidate['event'] == current['event']
            and candidate.get('head_repository',{}).get('id') == current.get('head_repository',{}).get('id')
            and candidate['run_number'] < current['run_number']
            and candidate['status'] in ('queued','in_progress','waiting','pending','requested'))


def main():
    repo = os.environ['GITHUB_REPOSITORY']
    base = 'https://api.github.com/repos/'+repo+'/actions'
    def request(path, method='GET'):
        req = urllib.request.Request(base+path,method=method,headers={
            'Authorization':'Bearer '+os.environ['GH_TOKEN'],
            'Accept':'application/vnd.github+json','X-GitHub-Api-Version':'2022-11-28'})
        with urllib.request.urlopen(req,timeout=30) as response:
            data=response.read()
        return json.loads(data) if data else None
    current = request('/runs/'+os.environ['GITHUB_RUN_ID'])
    for page in range(1,6):
        query = urllib.parse.urlencode({'branch':current['head_branch'],'per_page':100,'page':page})
        runs = request('/workflows/'+str(current['workflow_id'])+'/runs?'+query)['workflow_runs']
        for candidate in runs:
            if obsolete(current,candidate):
                try:
                    request('/runs/'+str(candidate['id'])+'/cancel',method='POST')
                except urllib.error.HTTPError as error:
                    if error.code != 409: raise
                    # It may already be finishing a previous cancellation request.
                print('CANCEL_REQUESTED_FOR_OBSOLETE_RUN:',candidate['id'],flush=True)
                time.sleep(2)
                latest=request('/runs/'+str(candidate['id']))
                if latest['status'] != 'completed':
                    try:
                        request('/runs/'+str(candidate['id'])+'/force-cancel',method='POST')
                    except urllib.error.HTTPError as error:
                        if error.code != 409: raise
                    print('FORCE_CANCEL_REQUESTED:',candidate['id'],flush=True)
        if len(runs)<100: break


if __name__ == '__main__':
    main()
