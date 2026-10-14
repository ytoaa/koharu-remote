from tools.cancel_obsolete_runs import obsolete


def test_cancellation_only_targets_older_active_runs_on_same_branch_and_workflow():
    current={'id':20,'workflow_id':8,'head_branch':'feature','event':'pull_request','run_number':20,'status':'in_progress'}
    previous={**current,'id':19,'run_number':19}
    assert obsolete(current,previous)
    for change in ({'id':20},{'run_number':21},{'workflow_id':9},{'head_branch':'main'},
                   {'event':'push'},{'status':'completed'}):
        assert not obsolete(current,{**previous,**change})
