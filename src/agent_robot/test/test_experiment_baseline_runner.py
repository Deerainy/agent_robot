"""Unit tests for baseline_runner pure functions."""

import json
import os
import tempfile

from agent_robot.experiments.baseline_runner import (
    baseline_outcome,
    build_baseline_run,
    collect_run_file,
)


def test_baseline_outcome_succeeded_when_last_step_complete():
    statuses = [
        {'status': 'completed', 'step_index': 1, 'total_steps': 3},
        {'status': 'completed', 'step_index': 2, 'total_steps': 3},
        {'status': 'completed', 'step_index': 3, 'total_steps': 3},
    ]

    outcome, reason = baseline_outcome(statuses)

    assert outcome == 'succeeded'
    assert reason == ''


def test_baseline_outcome_rejected():
    statuses = [
        {'status': 'rejected', 'reason': 'not feasible'},
    ]

    outcome, reason = baseline_outcome(statuses)

    assert outcome == 'rejected'
    assert reason == 'not feasible'


def test_baseline_outcome_failed_when_no_terminal():
    statuses = [
        {'status': 'completed', 'step_index': 1, 'total_steps': 3},
        {'status': 'failed', 'reason': 'collision', 'failed_step': 'pick'},
    ]

    outcome, reason = baseline_outcome(statuses)

    assert outcome == 'failed'
    assert reason == 'collision'


def test_baseline_outcome_none_when_empty():
    outcome, reason = baseline_outcome([])

    assert outcome is None
    assert reason == ''


def test_build_baseline_run_succeeded_expands_steps():
    plans = [{
        'command': 'put apple in basket',
        'feasible': True,
        'steps': ['move', 'pick', 'place'],
        'planner': 'deepseek',
        'reason': '',
    }]
    statuses = [
        {'status': 'completed', 'step_index': 1, 'total_steps': 3,
         'step': 'move'},
        {'status': 'completed', 'step_index': 2, 'total_steps': 3,
         'step': 'pick'},
        {'status': 'completed', 'step_index': 3, 'total_steps': 3,
         'step': 'place'},
    ]

    run = build_baseline_run('put apple in basket', plans, statuses,
                             'succeeded', '')

    assert run['outcome'] == 'succeeded'
    assert len(run['plans']) == 1
    assert run['plans'][0]['actions'] == [
        {'skill': 'step', 'description': 'move'},
        {'skill': 'step', 'description': 'pick'},
        {'skill': 'step', 'description': 'place'},
    ]
    types = [e['status'] for e in run['events']]
    assert types == [
        'action_started', 'action_completed',
        'action_started', 'action_completed',
        'action_started', 'action_completed',
        'succeeded',
    ]
    assert run['scene_graph_final'] is None


def test_build_baseline_run_rejected_has_no_actions():
    plans = [{
        'command': 'put banana in basket',
        'feasible': False,
        'steps': [],
        'planner': 'fallback',
        'reason': 'banana not in scene',
    }]
    statuses = [
        {'status': 'rejected', 'reason': 'banana not in scene'},
    ]

    run = build_baseline_run('put banana in basket', plans, statuses,
                             'rejected', 'banana not in scene')

    assert run['outcome'] == 'rejected'
    assert run['plans'][0]['actions'] == []
    assert [e['status'] for e in run['events']] == ['rejected']


def test_collect_run_file_writes_json():
    with tempfile.TemporaryDirectory() as tmp:
        cond = os.path.join(tmp, 'baseline')
        os.makedirs(cond)
        run = {'run_id': 'x', 'outcome': 'succeeded', 'events': []}

        path = collect_run_file(tmp, cond, 'A01', run)

        assert path == os.path.join(cond, 'A01.json')
        with open(path) as handle:
            assert json.load(handle)['outcome'] == 'succeeded'
