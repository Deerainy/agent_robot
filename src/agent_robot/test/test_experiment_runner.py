import os
import sys

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

import pytest  # noqa: E402

from agent_robot.experiments.batch_runner import (  # noqa: E402
    ReplaySequencer,
    observed_outcome,
    parse_task_selection,
)


def rejected_plan(command='x'):
    return {
        'command': command, 'feasible': False, 'actions': [],
        'planner': 'fallback', 'replans': 0,
    }


def accepted_plan(command='x'):
    return {
        'command': command, 'feasible': True,
        'actions': [{'skill': 'move_to', 'object': 'apple'}],
        'planner': 'deepseek', 'replans': 0,
    }


def test_sequencer_emits_plans_in_order():
    sequencer = ReplaySequencer([
        accepted_plan(), rejected_plan()
    ])

    assert sequencer.initial_plan()['feasible'] is True
    assert sequencer.plan_after_failure()['feasible'] is False
    assert sequencer.plan_after_failure() is None


def test_sequencer_rejects_empty_cache():
    with pytest.raises(ValueError, match='at least one plan'):
        ReplaySequencer([])


def test_sequencer_single_plan_has_no_replan():
    sequencer = ReplaySequencer([rejected_plan()])

    assert sequencer.initial_plan()['feasible'] is False
    assert sequencer.plan_after_failure() is None


def test_observed_outcome_terminal_succeeded():
    statuses = [
        {'status': 'action_started'},
        {'status': 'action_completed'},
        {'status': 'succeeded', 'reason': 'done'},
    ]

    assert observed_outcome(statuses) == ('succeeded', 'done')


def test_observed_outcome_rejected_takes_precedence_over_early_failed():
    statuses = [
        {'status': 'failed', 'reason': 'precondition_violation: x'},
        {'status': 'rejected', 'reason': 'final reject'},
    ]

    assert observed_outcome(statuses) == ('rejected', 'final reject')


def test_observed_outcome_failed_without_terminal():
    statuses = [
        {'status': 'action_started'},
        {'status': 'failed', 'reason': 'gripper error'},
    ]

    assert observed_outcome(statuses) == ('failed', 'gripper error')


def test_observed_outcome_empty_stream():
    assert observed_outcome([]) == (None, '')


def test_parse_task_selection_defaults_to_whole_suite():
    suite = [{'id': 'A01'}, {'id': 'B01'}]

    assert parse_task_selection(None, suite) == suite


def test_parse_task_selection_filters_and_rejects_unknown():
    suite = [{'id': 'A01'}, {'id': 'B01'}, {'id': 'C01'}]

    assert [task['id'] for task in parse_task_selection(
        ' C01 ,A01 ', suite
    )] == ['A01', 'C01']

    with pytest.raises(ValueError, match='Unknown task ids'):
        parse_task_selection('ZZ9', suite)
