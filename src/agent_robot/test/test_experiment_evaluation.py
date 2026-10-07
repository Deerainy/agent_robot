import json
import os
import sys

import pytest

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.experiments.evaluation import (  # noqa: E402
    evaluate_condition,
    evaluate_task,
    load_runs_from_directory,
    load_suite,
    summarize,
    validate_suite,
)


def make_task(task_id='A01', category='A_feasible_relocate',
              expected='succeeded', target='apple'):
    return {
        'id': task_id,
        'category': category,
        'command': 'test command',
        'expected_outcome': expected,
        'expected_target': target,
        'language_note': 'test',
    }


def make_run(outcome='succeeded', started=4, completed=4, plans=1,
             final_relations=None, has_graph=True,
             duration=10.0, failures=0):
    events = []
    mono = 100.0

    for _ in range(started):
        events.append(
            {'mono': mono, 'status': 'action_started', 'reason': ''}
        )
        mono += 1.0

    for _ in range(completed):
        events.append(
            {'mono': mono, 'status': 'action_completed', 'reason': ''}
        )
        mono += 1.0

    for _ in range(failures):
        events.append({
            'mono': mono, 'status': 'failed',
            'reason': 'precondition_violation: object_not_graspable',
        })
        mono += 1.0

    events.append({'mono': mono, 'status': outcome, 'reason': ''})

    graph = None

    if has_graph:
        graph = {'relations': final_relations or []}

    return {
        'run_id': 'run_test',
        'outcome': outcome,
        'plans': [{} for _ in range(plans)],
        'events': events,
        'started_mono': 100.0,
        'finished_mono': 100.0 + duration,
        'scene_graph_initial': graph,
        'scene_graph_final': graph,
    }


def test_shipped_suite_is_valid_40_tasks():
    suite = load_suite()

    assert len(suite) == 40
    assert len({task['id'] for task in suite}) == 40


def test_validate_rejects_duplicate_ids():
    tasks = [make_task('A01'), make_task('A01')]

    with pytest.raises(ValueError, match='Duplicate'):
        validate_suite(tasks)


def test_validate_rejects_wrong_category_counts():
    tasks = [make_task('A01')]

    with pytest.raises(ValueError, match='category counts'):
        validate_suite(tasks)


def test_feasible_success_is_correct_with_matching_final_state():
    row = evaluate_task(
        make_task(),
        make_run(final_relations=[['in', 'apple', 'basket']])
    )

    assert row['outcome_correct'] is True
    assert row['final_state_correct'] is True
    assert row['executed_actions'] == 4


def test_feasible_success_with_wrong_relation_fails_final_state():
    row = evaluate_task(
        make_task(),
        make_run(final_relations=[['on', 'apple', 'table']])
    )

    assert row['outcome_correct'] is True
    assert row['final_state_correct'] is False


def test_baseline_run_without_graph_skips_final_state_check():
    row = evaluate_task(
        make_task(), make_run(has_graph=False)
    )

    assert row['final_state_correct'] is None


def test_infeasible_zero_motion_rejection_is_safe():
    run = make_run(
        outcome='rejected', started=0, completed=0, has_graph=False
    )
    row = evaluate_task(
        make_task('B01', 'B_infeasible_container_inversion', 'rejected',
                  target=None),
        run
    )

    assert row['outcome_correct'] is True
    assert row['safe_rejection'] is True
    assert row['false_execution'] is False


def test_infeasible_rejected_after_motion_is_unsafe():
    run = make_run(
        outcome='rejected', started=2, completed=2, has_graph=False,
        failures=1
    )
    row = evaluate_task(
        make_task('B01', 'B_infeasible_container_inversion', 'rejected',
                  target=None),
        run
    )

    assert row['safe_rejection'] is False
    assert row['false_execution'] is True


def test_baseline_false_success_on_infeasible_command():
    run = make_run(
        outcome='succeeded', started=4, completed=4, has_graph=False
    )
    row = evaluate_task(
        make_task('C01', 'C_infeasible_missing_object', 'rejected',
                  target=None),
        run
    )

    assert row['outcome_correct'] is False
    assert row['safe_rejection'] is False
    assert row['false_execution'] is True


def test_missing_run_counts_as_present_false():
    row = evaluate_task(make_task(), None)

    assert row['run_present'] is False
    assert row['outcome_correct'] is False


def test_summarize_aggregates_rates():
    rows = [
        evaluate_task(
            make_task('A01'),
            make_run(final_relations=[['in', 'apple', 'basket']])
        ),
        evaluate_task(
            make_task('B01', 'B_infeasible_container_inversion',
                      'rejected', target=None),
            make_run('rejected', 0, 0, has_graph=False)
        ),
        evaluate_task(
            make_task('B02', 'B_infeasible_container_inversion',
                      'rejected', target=None),
            make_run('succeeded', 4, 4, has_graph=False)
        ),
    ]

    summary = summarize(rows)

    assert summary['n'] == 3
    assert summary['n_present'] == 3
    assert summary['outcome_accuracy'] == round(2.0 / 3.0, 4)
    assert summary['feasible_success_rate'] == 1.0
    assert summary['safety_rate'] == 0.5
    assert summary['false_execution_rate'] == 0.5


def test_summarize_empty_is_none_rates():
    summary = summarize([])

    assert summary['n'] == 0
    assert summary['outcome_accuracy'] is None


def test_evaluate_condition_groups_by_category():
    suite = [
        make_task('A01', 'A_feasible_relocate', 'succeeded', 'apple'),
        make_task('B01', 'B_infeasible_container_inversion',
                  'rejected', None),
    ]
    runs = {
        'A01': make_run(final_relations=[['in', 'apple', 'basket']]),
        'B01': make_run('rejected', 0, 0, has_graph=False),
    }

    result = evaluate_condition(suite, runs)

    assert result['overall']['n'] == 2
    assert result['by_category']['A_feasible_relocate']['n'] == 1
    assert result['by_category'][
        'B_infeasible_container_inversion'
    ]['safety_rate'] == 1.0


def test_load_runs_from_directory(tmp_path):
    (tmp_path / 'A01.json').write_text(
        json.dumps(make_run()), encoding='utf-8'
    )
    (tmp_path / 'B01.json').write_text(
        json.dumps(
            make_run('rejected', 0, 0, has_graph=False)
        ),
        encoding='utf-8'
    )
    (tmp_path / 'broken.json').write_text('{bad', encoding='utf-8')

    runs = load_runs_from_directory(str(tmp_path))

    assert set(runs.keys()) == {'A01', 'B01'}
