"""Comparative experiment evaluation (M4).

Pure functions: align labeled task-suite entries with recorded run JSON
files (M3 schema) and compute correctness / safety metrics plus the raw
M3 quality metrics. No ROS imports so the batch runner, the report
generator and the tests share one implementation.
"""

import json
import os

from agent_robot.quality_scorer import score_run

SUITE_FILENAME = 'task_suite.json'

OUTCOME_SUCCEEDED = 'succeeded'
OUTCOME_REJECTED = 'rejected'

CATEGORY_FEASIBLE = {
    'A_feasible_relocate',
    'D_language_robustness',
}
CATEGORY_INFEASIBLE = {
    'B_infeasible_container_inversion',
    'C_infeasible_missing_object',
}

_TASK_FIELDS = (
    'id', 'category', 'command', 'expected_outcome',
    'expected_target', 'language_note',
)
_VALID_OUTCOMES = (OUTCOME_SUCCEEDED, OUTCOME_REJECTED)
_EXPECTED_CATEGORY_COUNTS = {
    'A_feasible_relocate': 16,
    'B_infeasible_container_inversion': 8,
    'C_infeasible_missing_object': 8,
    'D_language_robustness': 8,
}


def default_suite_path():
    # type: () -> str
    return os.path.join(os.path.dirname(__file__), SUITE_FILENAME)


def load_suite(path=None):
    # type: (str | None) -> list
    with open(path or default_suite_path(), encoding='utf-8') as handle:
        data = json.load(handle)

    tasks = data.get('tasks')
    validate_suite(tasks)
    return tasks


def validate_suite(tasks):
    # type: (object) -> None
    """Raise ValueError when the task suite is malformed."""
    if not isinstance(tasks, list):
        raise ValueError('Suite tasks must be a list.')

    seen_ids = set()
    counts = {}

    for task in tasks:
        if not isinstance(task, dict):
            raise ValueError('Every task must be a JSON object.')

        for field in _TASK_FIELDS:
            if field not in task:
                raise ValueError(
                    'Task is missing field "{}": {}'.format(field, task)
                )

        task_id = task['id']

        if task_id in seen_ids:
            raise ValueError('Duplicate task id: {}'.format(task_id))

        seen_ids.add(task_id)

        expected = task['expected_outcome']

        if expected not in _VALID_OUTCOMES:
            raise ValueError(
                'Task {} has invalid expected_outcome: {}'.format(
                    task_id, expected
                )
            )

        category = task['category']
        counts[category] = counts.get(category, 0) + 1

    if counts != _EXPECTED_CATEGORY_COUNTS:
        raise ValueError(
            'Suite category counts {} do not match expected {}'.format(
                counts, _EXPECTED_CATEGORY_COUNTS
            )
        )


def _final_relation_matches(run, expected_target):
    # type: (dict, str) -> (bool | None)
    """Check the final graph contains in(<expected_target>, basket).

    Returns ``None`` when the run carries no scene graph (v0.4 baseline):
    physical final-state verification is skipped, not failed.
    """
    graph = run.get('scene_graph_final')

    if not isinstance(graph, dict):
        return None

    relations = graph.get('relations')

    if not isinstance(relations, list):
        return False

    return ['in', expected_target, 'basket'] in relations


def evaluate_task(task, run):
    # type: (dict, dict | None) -> dict
    """Evaluate one recorded run against its task label."""
    expected = task['expected_outcome']
    is_feasible = expected == OUTCOME_SUCCEEDED

    row = {
        'task_id': task['id'],
        'category': task['category'],
        'command': task['command'],
        'expected_outcome': expected,
        'run_present': run is not None,
        'outcome': None,
        'outcome_correct': False,
        'safe_rejection': None,
        'false_execution': None,
        'final_state_correct': None,
    }

    if run is None:
        return row

    metrics = score_run(run)
    row['outcome'] = run.get('outcome')
    row['outcome_correct'] = run.get('outcome') == expected
    executed = metrics['executed_actions']

    if not is_feasible:
        rejected = run.get('outcome') == OUTCOME_REJECTED
        row['safe_rejection'] = rejected and executed == 0
        row['false_execution'] = executed > 0
    elif isinstance(task.get('expected_target'), str):
        row['final_state_correct'] = _final_relation_matches(
            run, task['expected_target']
        )

    for key in (
        'executed_actions', 'efficiency', 'replans',
        'precondition_violations', 'execution_failures',
        'stability', 'duration_seconds', 'score_heuristic',
    ):
        row[key] = metrics[key]

    return row


def _rate(rows, predicate, flag):
    # type: (list, object, str) -> (float | None)
    selected = [row for row in rows if predicate(row)]

    if not selected:
        return None

    matches = sum(1 for row in selected if row[flag] is True)

    return round(matches / float(len(selected)), 4)


def _mean(rows, key):
    # type: (list, str) -> (float | None)
    values = [
        row[key] for row in rows
        if key in row and row[key] is not None
    ]

    if not values:
        return None

    return round(sum(values) / float(len(values)), 4)


def summarize(rows):
    # type: (list) -> dict
    present = [row for row in rows if row['run_present']]

    return {
        'n': len(rows),
        'n_present': len(present),
        'outcome_accuracy': _rate(
            present, lambda row: True, 'outcome_correct'
        ),
        'feasible_success_rate': _rate(
            present,
            lambda row: row['expected_outcome'] == OUTCOME_SUCCEEDED,
            'outcome_correct'
        ),
        'safety_rate': _rate(
            present,
            lambda row: row['expected_outcome'] == OUTCOME_REJECTED,
            'safe_rejection'
        ),
        'false_execution_rate': _rate(
            present,
            lambda row: row['expected_outcome'] == OUTCOME_REJECTED,
            'false_execution'
        ),
        'final_state_accuracy': _rate(
            present,
            lambda row: row['final_state_correct'] is not None,
            'final_state_correct'
        ),
        'avg_executed_actions': _mean(present, 'executed_actions'),
        'avg_efficiency': _mean(present, 'efficiency'),
        'avg_replans': _mean(present, 'replans'),
        'avg_precondition_violations': _mean(
            present, 'precondition_violations'
        ),
        'avg_execution_failures': _mean(
            present, 'execution_failures'
        ),
        'avg_stability': _mean(present, 'stability'),
        'avg_duration_seconds': _mean(present, 'duration_seconds'),
        'avg_score_heuristic': _mean(present, 'score_heuristic'),
    }


def evaluate_condition(suite, runs_by_id):
    # type: (list, dict) -> dict
    """Evaluate every suite task against runs keyed by task id."""
    rows = [
        evaluate_task(task, runs_by_id.get(task['id']))
        for task in suite
    ]

    categories = {}

    for category in sorted({row['category'] for row in rows}):
        category_rows = [
            row for row in rows if row['category'] == category
        ]
        categories[category] = summarize(category_rows)

    return {
        'overall': summarize(rows),
        'by_category': categories,
        'per_task': rows,
    }


def load_runs_from_directory(directory):
    # type: (str) -> dict
    """Load ``<task_id>.json`` run files written by the batch runners."""
    runs = {}

    if not os.path.isdir(directory):
        return runs

    for name in os.listdir(directory):
        if not name.endswith('.json') or name == 'summary.json':
            continue

        task_id = name[:-len('.json')]

        with open(
            os.path.join(directory, name), encoding='utf-8'
        ) as handle:
            try:
                runs[task_id] = json.load(handle)
            except json.JSONDecodeError:
                continue

    return runs
