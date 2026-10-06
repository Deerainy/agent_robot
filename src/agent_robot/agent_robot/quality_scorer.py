"""Quality scoring for recorded execution trajectories (M3).

Pure functions over the run JSON records written by the recorder; no ROS
imports. The composite score is an explicitly **heuristic** auxiliary
metric: M4 comparisons rely on the raw metrics (success rate, executed
actions, replans, violations, failures, duration).

Definitions (approved M3 plan §2.3):

- ``executed_actions``: number of ``action_started`` events;
- ``efficiency``: ``min(1, optimal / executed)`` against a task-template
  optimum (relocate commands = 4 actions); ``None`` when the template is
  unknown, ``0.0`` when nothing was executed;
- ``stability``: ``1 / (1 + replans + violations + failures)`` where
  ``replans = len(plans) - 1``, violations are failed events whose reason
  contains ``precondition_violation`` and failures are the remaining
  failed events;
- ``duration_seconds``: ``finished_mono - first started mono`` (monotonic
  clock, immune to wall-clock adjustments);
- ``score_heuristic``: 0~100 weighted composite (outcome 0.6, efficiency
  0.25, stability 0.15); missing dimensions are renormalized.
"""

import glob
import json
import os

SCORER_VERSION = '1.0'

OUTCOME_SUCCEEDED = 'succeeded'

_OPTIMAL_STEPS_RELOCATE = 4
_RELOCATE_MARKERS = ('放进', '放入', '放到', '放在', '搬')
_PRECONDITION_MARKER = 'precondition_violation'

_WEIGHT_OUTCOME = 0.6
_WEIGHT_EFFICIENCY = 0.25
_WEIGHT_STABILITY = 0.15


def executed_actions(run):
    # type: (dict) -> int
    return sum(
        1 for event in run.get('events', [])
        if event.get('status') == 'action_started'
    )


def replan_count(run):
    # type: (dict) -> int
    return max(0, len(run.get('plans', [])) - 1)


def violation_count(run):
    # type: (dict) -> int
    return sum(
        1 for event in run.get('events', [])
        if event.get('status') == 'failed'
        and _PRECONDITION_MARKER in str(event.get('reason', ''))
    )


def execution_failure_count(run):
    # type: (dict) -> int
    return sum(
        1 for event in run.get('events', [])
        if event.get('status') == 'failed'
        and _PRECONDITION_MARKER not in str(event.get('reason', ''))
    )


def duration_seconds(run):
    # type: (dict) -> (float | None)
    starts = [
        event.get('mono') for event in run.get('events', [])
        if event.get('status') == 'action_started'
        and isinstance(event.get('mono'), (int, float))
    ]

    finished = run.get('finished_mono')

    if not starts or not isinstance(finished, (int, float)):
        return None

    return max(0.0, finished - min(starts))


def optimal_steps(command):
    # type: (str) -> (int | None)
    """Task-template optimum; ``None`` for unrecognized commands."""
    if any(marker in command for marker in _RELOCATE_MARKERS):
        return _OPTIMAL_STEPS_RELOCATE

    return None


def score_run(run):
    # type: (dict) -> dict
    """Score one run record; see module docstring for definitions."""
    outcome = run.get('outcome')
    executed = executed_actions(run)
    optimal = optimal_steps(run.get('command', ''))

    if optimal is None:
        efficiency = None
    elif executed == 0:
        efficiency = 0.0
    else:
        efficiency = min(1.0, optimal / float(executed))

    stability = 1.0 / (
        1.0 + replan_count(run)
        + violation_count(run)
        + execution_failure_count(run)
    )

    outcome_score = 1.0 if outcome == OUTCOME_SUCCEEDED else 0.0

    components = [
        (_WEIGHT_OUTCOME, outcome_score),
        (_WEIGHT_EFFICIENCY, efficiency),
        (_WEIGHT_STABILITY, stability),
    ]
    available = [(w, v) for w, v in components if v is not None]
    total_weight = sum(w for w, _ in available)
    weighted = sum(w * v for w, v in available)

    return {
        'scorer_version': SCORER_VERSION,
        'run_id': run.get('run_id'),
        'outcome': outcome,
        'executed_actions': executed,
        'optimal_steps': optimal,
        'efficiency': efficiency,
        'replans': replan_count(run),
        'precondition_violations': violation_count(run),
        'execution_failures': execution_failure_count(run),
        'stability': stability,
        'duration_seconds': duration_seconds(run),
        'outcome_score': outcome_score,
        # Heuristic auxiliary metric only; M4 uses the raw metrics above.
        'score_heuristic': round(100.0 * weighted / total_weight, 1),
    }


def _mean(values):
    # type: (list) -> (float | None)
    values = [v for v in values if v is not None]

    if not values:
        return None

    return round(sum(values) / float(len(values)), 4)


def aggregate(scores):
    # type: (list) -> dict
    """Aggregate per-run score dicts into a comparison-ready summary."""
    if not scores:
        return {
            'scorer_version': SCORER_VERSION,
            'n': 0,
            'success_rate': None,
            'avg_executed_actions': None,
            'avg_efficiency': None,
            'avg_replans': None,
            'avg_precondition_violations': None,
            'avg_execution_failures': None,
            'avg_stability': None,
            'avg_duration_seconds': None,
            'avg_score_heuristic': None,
        }

    return {
        'scorer_version': SCORER_VERSION,
        'n': len(scores),
        'success_rate': _mean([
            1.0 if score['outcome'] == OUTCOME_SUCCEEDED else 0.0
            for score in scores
        ]),
        'avg_executed_actions': _mean([
            score['executed_actions'] for score in scores
        ]),
        'avg_efficiency': _mean([
            score['efficiency'] for score in scores
        ]),
        'avg_replans': _mean([score['replans'] for score in scores]),
        'avg_precondition_violations': _mean([
            score['precondition_violations'] for score in scores
        ]),
        'avg_execution_failures': _mean([
            score['execution_failures'] for score in scores
        ]),
        'avg_stability': _mean([score['stability'] for score in scores]),
        'avg_duration_seconds': _mean([
            score['duration_seconds'] for score in scores
        ]),
        'avg_score_heuristic': _mean([
            score['score_heuristic'] for score in scores
        ]),
    }


def score_directory(directory):
    # type: (str) -> (list, dict)
    """Score every ``*.json`` run record in *directory*."""
    runs = []

    for path in sorted(glob.glob(os.path.join(directory, '*.json'))):
        with open(path, encoding='utf-8') as handle:
            try:
                run = json.load(handle)
            except json.JSONDecodeError:
                continue

        if isinstance(run, dict) and run.get('run_id'):
            runs.append(run)

    scores = [score_run(run) for run in runs]

    return scores, aggregate(scores)


def main(args=None):
    import sys

    argv = sys.argv[1:] if args is None else args

    directory = os.path.expanduser(
        argv[0] if argv else '~/ros2_ws/trajectories'
    )

    scores, summary = score_directory(directory)

    for score in scores:
        print(
            '{run_id}  outcome={outcome}  executed={executed}  '
            'efficiency={efficiency}  stability={stability}  '
            'heuristic={heuristic}'.format(
                run_id=score['run_id'],
                outcome=score['outcome'],
                executed=score['executed_actions'],
                efficiency=score['efficiency'],
                stability=score['stability'],
                heuristic=score['score_heuristic'],
            )
        )

    print('--- summary ---')
    print(json.dumps(summary, indent=2, ensure_ascii=False))

    summary_path = os.path.join(directory, 'summary.json')

    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(
            {'scores': scores, 'summary': summary},
            handle,
            indent=2,
            ensure_ascii=False
        )

    print('Wrote {}'.format(summary_path))


if __name__ == '__main__':
    main()
