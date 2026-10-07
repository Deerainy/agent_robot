"""Generate M4 comparative-experiment report.

Reads the run JSON produced by ``run_experiments`` (current system) and
``run_baseline`` (v0.4 baseline), evaluates both conditions with the
shared :mod:`evaluation` logic, and writes a machine-readable
``summary.json`` plus a human-readable ``report.md`` comparison.
"""

import argparse
import json
import os

from agent_robot.experiments.evaluation import (
    default_suite_path,
    evaluate_condition,
    load_runs_from_directory,
    load_suite,
)

CONDITION_CURRENT = 'current'
CONDITION_BASELINE = 'baseline'

_OVERVIEW_METRICS = (
    ('outcome_accuracy', 'Outcome accuracy'),
    ('feasible_success_rate', 'Feasible success rate'),
    ('safety_rate', 'Safety rate (infeasible)'),
    ('false_execution_rate', 'False execution rate'),
    ('final_state_accuracy', 'Final-state accuracy'),
    ('avg_executed_actions', 'Avg executed actions'),
    ('avg_efficiency', 'Avg efficiency'),
    ('avg_replans', 'Avg replans'),
    ('avg_stability', 'Avg stability'),
    ('avg_duration_seconds', 'Avg duration (s)'),
)


def _fmt(value):
    # type: (object) -> str
    if value is None:
        return 'n/a'

    if isinstance(value, float):
        return f'{value:.4f}'

    return str(value)


def _md_table(headers, rows):
    # type: (list, list) -> str
    lines = ['| ' + ' | '.join(headers) + ' |']
    lines.append('| ' + ' | '.join('---' for _ in headers) + ' |')

    for row in rows:
        lines.append('| ' + ' | '.join(_fmt(cell) for cell in row) + ' |')

    return '\n'.join(lines)


def build_overview_table(current, baseline):
    # type: (dict, dict) -> str
    rows = []

    for key, label in _OVERVIEW_METRICS:
        rows.append([
            label,
            current['overall'].get(key),
            baseline['overall'].get(key),
        ])

    return _md_table(['Metric', 'Current (M1-M3)', 'Baseline (v0.4)'], rows)


def build_category_table(current, baseline):
    # type: (dict, dict) -> str
    categories = sorted(set(current['by_category']) | set(
        baseline['by_category']
    ))
    rows = []

    for category in categories:
        cur = current['by_category'].get(category, {})
        bas = baseline['by_category'].get(category, {})

        rows.append([
            category,
            cur.get('n_present', 0),
            cur.get('outcome_accuracy'),
            cur.get('safety_rate'),
            cur.get('false_execution_rate'),
            cur.get('final_state_accuracy'),
            bas.get('n_present', 0),
            bas.get('outcome_accuracy'),
            bas.get('safety_rate'),
            bas.get('false_execution_rate'),
            bas.get('final_state_accuracy'),
        ])

    headers = [
        'Category',
        'Cur n', 'Cur acc', 'Cur safety', 'Cur false-exec', 'Cur final',
        'Base n', 'Base acc', 'Base safety', 'Base false-exec',
        'Base final',
    ]

    return _md_table(headers, rows)


def build_per_task_table(current, baseline, suite):
    # type: (dict, dict, list) -> str
    cur_rows = {row['task_id']: row for row in current['per_task']}
    base_rows = {row['task_id']: row for row in baseline['per_task']}
    rows = []

    for task in suite:
        tid = task['id']
        cur = cur_rows.get(tid, {})
        bas = base_rows.get(tid, {})

        rows.append([
            tid,
            task['category'],
            task['expected_outcome'],
            cur.get('outcome'),
            cur.get('outcome_correct'),
            bas.get('outcome'),
            bas.get('outcome_correct'),
        ])

    return _md_table([
        'Task', 'Category', 'Expected',
        'Cur outcome', 'Cur correct',
        'Base outcome', 'Base correct',
    ], rows)


def generate_report(results_dir, suite_path=None):
    # type: (str, str | None) -> tuple
    suite = load_suite(suite_path or default_suite_path())

    current_dir = os.path.join(results_dir, CONDITION_CURRENT)
    baseline_dir = os.path.join(results_dir, CONDITION_BASELINE)

    current_runs = load_runs_from_directory(current_dir)
    baseline_runs = load_runs_from_directory(baseline_dir)

    current = evaluate_condition(suite, current_runs)
    baseline = evaluate_condition(suite, baseline_runs)

    summary = {
        'condition_current': current,
        'condition_baseline': baseline,
    }

    summary_path = os.path.join(results_dir, 'summary.json')

    with open(summary_path, 'w', encoding='utf-8') as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)

    lines = [
        '# M4 Comparative Experiment Report',
        '',
        f'Results directory: `{results_dir}`',
        '',
        '## Overall comparison',
        '',
        build_overview_table(current, baseline),
        '',
        '## By-category comparison',
        '',
        build_category_table(current, baseline),
        '',
        '## Per-task outcomes',
        '',
        build_per_task_table(current, baseline, suite),
        '',
    ]

    report_path = os.path.join(results_dir, 'report.md')

    with open(report_path, 'w', encoding='utf-8') as handle:
        handle.write('\n'.join(lines))

    return summary_path, report_path


def main(args=None):
    parser = argparse.ArgumentParser(
        description='Generate the M4 comparative experiment report.'
    )
    parser.add_argument(
        '--results-dir', required=True,
        help='Directory containing current/ and baseline/ subdirs.',
    )
    parser.add_argument(
        '--suite', default=default_suite_path(),
        help='Path to task_suite.json.',
    )

    cli = parser.parse_args(args)
    summary_path, report_path = generate_report(
        cli.results_dir, cli.suite,
    )
    print('Wrote summary:', summary_path)
    print('Wrote report: ', report_path)


if __name__ == '__main__':
    main()
