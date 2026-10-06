import json
import os
import sys

# Allow running `python3 -m pytest test/test_quality_scorer.py` from the
# package root without a colcon/ROS install step.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.quality_scorer import (  # noqa: E402
    aggregate,
    score_directory,
    score_run,
)


def make_run(command='把红色苹果放进篮子里', outcome='succeeded',
             plans=1, started=4, completed=4,
             precondition_failures=0, other_failures=0,
             duration=10.0, run_id='run_test_0001'):
    events = []
    mono = 100.0

    for _ in range(started):
        events.append({
            't': mono, 'mono': mono, 'status': 'action_started',
            'action': None, 'step_index': 1, 'total_steps': 4, 'reason': '',
        })
        mono += 1.0

    for _ in range(completed):
        events.append({
            't': mono, 'mono': mono, 'status': 'action_completed',
            'action': None, 'step_index': 1, 'total_steps': 4, 'reason': '',
        })
        mono += 1.0

    for _ in range(precondition_failures):
        events.append({
            't': mono, 'mono': mono, 'status': 'failed',
            'reason': 'precondition_violation: object_not_graspable: x',
        })
        mono += 1.0

    for _ in range(other_failures):
        events.append({
            't': mono, 'mono': mono, 'status': 'failed',
            'reason': '动作 pick(apple) 执行失败：gripper error',
        })
        mono += 1.0

    if outcome == 'succeeded':
        events.append({
            't': mono, 'mono': mono, 'status': 'succeeded',
            'reason': 'All 4 action(s) completed.',
        })

    return {
        'schema_version': '1.0',
        'run_id': run_id,
        'command': command,
        'started_mono': 100.0,
        'finished_mono': 100.0 + duration,
        'plans': [
            {
                'feasible': True, 'actions': [], 'planner': 'deepseek',
                'replans': index, 'reason': '',
            }
            for index in range(plans)
        ],
        'events': events,
        'outcome': outcome,
        'outcome_reason': '',
        'scene_graph_initial': None,
        'scene_graph_final': None,
    }


def test_succeeded_run_scores_full():
    score = score_run(make_run())

    assert score['outcome_score'] == 1.0
    assert score['executed_actions'] == 4
    assert score['optimal_steps'] == 4
    assert score['efficiency'] == 1.0
    assert score['stability'] == 1.0
    assert score['duration_seconds'] == 10.0
    assert score['score_heuristic'] == 100.0


def test_executed_counts_action_started_only():
    run = make_run(started=5, completed=3)

    assert score_run(run)['executed_actions'] == 5


def test_replan_and_violation_reduce_stability():
    score = score_run(make_run(plans=2, precondition_failures=1))

    # 1 / (1 + 1 replan + 1 violation + 0 failures)
    assert abs(score['stability'] - 1.0 / 3.0) < 1e-9
    assert score['replans'] == 1
    assert score['precondition_violations'] == 1
    assert score['execution_failures'] == 0


def test_execution_failures_also_reduce_stability():
    score = score_run(make_run(other_failures=2))

    assert score['execution_failures'] == 2
    assert abs(score['stability'] - 1.0 / 3.0) < 1e-9


def test_efficiency_penalizes_extra_actions():
    score = score_run(make_run(started=8, completed=8))

    assert score['efficiency'] == 0.5
    assert score['score_heuristic'] == 87.5


def test_unknown_template_skips_efficiency_and_renormalizes():
    score = score_run(make_run(command='走到苹果旁边看一下'))

    assert score['optimal_steps'] is None
    assert score['efficiency'] is None
    # 100 * (0.6 * 1 + 0.15 * 1) / (0.6 + 0.15)
    assert score['score_heuristic'] == 100.0


def test_failed_outcome_scores_zero_for_outcome():
    score = score_run(make_run(outcome='failed', started=2, completed=0))

    assert score['outcome_score'] == 0.0
    assert score['score_heuristic'] < 100.0


def test_plan_only_rejected_run_has_no_duration():
    score = score_run(make_run(outcome='rejected', started=0, completed=0))

    assert score['executed_actions'] == 0
    assert score['efficiency'] == 0.0
    assert score['duration_seconds'] is None
    assert score['stability'] == 1.0
    # 100 * (0.15 * 1) / 1.0
    assert score['score_heuristic'] == 15.0


def test_aggregate_summarizes_mixed_runs():
    scores = [
        score_run(make_run(run_id='run_a')),
        score_run(make_run(
            outcome='failed',
            started=2,
            completed=0,
            precondition_failures=1,
            run_id='run_b',
        )),
    ]

    summary = aggregate(scores)

    assert summary['n'] == 2
    assert summary['success_rate'] == 0.5
    assert summary['avg_executed_actions'] == 3.0
    assert summary['avg_score_heuristic'] is not None


def test_aggregate_empty_input_is_all_none():
    summary = aggregate([])

    assert summary['n'] == 0
    assert summary['success_rate'] is None


def test_score_directory_reads_run_records(tmp_path):
    (tmp_path / 'run_a.json').write_text(
        json.dumps(make_run(run_id='run_a')), encoding='utf-8'
    )
    (tmp_path / 'run_b.json').write_text(
        json.dumps(make_run(
            outcome='rejected', started=0, completed=0, run_id='run_b'
        )),
        encoding='utf-8'
    )

    scores, summary = score_directory(str(tmp_path))

    assert summary['n'] == 2
    assert [score['run_id'] for score in scores] == ['run_a', 'run_b']


def test_score_directory_skips_non_run_files(tmp_path):
    (tmp_path / 'summary.json').write_text(
        json.dumps({'n': 1}), encoding='utf-8'
    )
    (tmp_path / 'broken.json').write_text('{not json', encoding='utf-8')

    scores, summary = score_directory(str(tmp_path))

    assert scores == []
    assert summary['n'] == 0
