import os
import re
import sys

# Allow running `python3 -m pytest test/test_trajectory.py` from the
# package root without a colcon/ROS install step.
sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.trajectory import (  # noqa: E402
    FINAL_GRAPH_WAIT,
    IDLE_FLUSH_TIMEOUT,
    RunBuilder,
    make_run_id,
)


class FakeClock(object):
    """Shared fake time for wall and monotonic clocks."""

    def __init__(self, start=1000.0):
        self.now = start

    def value(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds
        return self.now


def make_builder():
    clock = FakeClock()
    mono = FakeClock(start=5000.0)
    return RunBuilder(clock=clock.value, monotonic=mono.value), clock, mono


def plan(command='把红色苹果放进篮子里', feasible=True, replans=0):
    return {
        'command': command,
        'schema_version': '2.1',
        'feasible': feasible,
        'reason': '',
        'actions': [{'skill': 'move_to', 'object': 'apple'}],
        'planner': 'deepseek',
        'replans': replans,
    }


def status(name, command='把红色苹果放进篮子里', reason='', **fields):
    payload = {
        'command': command,
        'status': name,
        'reason': reason,
    }
    payload.update(fields)
    return payload


def test_rejected_run_flushes_only_after_final_graph_wait():
    builder, clock, mono = make_builder()

    builder.on_plan(plan(feasible=False))
    builder.on_status(status('rejected', reason='任务规划服务暂时不可用'))

    assert builder.pop_ready_runs() == []

    mono.advance(FINAL_GRAPH_WAIT - 0.01)
    assert builder.pop_ready_runs() == []

    mono.advance(0.02)
    runs = builder.pop_ready_runs()

    assert len(runs) == 1
    run = runs[0]
    assert run['outcome'] == 'rejected'
    assert run['outcome_reason'] == '任务规划服务暂时不可用'
    assert len(run['plans']) == 1
    assert run['plans'][0]['feasible'] is False


def test_full_success_run_records_events_and_graph_snapshots():
    builder, clock, mono = make_builder()

    graph_initial = {
        'schema_version': '1.0',
        'relations': [['on', 'apple', 'table']],
    }
    graph_final = {
        'schema_version': '1.0',
        'relations': [['in', 'apple', 'basket']],
    }

    builder.on_scene_graph(graph_initial)
    builder.on_plan(plan())

    for index in range(1, 5):
        builder.on_status(
            status('action_started', step_index=index, total_steps=4)
        )
        builder.on_status(
            status('action_completed', step_index=index, total_steps=4)
        )

    builder.on_status(status('succeeded', reason='All 4 action(s) completed.'))
    builder.on_scene_graph(graph_final)
    mono.advance(FINAL_GRAPH_WAIT + 0.01)

    runs = builder.pop_ready_runs()

    assert len(runs) == 1
    run = runs[0]
    assert run['outcome'] == 'succeeded'
    assert run['outcome_reason'] == 'All 4 action(s) completed.'
    assert len(run['events']) == 9
    assert run['scene_graph_initial'] == graph_initial
    assert run['scene_graph_final'] == graph_final
    assert run['finished_mono'] is not None


def test_graph_published_before_run_becomes_initial_snapshot():
    builder, clock, mono = make_builder()

    startup_graph = {'schema_version': '1.0', 'source': 'catalog_default'}
    builder.on_scene_graph(startup_graph)

    builder.on_plan(plan())
    assert builder.pop_ready_runs() == []

    mono.advance(FINAL_GRAPH_WAIT + 1.0)
    # A plan-only run never sees a terminal status; without a failure it
    # flushes as incomplete via the idle timeout.
    mono.advance(IDLE_FLUSH_TIMEOUT)
    runs = builder.pop_ready_runs()

    assert len(runs) == 1
    assert runs[0]['scene_graph_initial'] == startup_graph


def test_failed_then_replan_merges_into_single_run():
    builder, clock, mono = make_builder()

    builder.on_plan(plan())
    builder.on_status(status('action_started', step_index=1, total_steps=4))
    builder.on_status(
        status(
            'failed',
            reason='precondition_violation: object_not_graspable: x',
            step_index=2,
            total_steps=4,
        )
    )
    builder.on_plan(plan(replans=1))
    builder.on_status(status('action_started', step_index=1, total_steps=4))
    builder.on_status(status('succeeded', reason='All 4 action(s) completed.'))

    mono.advance(FINAL_GRAPH_WAIT + 0.01)
    runs = builder.pop_ready_runs()

    assert len(runs) == 1
    run = runs[0]
    assert run['outcome'] == 'succeeded'
    assert len(run['plans']) == 2
    assert run['plans'][1]['replans'] == 1
    assert len(run['events']) == 4


def test_grasp_failure_replan_records_recovery_fields():
    builder, clock, mono = make_builder()
    builder.on_plan(plan())
    builder.on_status(status(
        'failed',
        reason='grasp failed',
        failure_code='grasp_failed',
        action={'skill': 'pick', 'params': {'object': 'apple'}},
        step_index=2,
    ))
    builder.on_plan(dict(
        plan(replans=1),
        status='replanned',
    ))
    builder.on_status(status('succeeded'))

    mono.advance(FINAL_GRAPH_WAIT + 0.01)
    run = builder.pop_ready_runs()[0]

    assert run['failures'][0]['failure_code'] == 'grasp_failed'
    assert run['replans'] == 1
    assert run['recovered'] is True


def test_run_records_scenario_revision_and_world_event_context():
    builder, clock, mono = make_builder()
    builder.on_environment({
        'scenario_id': 'recovery_seed7',
        'category': 'recovery',
        'seed': 7,
        'world_revision': 0,
    })
    builder.on_plan(plan())
    action = {
        'skill': 'move_to',
        'params': {'object': 'apple'},
    }
    builder.on_status(status(
        'action_started',
        action=action,
        step_index=1,
        world_revision=0,
    ))
    builder.on_environment({
        'scenario_id': 'recovery_seed7',
        'category': 'recovery',
        'seed': 7,
        'world_revision': 1,
    })
    builder.on_environment({
        'scenario_id': 'recovery_seed7',
        'category': 'recovery',
        'seed': 7,
        'world_revision': 1,
    })
    builder.on_world_event({
        'revision': 1,
        'event': 'object_slide',
        'target': 'apple',
    })
    builder.on_status(status(
        'action_completed',
        action=action,
        step_index=1,
        world_revision=1,
    ))
    builder.on_status(
        status('failed', reason='stale_plan: object_not_in_scene')
    )
    builder.on_plan(plan(replans=1))
    builder.on_status(status('succeeded'))

    mono.advance(FINAL_GRAPH_WAIT + 0.01)
    run = builder.pop_ready_runs()[0]

    assert run['scenario'] == {
        'scenario_id': 'recovery_seed7',
        'category': 'recovery',
        'seed': 7,
    }
    assert run['world_revisions'] == [0, 1]
    assert run['action_revisions'] == [
        {
            'status': 'action_started',
            'step_index': 1,
            'skill': 'move_to',
            'object': 'apple',
            'target': None,
            'world_revision': 0,
        },
        {
            'status': 'action_completed',
            'step_index': 1,
            'skill': 'move_to',
            'object': 'apple',
            'target': None,
            'world_revision': 1,
        },
    ]
    assert run['world_events'] == [
        {
            'kind': 'revision_changed',
            'from_revision': 0,
            'revision': 1,
        },
        {
            'revision': 1,
            'event': 'object_slide',
            'target': 'apple',
        },
    ]
    assert run['failure_reasons'] == ['stale_plan: object_not_in_scene']
    assert run['plans'][0]['based_on_world_revision'] is None

    builder.on_plan(plan(command='next task'))
    builder.on_status(status('succeeded', command='next task'))
    mono.advance(FINAL_GRAPH_WAIT + 0.01)
    next_run = builder.pop_ready_runs()[0]
    assert next_run['world_revisions'] == [1]


def test_new_scenario_resets_episode_revision_tracking():
    builder, clock, mono = make_builder()
    builder.on_environment({
        'scenario_id': 'episode_a',
        'category': 'target_pick',
        'seed': 1,
        'world_revision': 3,
    })
    builder.on_plan(plan(command='episode A'))
    builder.on_status(status('succeeded', command='episode A'))
    mono.advance(FINAL_GRAPH_WAIT + 0.01)
    first = builder.pop_ready_runs()[0]
    assert first['world_revisions'] == [3]

    builder.on_environment({
        'scenario_id': 'episode_b',
        'category': 'recovery',
        'seed': 2,
        'world_revision': 0,
    })
    builder.on_plan(plan(command='episode B'))
    builder.on_status(status('succeeded', command='episode B'))
    mono.advance(FINAL_GRAPH_WAIT + 0.01)
    second = builder.pop_ready_runs()[0]

    assert second['scenario']['scenario_id'] == 'episode_b'
    assert second['world_revisions'] == [0]


def test_failed_without_replan_flushes_as_failed_on_idle_timeout():
    builder, clock, mono = make_builder()

    builder.on_plan(plan())
    builder.on_status(status('action_started'))
    builder.on_status(status('failed', reason='precondition_violation: x'))

    mono.advance(IDLE_FLUSH_TIMEOUT - 1.0)
    assert builder.pop_ready_runs() == []

    mono.advance(2.0)
    runs = builder.pop_ready_runs()

    assert len(runs) == 1
    run = runs[0]
    assert run['outcome'] == 'failed'
    assert 'no replan' in run['outcome_reason']


def test_plan_only_run_flushes_as_incomplete_on_idle_timeout():
    builder, clock, mono = make_builder()

    builder.on_plan(plan())

    mono.advance(IDLE_FLUSH_TIMEOUT + 1.0)
    runs = builder.pop_ready_runs()

    assert len(runs) == 1
    assert runs[0]['outcome'] == 'incomplete'
    assert runs[0]['events'] == []


def test_idle_timeout_is_measured_from_the_last_event():
    builder, clock, mono = make_builder()

    builder.on_plan(plan())
    builder.on_status(status('action_started'))

    mono.advance(IDLE_FLUSH_TIMEOUT / 2.0)
    builder.on_status(status('action_completed'))
    mono.advance(IDLE_FLUSH_TIMEOUT / 2.0 + 1.0)

    # Only IDLE/2 + 1 seconds elapsed since the last event.
    assert builder.pop_ready_runs() == []


def test_new_command_supersedes_the_active_run():
    builder, clock, mono = make_builder()

    builder.on_plan(plan(command='任务A'))
    builder.on_status(status('failed', command='任务A'))
    builder.on_plan(plan(command='任务B'))
    builder.on_status(status('succeeded', command='任务B'))

    runs = builder.pop_ready_runs()

    assert len(runs) == 1
    assert runs[0]['command'] == '任务A'
    assert runs[0]['outcome'] == 'failed'
    assert runs[0]['outcome_reason'] == 'superseded_by_new_command'

    mono.advance(FINAL_GRAPH_WAIT + 0.01)
    runs = builder.pop_ready_runs()

    assert len(runs) == 1
    assert runs[0]['command'] == '任务B'
    assert runs[0]['outcome'] == 'succeeded'


def test_run_ids_are_unique_and_uuid_suffixed():
    builder, clock, mono = make_builder()

    builder.on_plan(plan(command='任务A'))
    builder.on_status(status('succeeded', command='任务A'))
    mono.advance(FINAL_GRAPH_WAIT + 0.01)
    first = builder.pop_ready_runs()[0]

    clock.advance(2.0)
    builder.on_plan(plan(command='任务B'))
    builder.on_status(status('succeeded', command='任务B'))
    mono.advance(FINAL_GRAPH_WAIT + 0.01)
    second = builder.pop_ready_runs()[0]

    pattern = re.compile(r'^run_\d{8}T\d{6}_[0-9a-f]{8}$')

    assert pattern.match(first['run_id'])
    assert pattern.match(second['run_id'])
    assert first['run_id'] != second['run_id']


def test_make_run_id_alone_is_unique():
    ids = {make_run_id(FakeClock().value) for _ in range(20)}

    assert len(ids) == 20


def test_events_carry_wall_and_monotonic_timestamps():
    builder, clock, mono = make_builder()

    builder.on_plan(plan())
    builder.on_status(status('action_started'))

    run = builder.close_active('recorder_shutdown')

    event = run['events'][0]
    assert event['t'] == 1000.0
    assert event['mono'] == 5000.0


def test_close_active_flushes_the_open_run_immediately():
    builder, clock, mono = make_builder()

    builder.on_plan(plan())
    builder.on_status(status('action_started'))

    run = builder.close_active('recorder_shutdown')

    assert run['outcome'] == 'incomplete'
    assert run['outcome_reason'] == 'recorder_shutdown'
    assert builder.pop_ready_runs() == []


def test_scene_graph_updates_refresh_active_run_final_snapshot():
    builder, clock, mono = make_builder()

    builder.on_plan(plan())
    builder.on_status(status('action_started'))

    graph_after_action = {'schema_version': '1.0', 'holding': 'apple'}
    builder.on_scene_graph(graph_after_action)

    run = builder.close_active('recorder_shutdown')

    assert run['scene_graph_final'] == graph_after_action


def test_malformed_message_is_ignored_defensively():
    builder, clock, mono = make_builder()

    builder.on_plan('not-a-dict')
    builder.on_status(None)
    builder.on_scene_graph(None)

    assert builder.pop_ready_runs() == []

    # The builder stays functional for well-formed messages.
    builder.on_plan(plan())
    builder.on_status(status('succeeded'))
    mono.advance(FINAL_GRAPH_WAIT + 0.01)

    runs = builder.pop_ready_runs()

    assert len(runs) == 1
    assert runs[0]['outcome'] == 'succeeded'


def test_episode_contains_image_instruction_plan_trajectory_and_success():
    builder, clock, mono = make_builder()
    builder.on_environment({
        'scenario_id': 'image_scene_2',
        'image_path': '/tmp/images/2.png',
        'source': 'perception',
        'world_revision': 0,
    })
    builder.on_plan(dict(
        plan(command='put apple into basket'),
        image_path='/tmp/images/2.png',
    ))
    builder.on_trajectory_point({
        'position': [0.4, 0.1, 0.2],
        'joint_positions': [0.0] * 7,
    })
    builder.on_status(status('succeeded', command='put apple into basket'))
    mono.advance(FINAL_GRAPH_WAIT + 0.01)

    run = builder.pop_ready_runs()[0]

    assert run['image'] == '/tmp/images/2.png'
    assert run['instruction'] == 'put apple into basket'
    assert len(run['plan']) == 1
    assert len(run['trajectory']) == 1
    assert run['success'] is True
