"""Tests for revision-aware executor fences and observation caching."""

import json
import os
import sys
from types import SimpleNamespace

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.scene_graph import SceneGraph  # noqa: E402
from agent_robot.skill_registry import SkillAction  # noqa: E402
from agent_robot.task_executor import TaskExecutor  # noqa: E402


class FakeLogger(object):
    def warning(self, message):
        pass

    def error(self, message):
        pass

    def info(self, message):
        pass


class FakeBackend(object):
    def __init__(self):
        self.actions = []

    def execute(self, action):
        self.actions.append(action)


def environment(revision, include_apple=True, block_x=0.4):
    objects = [
        {
            'id': 'basket',
            'type': 'basket',
            'position': [0.7, 0.2, 0.05],
            'location': 'table',
        },
        {
            'id': 'red_block_1',
            'type': 'block',
            'color': 'red',
            'position': [block_x, -0.2, 0.025],
            'location': 'table',
        },
    ]
    if include_apple:
        objects.append({
            'id': 'apple',
            'type': 'apple',
            'position': [0.5, 0.0, 0.06],
            'location': 'table',
        })
    return {
        'scenario_id': 'revision_test',
        'world_revision': revision,
        'objects': objects,
    }


def make_executor_state(graph, actions, active_index, plan_revision):
    state = SimpleNamespace(
        scene_graph=graph,
        _active_actions=actions,
        _active_index=active_index,
        _active_command='test command',
        _active_plan_revision=plan_revision,
        executing=True,
        _motion_busy=False,
        statuses=[],
        graph_publications=0,
        logger=FakeLogger(),
    )
    state.get_logger = lambda: state.logger
    state._publish_status = lambda **status: state.statuses.append(status)
    state._publish_scene_graph = lambda: setattr(
        state, 'graph_publications', state.graph_publications + 1
    )
    state._finish_revision_plan = lambda success: finish_plan(
        state, success
    )
    state._publish_backend_trajectory = (
        TaskExecutor._publish_backend_trajectory.__get__(
            state, TaskExecutor
        )
    )
    return state


def finish_plan(state, success):
    if success:
        state.statuses.append({'status': 'succeeded'})
    state._active_actions = []
    state._active_command = ''
    state._active_index = 0
    state._active_plan_revision = None
    state.executing = False


def test_revision_fence_rejects_stale_remaining_pick_before_start():
    graph = SceneGraph.build_from_environment(
        environment(1, include_apple=False)
    )
    graph.robot_at = 'apple'
    actions = [
        SkillAction(skill='move_to', params={'object': 'apple'}),
        SkillAction(skill='pick', params={'object': 'apple'}),
    ]
    state = make_executor_state(graph, actions, 1, 0)

    accepted = TaskExecutor._passes_revision_fence(state)

    assert accepted is False
    assert state._active_actions == []
    failed = [s for s in state.statuses if s.get('status') == 'failed']
    assert len(failed) == 1
    assert failed[0]['reason'].startswith('stale_plan: object_not_in_scene')
    assert not any(
        status.get('status') == 'action_started'
        for status in state.statuses
    )


def test_unrelated_revision_change_allows_remaining_plan_to_succeed():
    graph = SceneGraph.build_from_environment(environment(0))
    graph.apply_action_effect(
        SkillAction(skill='move_to', params={'object': 'apple'})
    )
    assert graph.update_observation(environment(1, block_x=0.42))

    actions = [
        SkillAction(skill='move_to', params={'object': 'apple'}),
        SkillAction(skill='pick', params={'object': 'apple'}),
        SkillAction(skill='move_to', params={'object': 'basket'}),
        SkillAction(
            skill='place',
            params={'object': 'apple', 'target': 'basket'},
        ),
    ]
    state = make_executor_state(graph, actions, 1, 0)
    backend = FakeBackend()
    state._drain_environment = lambda: None
    state._failure_matches = lambda action: False
    state._get_backend = lambda: backend

    for name in (
        '_execution_step',
        '_passes_revision_fence',
    ):
        setattr(
            state,
            name,
            getattr(TaskExecutor, name).__get__(state, TaskExecutor),
        )

    for _ in range(3):
        state._execution_step()

    assert [action.skill for action in backend.actions] == [
        'pick', 'move_to', 'place'
    ]
    assert [s['status'] for s in state.statuses].count('failed') == 0
    assert state.statuses[-1]['status'] == 'succeeded'
    assert graph.world_revision == 1


def test_plan_hash_includes_world_revision():
    actions = [{'skill': 'move_to', 'object': 'apple'}]
    first = TaskExecutor._hash_plan('task', True, actions, 1)
    newer = TaskExecutor._hash_plan('task', True, actions, 2)
    assert first != newer


def test_plan_hash_allows_revised_plan_with_same_actions():
    actions = [{'skill': 'move_to', 'object': 'apple'}]
    initial = TaskExecutor._hash_plan('task', True, actions, 1, 0)
    revised = TaskExecutor._hash_plan('task', True, actions, 1, 1)
    assert initial != revised


def test_same_plan_at_new_revision_is_accepted_again():
    graph = SceneGraph.build_from_environment(environment(0))
    state = SimpleNamespace(
        executing=False,
        last_plan_hash=None,
        _revision_protocol=True,
        _belief_initialized=True,
        scene_graph=graph,
        current_environment=environment(0),
        logger=FakeLogger(),
        accepted_plans=[],
    )
    state.get_logger = lambda: state.logger
    state._hash_plan = TaskExecutor._hash_plan
    state._drain_environment = lambda force_initialize=False: None
    state._start_revision_plan = lambda command, actions, revision: (
        state.accepted_plans.append(revision)
    )
    plan = {
        'command': 'put apple in basket',
        'feasible': True,
        'actions': [
            {'skill': 'move_to', 'object': 'apple'},
            {'skill': 'pick', 'object': 'apple'},
            {'skill': 'move_to', 'object': 'basket'},
            {
                'skill': 'place',
                'object': 'apple',
                'target': 'basket',
            },
        ],
    }

    for revision in (1, 2):
        current_plan = dict(plan)
        current_plan['based_on_world_revision'] = revision
        message = SimpleNamespace(
            data=json.dumps(current_plan, ensure_ascii=False)
        )
        TaskExecutor.plan_callback(state, message)

    assert state.accepted_plans == [1, 2]


def test_old_and_duplicate_environment_frames_do_not_replace_cache():
    latest = environment(2)
    state = SimpleNamespace(
        current_environment=latest,
        _revision_protocol=True,
        _cached_environment=latest,
        _cached_revision=2,
    )
    old_message = SimpleNamespace(data=json.dumps(environment(1)))
    duplicate_message = SimpleNamespace(
        data=json.dumps(environment(2, block_x=0.55))
    )

    TaskExecutor.environment_callback(state, old_message)
    TaskExecutor.environment_callback(state, duplicate_message)

    assert state._cached_environment is latest
    assert state.current_environment is latest
    assert state._cached_revision == 2
