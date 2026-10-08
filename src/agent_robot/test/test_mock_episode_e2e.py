"""Deterministic mock episode acceptance tests for M5 Task 10."""

import os
import sys
import time
import json
from types import SimpleNamespace

sys.path.insert(
    0,
    os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
)

from agent_robot.scene_graph import (  # noqa: E402
    SceneGraph,
    validate_actions,
)
from agent_robot.scenarios.generator import generate_scenario  # noqa: E402
from agent_robot.scenarios.goals import evaluate_goal  # noqa: E402
from agent_robot.scenarios.spec import (  # noqa: E402
    CATEGORY_CLASSIFICATION,
    CATEGORY_RECOVERY,
    CATEGORY_SAFE_REJECTION,
    CATEGORY_SEQUENTIAL,
    CATEGORY_TARGET_PICK,
    REJECTION_SUBTYPES,
)
from agent_robot.scenarios.world_model import WorldModel  # noqa: E402
from agent_robot.skill_registry import SkillAction  # noqa: E402
from agent_robot.task_executor import (  # noqa: E402
    MockSkillBackend,
    TaskExecutor,
)
from agent_robot.trajectory import RunBuilder  # noqa: E402


class FakeLogger(object):
    def info(self, message):
        pass

    def error(self, message):
        pass


class FakeClock(object):
    def __init__(self):
        self.now = 1000.0

    def value(self):
        return self.now

    def advance(self, seconds):
        self.now += seconds


def make_action(skill, object_id, target_id=None):
    params = {'object': object_id}
    if target_id is not None:
        params['target'] = target_id
    return SkillAction(skill=skill, params=params)


def relocation_actions(object_id, container_id):
    return [
        make_action('move_to', object_id),
        make_action('pick', object_id),
        make_action('move_to', container_id),
        make_action('place', object_id, container_id),
    ]


def oracle_actions(spec):
    params = spec.goal.params
    if spec.category == CATEGORY_CLASSIFICATION:
        actions = []
        for pair in params['pairs']:
            blocks = sorted(
                obj.instance_id
                for obj in spec.instances_of('block', pair['color'])
            )
            for block_id in blocks:
                actions.extend(
                    relocation_actions(block_id, pair['bin_id'])
                )
        return actions

    if spec.category == CATEGORY_TARGET_PICK:
        return relocation_actions(
            params['target_id'], params['container_id']
        )

    if spec.category == CATEGORY_SEQUENTIAL:
        first = params['first']
        second = params['second']
        return (
            relocation_actions(
                first['object_id'], first['container_id']
            )
            + relocation_actions(
                second['object_id'], second['container_id']
            )
        )

    if spec.category == CATEGORY_RECOVERY:
        return relocation_actions(
            params['target_id'], params['container_id']
        )

    raise ValueError('No executable oracle plan for this category.')


def run_episode(spec, initial_actions=None, allow_replan=False):
    clock = FakeClock()
    builder = RunBuilder(clock=clock.value, monotonic=clock.value)
    world = WorldModel(spec, physical=False)
    graph = SceneGraph.build_from_environment(world.environment_payload())
    backend = MockSkillBackend(FakeLogger())
    command = spec.command
    actions = initial_actions or oracle_actions(spec)
    started_at = time.monotonic()

    builder.on_environment(world.environment_payload())
    builder.on_scene_graph(graph.to_dict())

    replan_count = 0
    while True:
        plan = {
            'command': command,
            'feasible': True,
            'actions': [action.to_dict() for action in actions],
            'planner': 'scripted_oracle',
            'replans': replan_count,
            'based_on_world_revision': graph.world_revision,
        }
        builder.on_plan(plan)

        report = validate_actions(actions, graph)
        assert report.ok, report.reason_detail

        failed = False
        for index, action in enumerate(actions, start=1):
            violation = graph.check_preconditions(action)
            if violation is not None:
                return None

            builder.on_status({
                'command': command,
                'status': 'action_started',
                'action': action.to_dict(),
                'step_index': index,
                'total_steps': len(actions),
                'world_revision': graph.world_revision,
            })
            assert world.on_action_started(
                action.skill, action.object, index
            ) is None

            backend.execute(action)
            graph.apply_action_effect(action)
            result = world.on_action_completed(
                action.skill,
                action.object,
                action.target,
                index,
            )
            assert result.error is None

            builder.on_status({
                'command': command,
                'status': 'action_completed',
                'action': action.to_dict(),
                'step_index': index,
                'total_steps': len(actions),
                'world_revision': graph.world_revision,
            })

            payload = world.environment_payload()
            if payload['world_revision'] > graph.world_revision:
                assert graph.update_observation(payload)
                builder.on_environment(payload)
                builder.on_scene_graph(graph.to_dict())

                if graph.world_revision > plan['based_on_world_revision']:
                    state = SimpleNamespace(
                        scene_graph=graph,
                        _active_actions=actions,
                        _active_index=index,
                        _active_command=command,
                        _active_plan_revision=(
                            plan['based_on_world_revision']
                        ),
                        logger=FakeLogger(),
                    )
                    state.get_logger = lambda: state.logger

                    def publish_status(
                        command,
                        status,
                        action=None,
                        index=None,
                        total=None,
                        reason='',
                    ):
                        builder.on_status({
                            'command': command,
                            'status': status,
                            'action': (
                                action.to_dict()
                                if action is not None else None
                            ),
                            'step_index': index,
                            'total_steps': total,
                            'reason': reason,
                            'world_revision': graph.world_revision,
                        })

                    state._publish_status = publish_status
                    state._publish_scene_graph = lambda: (
                        builder.on_scene_graph(graph.to_dict())
                    )
                    state._finish_revision_plan = lambda success: None
                    if not TaskExecutor._passes_revision_fence(state):
                        failed = True
                        break

        if failed:
            assert allow_replan
            replan_count += 1
            actions = oracle_actions(spec)
            continue

        builder.on_status({
            'command': command,
            'status': 'succeeded',
            'total_steps': len(actions),
            'reason': 'scripted mock episode completed',
            'world_revision': graph.world_revision,
        })
        break

    clock.advance(0.51)
    runs = builder.pop_ready_runs()
    assert len(runs) == 1
    run = runs[0]
    run['elapsed_test_seconds'] = time.monotonic() - started_at
    backend.shutdown()
    evaluation = evaluate_goal(spec.goal, graph, run)
    return run, graph, evaluation


def test_classification_mock_episode_reaches_goal():
    spec = generate_scenario(CATEGORY_CLASSIFICATION, seed=11)
    run, graph, evaluation = run_episode(spec)

    assert run['outcome'] == 'succeeded'
    assert evaluation['success'] is True
    assert run['scenario'] == {
        'scenario_id': spec.scenario_id,
        'category': CATEGORY_CLASSIFICATION,
        'seed': 11,
    }
    assert run['elapsed_test_seconds'] < 5.0


def test_target_pick_mock_episode_reaches_goal():
    spec = generate_scenario(CATEGORY_TARGET_PICK, seed=13)
    run, graph, evaluation = run_episode(spec)

    assert run['outcome'] == 'succeeded'
    assert evaluation['success'] is True
    assert run['elapsed_test_seconds'] < 5.0


def test_sequential_mock_episode_reaches_goal_in_order():
    spec = generate_scenario(CATEGORY_SEQUENTIAL, seed=17)
    run, graph, evaluation = run_episode(spec)

    assert run['outcome'] == 'succeeded'
    assert evaluation['success'] is True
    assert run['elapsed_test_seconds'] < 5.0


def test_recovery_mock_episode_rejects_stale_plan_then_replans():
    spec = generate_scenario(CATEGORY_RECOVERY, seed=7)
    first_plan = [
        make_action('move_to', spec.goal.params['target_id']),
        make_action('pick', spec.goal.params['target_id']),
        make_action('move_to', spec.goal.params['container_id']),
        make_action(
            'place',
            spec.goal.params['target_id'],
            spec.goal.params['container_id'],
        ),
    ]
    run, graph, evaluation = run_episode(
        spec,
        initial_actions=first_plan,
        allow_replan=True,
    )

    failures = [
        event for event in run['events']
        if event.get('status') == 'failed'
    ]
    assert len(failures) == 1
    assert failures[0]['reason'].startswith('stale_plan:')
    assert failures[0]['action']['skill'] == 'pick'
    assert run['outcome'] == 'succeeded'
    assert evaluation['success'] is True
    assert run['plans'][-1]['based_on_world_revision'] == 1
    assert run['world_revisions'] == [0, 1]
    assert run['failure_reasons'] == [failures[0]['reason']]
    assert json.loads(json.dumps(run))['scenario']['scenario_id'] \
        == spec.scenario_id
    assert any(
        item['status'] == 'action_started'
        and item['object'] == spec.goal.params['target_id']
        for item in run['action_revisions']
    )
    assert run['elapsed_test_seconds'] < 5.0


def run_rejection_episode(spec):
    clock = FakeClock()
    builder = RunBuilder(clock=clock.value, monotonic=clock.value)
    graph = SceneGraph.build_from_environment(
        WorldModel(spec).environment_payload()
    )
    builder.on_environment({
        'scenario_id': spec.scenario_id,
        'category': spec.category,
        'seed': spec.seed,
        'world_revision': 0,
    })
    builder.on_scene_graph(graph.to_dict())
    builder.on_plan({
        'command': spec.command,
        'feasible': False,
        'actions': [],
        'planner': 'scripted_oracle',
        'reason': 'safe_rejection: {}'.format(
            spec.goal.params['subtype']
        ),
        'based_on_world_revision': 0,
    })
    builder.on_status({
        'command': spec.command,
        'status': 'rejected',
        'reason': 'safe_rejection: {}'.format(
            spec.goal.params['subtype']
        ),
        'world_revision': 0,
    })
    clock.advance(0.51)
    run = builder.pop_ready_runs()[0]
    return run, evaluate_goal(spec.goal, graph, run)


def test_safe_rejection_mock_episode_rejects_all_subtypes_without_motion():
    for seed, subtype in enumerate(REJECTION_SUBTYPES, start=20):
        spec = generate_scenario(
            CATEGORY_SAFE_REJECTION,
            seed=seed,
            subtype=subtype,
        )
        run, evaluation = run_rejection_episode(spec)
        assert run['outcome'] == 'rejected'
        assert evaluation['success'] is True
        assert evaluation['details']['rejection_subtype'] == subtype
        assert evaluation['details']['action_started_count'] == 0
