"""
Deterministic world truth and boundary event state machine (M5.3).

The WorldModel is the *only* logical ground truth for mock and PyBullet
backends. It advances on executor ``/task_status`` boundaries and publishes
observable environment payloads for the executor belief layer.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agent_robot.scene_graph import OBJECT_TYPES
from agent_robot.scenarios.spec import (
    EVENT_GRASP_FAILURE,
    EVENT_OBJECT_ROLL,
    EVENT_OBJECT_SLIDE,
    EVENT_OCCLUDE,
    EVENT_REVEAL,
    EventScript,
    SceneSpec,
)


class EventPreconditionError(RuntimeError):
    """A world event cannot fire because preconditions are not met."""


@dataclass(frozen=True)
class FailureInject:
    """One-shot simulated grasp failure for ``/simulate_failure``."""

    skill: str
    object: Optional[str] = None


@dataclass(frozen=True)
class WorldEventCommand:
    """Physics-backend relocation command (PyBullet ``/world_event``)."""

    revision: int
    event: str
    target: str
    new_position: Tuple[float, float, float]
    new_support: Optional[str]
    new_visibility: Optional[bool]


@dataclass
class InstanceTruth:
    object_type: str
    color: Optional[str]
    position: Tuple[float, float, float]
    support: str
    visible: bool = True


@dataclass
class _PendingPhysicalEvent:
    script: EventScript
    revision_after: int


@dataclass
class ActionCompletedResult:
    failure_injects: List[FailureInject] = field(default_factory=list)
    world_events: List[WorldEventCommand] = field(default_factory=list)
    environment_changed: bool = False
    error: Optional[str] = None


class WorldModel(object):
    """Episode ground truth driven by SceneSpec and action boundaries."""

    def __init__(self, spec, physical=False):
        # type: (SceneSpec, bool) -> None
        self.spec = spec
        self.physical = bool(physical)
        self.instances = {}  # type: Dict[str, InstanceTruth]
        self.holding = None  # type: Optional[str]
        self.robot_at = None  # type: Optional[str]
        self.revision = 0
        self.timeline = []  # type: List[dict]
        self._events = list(spec.events)
        self._consumed_event_keys = set()  # type: set
        self._pending = None  # type: Optional[_PendingPhysicalEvent]
        self._pending_commands = []  # type: List[WorldEventCommand]
        self._failed_event_keys = set()  # type: set

        for obj in spec.objects:
            self.instances[obj.instance_id] = InstanceTruth(
                object_type=obj.object_type,
                color=obj.color,
                position=tuple(obj.position),
                support=obj.support,
                visible=obj.visible,
            )

    # ------------------------------------------------------------------
    # Observable environment
    # ------------------------------------------------------------------

    def environment_payload(self):
        # type: () -> dict
        visible = []
        occluded = []

        for instance_id, truth in sorted(self.instances.items()):
            if truth.visible:
                visible.append(self._object_record(instance_id, truth))
            else:
                occluded.append(instance_id)

        return {
            'scenario_id': self.spec.scenario_id,
            'category': self.spec.category,
            'seed': self.spec.seed,
            'world_revision': self.revision,
            'occluded': occluded,
            'objects': visible,
        }

    def _object_record(self, instance_id, truth):
        # type: (str, InstanceTruth) -> dict
        record = {
            'id': instance_id,
            'name': instance_id,
            'type': truth.object_type,
            'position': list(truth.position),
            'location': truth.support,
            'visible': truth.visible,
        }
        if truth.color is not None:
            record['color'] = truth.color

        support_obj = self.instances.get(truth.support)
        if support_obj is not None:
            support_info = OBJECT_TYPES[support_obj.object_type]
            if support_info.receptacle == 'container':
                record['relation'] = 'in'
            else:
                record['relation'] = 'on'
        else:
            record['relation'] = 'on'

        return record

    # ------------------------------------------------------------------
    # Action boundaries
    # ------------------------------------------------------------------

    def on_action_started(self, skill, object_name, index):
        # type: (str, Optional[str], int) -> Optional[FailureInject]
        """Deliver grasp_failure injections at action start."""
        for key, script in self._iter_events(index):
            if script.event != EVENT_GRASP_FAILURE:
                continue
            self._consumed_event_keys.add(key)
            self.timeline.append(
                {
                    'kind': 'grasp_failure_inject',
                    'after_action_index': index,
                    'target': script.target,
                    'revision': self.revision,
                }
            )
            return FailureInject(skill='pick', object=script.target)
        return None

    def on_action_completed(
        self,
        skill: str,
        object_name: Optional[str],
        target_name: Optional[str],
        index: int,
    ) -> ActionCompletedResult:
        result = ActionCompletedResult()

        if self._pending is not None:
            result.error = 'world_event_pending'
            return result

        boundary_events = [
            script for _, script in self._iter_events(index)
            if script.event != EVENT_GRASP_FAILURE
        ]
        holding_after_action = self.holding
        if skill == 'pick' and object_name:
            holding_after_action = object_name
        elif skill == 'place':
            holding_after_action = None
        if (
            boundary_events
            and holding_after_action is not None
        ):
            raise EventPreconditionError(
                'world_event_requires_empty_hand:{}'.format(
                    boundary_events[0].event
                )
            )

        self._apply_skill_effect(skill, object_name, target_name)
        result.environment_changed = True

        for key, script in self._iter_events(index):
            if script.event == EVENT_GRASP_FAILURE:
                continue
            self._fire_boundary_event(key, script, result)

        return result

    def _apply_skill_effect(self, skill, object_name, target_name):
        # type: (str, Optional[str], Optional[str]) -> None
        if skill == 'move_to':
            if object_name:
                self.robot_at = object_name
            return

        if skill == 'pick':
            if not object_name:
                return
            self.holding = object_name
            truth = self.instances.get(object_name)
            if truth is not None:
                truth.support = 'gripper'
            return

        if skill == 'place':
            if not object_name or not target_name:
                return
            truth = self.instances.get(object_name)
            if truth is not None:
                truth.support = target_name
            self.holding = None
            return

    def _iter_events(self, index):
        # type: (int) -> List[Tuple[str, EventScript]]
        matched = []
        for script in self._events:
            if script.after_action_index != index:
                continue
            key = self._event_key(script)
            if key in self._consumed_event_keys:
                continue
            if key in self._failed_event_keys:
                continue
            matched.append((key, script))
        return matched

    @staticmethod
    def _event_key(script):
        # type: (EventScript) -> str
        return '{}:{}:{}'.format(
            script.event,
            script.target,
            script.after_action_index,
        )

    def _fire_boundary_event(self, key, script, result):
        # type: (str, EventScript, ActionCompletedResult) -> None
        if self.holding is not None:
            raise EventPreconditionError(
                'world_event_requires_empty_hand:{}'.format(script.event)
            )

        if script.event in (EVENT_OBJECT_SLIDE, EVENT_OBJECT_ROLL):
            self._apply_reposition(key, script, result)
        elif script.event == EVENT_OCCLUDE:
            self._apply_visibility(key, script, False, result)
        elif script.event == EVENT_REVEAL:
            self._apply_visibility(key, script, True, result)
        else:
            raise EventPreconditionError(
                'unsupported_event:{}'.format(script.event)
            )

    def _apply_reposition(self, key, script, result):
        # type: (str, EventScript, ActionCompletedResult) -> None
        truth = self.instances.get(script.target)
        if truth is None:
            raise EventPreconditionError(
                'event_target_missing:{}'.format(script.target)
            )
        if script.new_position is None:
            raise EventPreconditionError(
                'event_missing_position:{}'.format(script.event)
            )

        new_support = script.new_support or truth.support
        new_position = tuple(script.new_position)
        next_revision = self.revision + 1

        if self.physical:
            command = WorldEventCommand(
                revision=next_revision,
                event=script.event,
                target=script.target,
                new_position=new_position,
                new_support=new_support,
                new_visibility=None,
            )
            self._pending = _PendingPhysicalEvent(
                script=script,
                revision_after=next_revision,
            )
            self._pending_commands.append(command)
            result.world_events.append(command)
            return

        truth.position = new_position
        truth.support = new_support
        self.revision = next_revision
        self._consumed_event_keys.add(key)
        self.timeline.append(
            {
                'kind': script.event,
                'target': script.target,
                'revision': self.revision,
                'position': list(new_position),
                'support': new_support,
            }
        )
        result.environment_changed = True

    def _apply_visibility(self, key, script, visible, result):
        # type: (str, EventScript, bool, ActionCompletedResult) -> None
        truth = self.instances.get(script.target)
        if truth is None:
            raise EventPreconditionError(
                'event_target_missing:{}'.format(script.target)
            )

        next_revision = self.revision + 1

        if self.physical:
            command = WorldEventCommand(
                revision=next_revision,
                event=script.event,
                target=script.target,
                new_position=truth.position,
                new_support=truth.support,
                new_visibility=visible,
            )
            self._pending = _PendingPhysicalEvent(
                script=script,
                revision_after=next_revision,
            )
            self._pending_commands.append(command)
            result.world_events.append(command)
            return

        truth.visible = visible
        self.revision = next_revision
        self._consumed_event_keys.add(key)
        self.timeline.append(
            {
                'kind': script.event,
                'target': script.target,
                'revision': self.revision,
                'visible': visible,
            }
        )
        result.environment_changed = True

    # ------------------------------------------------------------------
    # Physical backend ack
    # ------------------------------------------------------------------

    def on_event_ack(self, revision, ok, actual_position=None):
        # type: (int, bool, Optional[Tuple[float, float, float]]) -> bool
        """Commit or reject a pending physical event."""
        pending = self._pending
        if pending is None:
            return False
        if pending.revision_after != revision:
            return False

        key = self._event_key(pending.script)
        script = pending.script

        if not ok:
            self._pending = None
            self._failed_event_keys.add(key)
            self.timeline.append(
                {
                    'kind': 'event_ack_failed',
                    'event': script.event,
                    'target': script.target,
                    'revision': revision,
                }
            )
            return False

        truth = self.instances.get(script.target)
        if truth is None:
            self._pending = None
            return False

        if script.event in (EVENT_OBJECT_SLIDE, EVENT_OBJECT_ROLL):
            if actual_position is not None:
                truth.position = tuple(actual_position)
            elif script.new_position is not None:
                truth.position = tuple(script.new_position)
            if script.new_support is not None:
                truth.support = script.new_support
        elif script.event in (EVENT_OCCLUDE, EVENT_REVEAL):
            if script.new_visibility is not None:
                truth.visible = bool(script.new_visibility)
            elif script.event == EVENT_OCCLUDE:
                truth.visible = False
            else:
                truth.visible = True

        self.revision = pending.revision_after
        self._consumed_event_keys.add(key)
        self._pending = None
        self.timeline.append(
            {
                'kind': script.event,
                'target': script.target,
                'revision': self.revision,
                'ack': True,
            }
        )
        return True

    def pending_environment(self):
        # type: () -> bool
        """Whether observers should republish after the last transition."""
        return self._pending is None

    def drain_commands(self):
        # type: () -> List[WorldEventCommand]
        commands = list(self._pending_commands)
        self._pending_commands.clear()
        return commands

    def event_already_consumed(self, script):
        # type: (EventScript) -> bool
        return self._event_key(script) in self._consumed_event_keys
