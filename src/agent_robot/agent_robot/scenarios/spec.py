"""
Episode specification data model for M5 multi-scene simulations.

A :class:`SceneSpec` is the *read-only initial configuration* of one
episode: object instances, their layout, the deterministic event script,
the rendered command and the goal parameters. Runtime truth lives in the
WorldModel (world_node); the physics backend only executes; SceneGraph is
the executor's belief. None of those ownership rules live here -- this
module only defines structures, JSON conversion and validation.
"""

import math
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from agent_robot.scene_graph import (
    CANONICAL_OBJECTS,
    MULTI_INSTANCE_TYPES,
    OBJECT_TYPES,
    SCENE_COLORS,
    parse_instance_id,
)

SCENARIO_SCHEMA_VERSION = '1.0'

# Semantic category identifiers (also used as suite id prefixes).
CATEGORY_CLASSIFICATION = 'classification'
CATEGORY_TARGET_PICK = 'target_pick'
CATEGORY_SEQUENTIAL = 'sequential'
CATEGORY_RECOVERY = 'recovery'
CATEGORY_SAFE_REJECTION = 'safe_rejection'

SCENARIO_CATEGORIES = (
    CATEGORY_CLASSIFICATION,
    CATEGORY_TARGET_PICK,
    CATEGORY_SEQUENTIAL,
    CATEGORY_RECOVERY,
    CATEGORY_SAFE_REJECTION,
)

# Safe-rejection goal subtypes.
REJECT_MISSING_TARGET = 'missing_target'
REJECT_WRONG_CONTAINER = 'wrong_container'
REJECT_NOT_GRASPABLE = 'not_graspable'

REJECTION_SUBTYPES = (
    REJECT_MISSING_TARGET,
    REJECT_WRONG_CONTAINER,
    REJECT_NOT_GRASPABLE,
)

# Deterministic world event types.
EVENT_OBJECT_SLIDE = 'object_slide'
EVENT_OBJECT_ROLL = 'object_roll'
EVENT_OCCLUDE = 'occlude'
EVENT_REVEAL = 'reveal'
EVENT_GRASP_FAILURE = 'grasp_failure'

EVENT_TYPES = (
    EVENT_OBJECT_SLIDE,
    EVENT_OBJECT_ROLL,
    EVENT_OCCLUDE,
    EVENT_REVEAL,
    EVENT_GRASP_FAILURE,
)

EVENTS_REPOSITION = (EVENT_OBJECT_SLIDE, EVENT_OBJECT_ROLL)

# Franka tabletop reachable envelope (table loaded at [0.5, 0, -0.65]).
TABLE_SURFACE_Z = 0.0
TABLE_X_RANGE = (0.35, 0.80)
TABLE_Y_RANGE = (-0.35, 0.35)
TABLE_Z_RANGE = (0.0, 0.30)

# Wider envelope used for post-event floor/relocated positions.
WORLD_X_RANGE = (0.25, 0.85)
WORLD_Y_RANGE = (-0.45, 0.45)
WORLD_Z_RANGE = (0.0, 0.30)

# Horizontal half extents of each type's collision footprint.
FOOTPRINT_HALF_EXTENTS = {
    'apple': (0.045, 0.045),
    'cup': (0.045, 0.045),
    'basket': (0.13, 0.10),
    'block': (0.025, 0.025),
    'bin': (0.12, 0.09),
}

MIN_OBJECT_GAP = 0.03


@dataclass(frozen=True)
class ObjectInstance:
    """One physical object instance in the episode."""

    instance_id: str
    object_type: str
    color: Optional[str]
    position: Tuple[float, float, float]
    support: str = 'table'
    visible: bool = True

    @property
    def graspable(self):
        # type: () -> bool
        return OBJECT_TYPES[self.object_type].graspable

    @property
    def receptacle(self):
        # type: () -> Optional[str]
        return OBJECT_TYPES[self.object_type].receptacle

    @property
    def footprint(self):
        # type: () -> Tuple[float, float]
        return FOOTPRINT_HALF_EXTENTS[self.object_type]


@dataclass(frozen=True)
class EventScript:
    """
    One deterministic boundary event.

    ``after_action_index`` is 1-based: the event fires once, immediately
    after that action completed and before the next action starts.
    """

    event: str
    target: str
    after_action_index: int = 1
    new_position: Optional[Tuple[float, float, float]] = None
    new_support: Optional[str] = None
    new_visibility: Optional[bool] = None


@dataclass(frozen=True)
class ScenarioGoal:
    """Category plus JSON-serializable goal parameters."""

    category: str
    params: Dict[str, object] = field(default_factory=dict)


@dataclass(frozen=True)
class SceneSpec:
    """Read-only initial configuration of one episode."""

    scenario_id: str
    category: str
    seed: int
    command: str
    objects: Tuple[ObjectInstance, ...]
    goal: ScenarioGoal
    events: Tuple[EventScript, ...] = ()
    schema_version: str = SCENARIO_SCHEMA_VERSION

    def object_by_id(self, instance_id):
        # type: (str) -> Optional[ObjectInstance]
        for obj in self.objects:
            if obj.instance_id == instance_id:
                return obj
        return None

    def instances_of(self, object_type=None, color=None):
        # type: (Optional[str], Optional[str]) -> List[ObjectInstance]
        result = []
        for obj in self.objects:
            if object_type is not None and obj.object_type != object_type:
                continue
            if color is not None and obj.color != color:
                continue
            result.append(obj)
        return result

    def to_dict(self):
        # type: () -> dict
        return {
            'scenario_id': self.scenario_id,
            'category': self.category,
            'seed': self.seed,
            'command': self.command,
            'schema_version': self.schema_version,
            'objects': [
                {
                    'id': obj.instance_id,
                    'type': obj.object_type,
                    'color': obj.color,
                    'position': list(obj.position),
                    'support': obj.support,
                    'visible': obj.visible,
                }
                for obj in self.objects
            ],
            'events': [
                {
                    'event': evt.event,
                    'target': evt.target,
                    'after_action_index': evt.after_action_index,
                    'new_position': (
                        list(evt.new_position)
                        if evt.new_position is not None else None
                    ),
                    'new_support': evt.new_support,
                    'new_visibility': evt.new_visibility,
                }
                for evt in self.events
            ],
            'goal': {
                'category': self.goal.category,
                'params': self.goal.params,
            },
        }


def _as_tuple3(value, label):
    # type: (object, str) -> Tuple[float, float, float]
    if not isinstance(value, (list, tuple)) or len(value) != 3:
        raise ValueError(
            '{} must be a list/tuple of three numbers.'.format(label)
        )
    try:
        return (float(value[0]), float(value[1]), float(value[2]))
    except (TypeError, ValueError):
        raise ValueError(
            '{} contains a non-numeric coordinate.'.format(label)
        )


def object_instance_from_dict(raw):
    # type: (dict) -> ObjectInstance
    instance_id = raw.get('id')
    object_type = raw.get('type')

    if not isinstance(instance_id, str) or not instance_id:
        raise ValueError('Object is missing an id.')

    if object_type not in OBJECT_TYPES:
        raise ValueError(
            'Object {} has unknown type: {}'.format(
                instance_id, object_type
            )
        )

    color = raw.get('color')
    parsed = parse_instance_id(instance_id)

    if (
        object_type in MULTI_INSTANCE_TYPES
        and instance_id != object_type
    ):
        if parsed is None:
            raise ValueError(
                'Instance id "{}" must be <color>_<type>_<n>.'.format(
                    instance_id
                )
            )
        if parsed.object_type != object_type:
            raise ValueError(
                'Instance id "{}" does not match type "{}".'.format(
                    instance_id, object_type
                )
            )
        if color is None:
            color = parsed.color
        if color not in SCENE_COLORS:
            raise ValueError(
                'Object {} has invalid color: {}'.format(
                    instance_id, color
                )
            )
        if parsed.color != color:
            raise ValueError(
                'Instance id "{}" color conflicts with "{}".'.format(
                    instance_id, color
                )
            )
    elif instance_id not in CANONICAL_OBJECTS:
        raise ValueError('Unknown singleton object id: {}'.format(instance_id))

    support = raw.get('support', 'table')
    if not isinstance(support, str) or not support:
        raise ValueError(
            'Object {} has an invalid support.'.format(instance_id)
        )

    return ObjectInstance(
        instance_id=instance_id,
        object_type=object_type,
        color=color,
        position=_as_tuple3(raw.get('position'), 'position'),
        support=support,
        visible=bool(raw.get('visible', True)),
    )


def event_from_dict(raw):
    # type: (dict) -> EventScript
    event = raw.get('event')
    target = raw.get('target')

    if event not in EVENT_TYPES:
        raise ValueError('Unknown event type: {}'.format(event))
    if not isinstance(target, str) or not target:
        raise ValueError('Event is missing a target id.')

    index = raw.get('after_action_index', 1)
    if not isinstance(index, int) or index < 1:
        raise ValueError('after_action_index must be >= 1.')

    new_position = raw.get('new_position')
    if new_position is not None:
        new_position = _as_tuple3(new_position, 'new_position')

    return EventScript(
        event=event,
        target=target,
        after_action_index=index,
        new_position=new_position,
        new_support=raw.get('new_support'),
        new_visibility=raw.get('new_visibility'),
    )


def scene_spec_from_dict(raw):
    # type: (dict) -> SceneSpec
    category = raw.get('category')
    if category not in SCENARIO_CATEGORIES:
        raise ValueError('Unknown scenario category: {}'.format(category))

    objects = tuple(
        object_instance_from_dict(item)
        for item in raw.get('objects', [])
    )
    events = tuple(event_from_dict(item) for item in raw.get('events', []))
    goal_raw = raw.get('goal', {})
    goal = ScenarioGoal(
        category=goal_raw.get('category', category),
        params=dict(goal_raw.get('params', {})),
    )

    seed = raw.get('seed')
    if not isinstance(seed, int):
        raise ValueError('seed must be an integer.')

    command = raw.get('command')
    if not isinstance(command, str) or not command.strip():
        raise ValueError('command must be a non-empty string.')

    spec = SceneSpec(
        scenario_id=raw['scenario_id'],
        category=category,
        seed=seed,
        command=command,
        objects=objects,
        events=events,
        goal=goal,
        schema_version=raw.get(
            'schema_version', SCENARIO_SCHEMA_VERSION
        ),
    )
    validate(spec)
    return spec


# ---------------------------------------------------------------------------
# Validation
# ---------------------------------------------------------------------------

def _in_range(position, x_range, y_range, z_range):
    # type: (...) -> bool
    return (
        x_range[0] <= position[0] <= x_range[1]
        and y_range[0] <= position[1] <= y_range[1]
        and z_range[0] <= position[2] <= z_range[1]
    )


def _footprints_separated(left, right):
    # type: (ObjectInstance, ObjectInstance) -> bool
    left_hx, left_hy = left.footprint
    right_hx, right_hy = right.footprint
    gap_x = abs(left.position[0] - right.position[0])
    gap_y = abs(left.position[1] - right.position[1])
    return (
        gap_x >= left_hx + right_hx + MIN_OBJECT_GAP
        or gap_y >= left_hy + right_hy + MIN_OBJECT_GAP
    )


def _inside_container(obj, container):
    # type: (ObjectInstance, ObjectInstance) -> bool
    hx, hy = container.footprint
    return (
        abs(obj.position[0] - container.position[0]) < hx
        and abs(obj.position[1] - container.position[1]) < hy
    )


def validate_object_layout(spec):
    # type: (SceneSpec) -> None
    seen_ids = set()

    for obj in spec.objects:
        if obj.instance_id in seen_ids:
            raise ValueError('Duplicate object id: {}'.format(
                obj.instance_id))
        seen_ids.add(obj.instance_id)

        if not _in_range(
            obj.position,
            TABLE_X_RANGE,
            TABLE_Y_RANGE,
            TABLE_Z_RANGE
        ):
            raise ValueError(
                'Object {} is outside the tabletop reachable envelope: '
                '{}'.format(obj.instance_id, obj.position)
            )

    for index, left in enumerate(spec.objects):
        for right in spec.objects[index + 1:]:
            if not _footprints_separated(left, right):
                raise ValueError(
                    'Objects {} and {} overlap (gap < {}).'.format(
                        left.instance_id,
                        right.instance_id,
                        MIN_OBJECT_GAP
                    )
                )

    containers = [
        obj for obj in spec.objects if obj.receptacle == 'container'
    ]
    graspables = [obj for obj in spec.objects if obj.graspable]

    for obj in graspables:
        for container in containers:
            if obj.instance_id == container.instance_id:
                continue
            if _inside_container(obj, container):
                raise ValueError(
                    'Graspable {} starts inside container {}.'.format(
                        obj.instance_id, container.instance_id
                    )
                )


def validate_events(spec):
    # type: (SceneSpec) -> None
    known_ids = {obj.instance_id for obj in spec.objects}

    for evt in spec.events:
        if evt.target not in known_ids:
            raise ValueError(
                'Event {} targets unknown object: {}'.format(
                    evt.event, evt.target
                )
            )

        if evt.event in EVENTS_REPOSITION:
            if evt.new_position is None:
                raise ValueError(
                    '{} on {} requires new_position.'.format(
                        evt.event, evt.target
                    )
                )
            if not _in_range(
                evt.new_position,
                WORLD_X_RANGE,
                WORLD_Y_RANGE,
                WORLD_Z_RANGE
            ):
                raise ValueError(
                    'Event {} lands outside the world envelope: '
                    '{}'.format(evt.event, evt.new_position)
                )
            old = spec.object_by_id(evt.target)
            if (
                math.isclose(
                    old.position[0], evt.new_position[0], abs_tol=1e-6
                )
                and math.isclose(
                    old.position[1], evt.new_position[1], abs_tol=1e-6
                )
            ):
                raise ValueError(
                    '{} on {} does not change its position.'.format(
                        evt.event, evt.target
                    )
                )

            moved = ObjectInstance(
                instance_id=old.instance_id,
                object_type=old.object_type,
                color=old.color,
                position=evt.new_position,
                support=evt.new_support or old.support,
                visible=old.visible,
            )
            for other in spec.objects:
                if other.instance_id == evt.target:
                    continue
                if not _footprints_separated(moved, other):
                    raise ValueError(
                        '{} landing spot overlaps object {}.'.format(
                            evt.event, other.instance_id
                        )
                    )

        if evt.event == EVENT_GRASP_FAILURE:
            target_obj = spec.object_by_id(evt.target)
            if target_obj is None or not target_obj.graspable:
                raise ValueError(
                    'grasp_failure target must be graspable: {}'.format(
                        evt.target
                    )
                )

    if spec.category == CATEGORY_RECOVERY:
        slides = [
            evt for evt in spec.events
            if evt.event in EVENTS_REPOSITION
        ]
        if len(slides) != 1 or slides[0].after_action_index != 1:
            raise ValueError(
                'Recovery specs need exactly one reposition event with '
                'after_action_index=1.'
            )


def validate_goal(spec):
    # type: (SceneSpec) -> None
    goal = spec.goal

    if goal.category != spec.category:
        raise ValueError('Goal category does not match spec category.')

    def require_id(key):
        instance_id = goal.params.get(key)
        if spec.object_by_id(instance_id) is None:
            raise ValueError(
                'Goal parameter {} does not name a scene object: '
                '{}'.format(key, instance_id)
            )
        return instance_id

    if goal.category == CATEGORY_CLASSIFICATION:
        pairs = goal.params.get('pairs')
        if not isinstance(pairs, list) or not pairs:
            raise ValueError('classification goal needs non-empty pairs.')
        for pair in pairs:
            color = pair.get('color')
            bin_id = pair.get('bin_id')
            if color not in SCENE_COLORS:
                raise ValueError('Invalid pair color: {}'.format(color))
            bin_obj = spec.object_by_id(bin_id)
            if bin_obj is None or bin_obj.object_type != 'bin':
                raise ValueError(
                    'Pair target must be a bin instance: {}'.format(bin_id)
                )
            if bin_obj.color != color:
                raise ValueError(
                    'Bin {} color does not match pair color {}.'.format(
                        bin_id, color
                    )
                )
            if not spec.instances_of('block', color):
                raise ValueError(
                    'No {} block exists for pair {}.'.format(
                        color, bin_id
                    )
                )

    elif goal.category == CATEGORY_TARGET_PICK:
        target_id = require_id('target_id')
        require_id('container_id')
        target = spec.object_by_id(target_id)
        if not target.graspable:
            raise ValueError('target_pick target must be graspable.')

    elif goal.category == CATEGORY_SEQUENTIAL:
        for key in ('first', 'second'):
            stage = goal.params.get(key)
            if not isinstance(stage, dict):
                raise ValueError(
                    'sequential goal is missing stage "{}".'.format(key)
                )
            obj = spec.object_by_id(stage.get('object_id'))
            container = spec.object_by_id(stage.get('container_id'))
            if obj is None or not obj.graspable:
                raise ValueError(
                    'Stage {} object must be a graspable instance.'.format(
                        key
                    )
                )
            if container is None or container.receptacle is None:
                raise ValueError(
                    'Stage {} container must be a receptacle.'.format(key)
                )

    elif goal.category == CATEGORY_RECOVERY:
        target_id = require_id('target_id')
        require_id('container_id')
        target = spec.object_by_id(target_id)
        if not target.graspable:
            raise ValueError('recovery target must be graspable.')

    elif goal.category == CATEGORY_SAFE_REJECTION:
        subtype = goal.params.get('subtype')
        if subtype not in REJECTION_SUBTYPES:
            raise ValueError(
                'Invalid safe_rejection subtype: {}'.format(subtype)
            )


def validate(spec):
    # type: (SceneSpec) -> None
    """Validate all structural and layout invariants of one spec."""
    if not isinstance(spec.scenario_id, str) or not spec.scenario_id:
        raise ValueError('scenario_id must be a non-empty string.')
    if spec.category not in SCENARIO_CATEGORIES:
        raise ValueError(
            'Unknown category: {}'.format(spec.category)
        )
    validate_object_layout(spec)
    validate_events(spec)
    validate_goal(spec)
