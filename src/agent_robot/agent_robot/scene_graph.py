"""Scene graph and plan precondition validation for EmbodiedPlan (M2).

This module is intentionally free of ROS and PyBullet imports so that the
executor, the planner and the offline tests share one symbolic world model.

World model (STRIPS-style, deliberately minimal):

- ``objects``   : catalogued manipulable objects (apple/cup/basket) with
                  ``graspable`` / ``receptacle`` properties;
- ``surfaces``  : support faces coming from perception (table/floor/...).
                  Known faces are placeable surfaces; unknown location words
                  are kept verbatim (never silently remapped to "table") and
                  are not placeable;
- ``holding``   : the single object held by the robot, or ``None``;
- ``at``        : the object the end effector has most recently moved above;
- ``supports``  : one ``on``/``in`` support relation per manipulable object.

Skill semantics (the only three high-level skills, see skill_registry):

- ``move_to(o)`` requires the object to exist; effect ``at = o``;
- ``pick(o)``    requires o graspable, empty hand, ``at == o``;
- ``place(o,t)`` requires ``holding == o``, a placeable target, ``at == t``.
"""

import re
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

from agent_robot.skill_registry import SkillAction


# ---------------------------------------------------------------------------
# Object grounding (single source of truth, moved out of task_executor in M2)
# ---------------------------------------------------------------------------

CANONICAL_OBJECTS = ('apple', 'cup', 'basket')

_COLOR_WORDS = {
    'red', 'blue', 'brown', 'green', 'yellow',
    'black', 'white', 'orange', 'purple', 'gray', 'grey'
}

_CHINESE_ALIASES = {
    '苹果': 'apple',
    '篮子': 'basket',
    '篮筐': 'basket',
    '杯子': 'cup',
    '水杯': 'cup',
}


class GroundingError(ValueError):
    """An action parameter cannot be mapped to a catalogued object."""


def semantic_tokens(text):
    # type: (str) -> List[str]
    """Extract color-free semantic tokens from an object name."""
    if text in _CHINESE_ALIASES:
        return [_CHINESE_ALIASES[text]]

    tokens = re.findall(r'[a-z0-9]+', text.lower())
    return [token for token in tokens if token not in _COLOR_WORDS]


def ground_name(query, env_names):
    # type: (str, List[str]) -> Optional[str]
    """Map an action parameter to a canonical catalogued object name.

    Matching order: Chinese alias, normalized exact match against perceived
    names, semantic token overlap (color adjectives ignored, so
    ``red_apple`` grounds ``apple``), then canonical self-fallback.
    Returns ``None`` when the result is not a catalogued object.
    """
    if not isinstance(query, str) or not query.strip():
        return None

    normalized_query = query.strip().lower()

    if normalized_query in _CHINESE_ALIASES:
        canonical = _CHINESE_ALIASES[normalized_query]
        if canonical in CANONICAL_OBJECTS:
            return canonical
        return None

    query_tokens = semantic_tokens(normalized_query)
    matched_env_name = None

    for env_name in env_names:
        normalized_env = env_name.strip().lower()

        if normalized_query == normalized_env:
            matched_env_name = normalized_env
            break

        env_tokens = semantic_tokens(normalized_env)

        if query_tokens and set(query_tokens) & set(env_tokens):
            matched_env_name = normalized_env
            break

    if matched_env_name is None:
        # The environment may be missing while the query itself is a
        # canonical scene name (debugging / fixed-scene scenarios).
        matched_env_name = normalized_query

    for token in semantic_tokens(matched_env_name):
        if token in CANONICAL_OBJECTS:
            return token

    return None


def ground_action(action, env_names):
    # type: (SkillAction, List[str]) -> SkillAction
    """Return a copy of *action* with grounded canonical object names."""
    grounded = {}

    for param_name, value in action.params.items():
        canonical = ground_name(value, env_names)

        if canonical is None:
            raise GroundingError(
                'Cannot ground {} parameter "{}" against scene objects: '
                '{}'.format(action.skill, param_name, value)
            )

        grounded[param_name] = canonical

    return SkillAction(skill=action.skill, params=grounded)


# ---------------------------------------------------------------------------
# Object / surface catalog
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ObjectInfo:
    """Static properties of one catalogued manipulable object."""

    name: str
    graspable: bool
    receptacle: Optional[str]  # 'container' | 'surface' | None


OBJECT_CATALOG = {
    'apple': ObjectInfo(name='apple', graspable=True, receptacle=None),
    'cup': ObjectInfo(name='cup', graspable=True, receptacle=None),
    'basket': ObjectInfo(
        name='basket', graspable=False, receptacle='container'
    ),
}

# Perceived location words that are known placeable surfaces.
SUPPORT_KINDS = {
    'table': 'surface',
    'desk': 'surface',
    'floor': 'surface',
    'ground': 'surface',
}

SCENE_GRAPH_VERSION = '1.0'

# Scene graph source tags.
SOURCE_DEFAULT = 'catalog_default'
SOURCE_PERCEPTION = 'perception'
SOURCE_EXECUTION = 'execution'


@dataclass
class ObjectNode:
    name: str
    graspable: bool
    receptacle: Optional[str]


@dataclass
class SurfaceNode:
    """A support face; known=False means perception used an unknown word."""

    name: str
    receptacle: Optional[str]
    known: bool


# ---------------------------------------------------------------------------
# Precondition violation reason codes
# ---------------------------------------------------------------------------

OBJECT_NOT_IN_SCENE = 'object_not_in_scene'
OBJECT_NOT_GRASPABLE = 'object_not_graspable'
HAND_OCCUPIED = 'hand_occupied'
NOT_HOLDING_OBJECT = 'not_holding_object'
ROBOT_NOT_AT_OBJECT = 'robot_not_at_object'
TARGET_NOT_RECEPTACLE = 'target_not_receptacle'


@dataclass
class Violation:
    index: int
    action: SkillAction
    reason_code: str
    detail: str


@dataclass
class ValidationReport:
    ok: bool
    checks: List[bool]
    violation: Optional[Violation]
    final_graph: 'SceneGraph'

    @property
    def reason_code(self):
        return self.violation.reason_code if self.violation else ''

    @property
    def reason_detail(self):
        return self.violation.detail if self.violation else ''


class SceneGraph(object):
    """Symbolic world state with skill preconditions and effects."""

    def __init__(self, source):
        # type: (str) -> None
        self.source = source
        self.objects = {}  # type: Dict[str, ObjectNode]
        self.surfaces = {}  # type: Dict[str, SurfaceNode]
        self.supports = {}  # type: Dict[str, Tuple[str, str]]
        self.holding = None  # type: Optional[str]
        self.robot_at = None  # type: Optional[str]
        self.warnings = []  # type: List[str]

    # ------------------------------------------------------------------
    # Construction
    # ------------------------------------------------------------------

    @classmethod
    def build_default(cls):
        """Fixed tabletop scene for mock / fixed PyBullet backends only."""
        graph = cls(source=SOURCE_DEFAULT)
        graph._ensure_surface('table')

        for name, info in OBJECT_CATALOG.items():
            graph._add_catalog_object(name)
            graph._set_support(name, 'on', 'table')

        return graph

    @classmethod
    def build_from_environment(cls, environment):
        """Build from a flat ``/environment_state`` payload.

        Unknown location words are preserved as non-placeable surfaces and
        recorded in ``warnings`` (never silently remapped to "table").
        """
        graph = cls(source=SOURCE_PERCEPTION)

        for raw in environment.get('objects', []):
            if not isinstance(raw, dict):
                continue

            canonical = ground_name(raw.get('name', ''), [])

            if canonical is None:
                graph.warnings.append(
                    'unknown_object:{}'.format(raw.get('name', ''))
                )
                continue

            graph._add_catalog_object(canonical)

            location = raw.get('location')

            if not isinstance(location, str) or not location.strip():
                graph.warnings.append(
                    'missing_support:{}'.format(canonical)
                )
                continue

            word = location.strip().lower()
            graph._ensure_surface(word)
            graph._set_support(canonical, 'on', word)

        return graph

    def _add_catalog_object(self, name):
        # type: (str) -> None
        info = OBJECT_CATALOG[name]
        self.objects[name] = ObjectNode(
            name=name,
            graspable=info.graspable,
            receptacle=info.receptacle
        )

    def _ensure_surface(self, word):
        # type: (str) -> str
        if word in self.surfaces:
            return word

        kind = SUPPORT_KINDS.get(word)

        if kind is None:
            self.surfaces[word] = SurfaceNode(
                name=word, receptacle=None, known=False
            )
            warning = 'unknown_support:{}'.format(word)

            if warning not in self.warnings:
                self.warnings.append(warning)
        else:
            self.surfaces[word] = SurfaceNode(
                name=word, receptacle=kind, known=True
            )

        return word

    def _set_support(self, name, relation, target):
        # type: (str, str, str) -> None
        self.supports[name] = (relation, target)

    # ------------------------------------------------------------------
    # Queries
    # ------------------------------------------------------------------

    def node(self, name):
        # type: (str) -> Optional[object]
        return self.objects.get(name) or self.surfaces.get(name)

    def clone(self):
        # type: () -> SceneGraph
        graph = SceneGraph(source=self.source)
        graph.objects = dict(self.objects)
        graph.surfaces = dict(self.surfaces)
        graph.supports = dict(self.supports)
        graph.holding = self.holding
        graph.robot_at = self.robot_at
        graph.warnings = list(self.warnings)
        return graph

    # ------------------------------------------------------------------
    # Preconditions / effects
    # ------------------------------------------------------------------

    def check_preconditions(self, action):
        # type: (SkillAction) -> Optional[Tuple[str, str]]
        """Return ``(reason_code, detail)`` on violation, else ``None``."""
        if action.skill == 'move_to':
            if action.object not in self.objects:
                return (
                    OBJECT_NOT_IN_SCENE,
                    'move_to target "{}" is not an object in the scene '
                    'graph.'.format(action.object)
                )
            return None

        if action.skill == 'pick':
            obj = self.objects.get(action.object)

            if obj is None:
                return (
                    OBJECT_NOT_IN_SCENE,
                    'Cannot pick "{}": it is not an object in the scene '
                    'graph.'.format(action.object)
                )

            if not obj.graspable:
                return (
                    OBJECT_NOT_GRASPABLE,
                    'Object "{}" is not graspable.'.format(obj.name)
                )

            if self.holding is not None:
                return (
                    HAND_OCCUPIED,
                    'Gripper is already holding "{}".'.format(self.holding)
                )

            if self.robot_at != obj.name:
                return (
                    ROBOT_NOT_AT_OBJECT,
                    'pick("{}") requires move_to("{}") first; robot is '
                    'at "{}".'.format(
                        obj.name, obj.name, self.robot_at
                    )
                )

            return None

        if action.skill == 'place':
            if self.holding != action.object:
                if self.holding is None:
                    detail = (
                        'place("{}") requires picking it first; the gripper '
                        'is empty.'.format(action.object)
                    )
                else:
                    detail = (
                        'place("{}") requires holding it; the gripper is '
                        'holding "{}".'.format(
                            action.object, self.holding
                        )
                    )
                return NOT_HOLDING_OBJECT, detail

            target = self.node(action.target)

            if target is None:
                return (
                    OBJECT_NOT_IN_SCENE,
                    'place target "{}" is not in the scene graph.'.format(
                        action.target
                    )
                )

            if target.receptacle is None:
                return (
                    TARGET_NOT_RECEPTACLE,
                    '"{}" cannot receive placed objects.'.format(
                        action.target
                    )
                )

            if self.robot_at != action.target:
                return (
                    ROBOT_NOT_AT_OBJECT,
                    'place to "{}" requires move_to("{}") first; robot is '
                    'at "{}".'.format(
                        action.target, action.target, self.robot_at
                    )
                )

            return None

        return (
            OBJECT_NOT_IN_SCENE,
            'Unknown skill "{}".'.format(action.skill)
        )

    def apply(self, action):
        # type: (SkillAction) -> None
        """Apply a precondition-checked action's effects to this graph."""
        if action.skill == 'move_to':
            self.robot_at = action.object
        elif action.skill == 'pick':
            self.holding = action.object
            self.supports.pop(action.object, None)
        elif action.skill == 'place':
            target = self.node(action.target)
            relation = (
                'in'
                if target is not None and target.receptacle == 'container'
                else 'on'
            )
            self._set_support(action.object, relation, action.target)
            self.holding = None

    # ------------------------------------------------------------------
    # Serialization
    # ------------------------------------------------------------------

    def to_dict(self):
        objects = []

        for name in sorted(self.objects):
            node = self.objects[name]
            support = self.supports.get(name)
            objects.append(
                {
                    'name': name,
                    'graspable': node.graspable,
                    'receptacle': node.receptacle,
                    'support': (
                        {'relation': support[0], 'target': support[1]}
                        if support is not None
                        else None
                    ),
                }
            )

        surfaces = [
            {
                'name': name,
                'receptacle': node.receptacle,
                'known': node.known,
            }
            for name, node in sorted(self.surfaces.items())
        ]

        relations = []

        for name in sorted(self.supports):
            relation, target = self.supports[name]
            relations.append([relation, name, target])

        return {
            'schema_version': SCENE_GRAPH_VERSION,
            'source': self.source,
            'warnings': list(self.warnings),
            'robot': {'holding': self.holding, 'at': self.robot_at},
            'objects': objects,
            'surfaces': surfaces,
            'relations': relations,
        }

    @classmethod
    def from_dict(cls, data):
        # type: (dict) -> SceneGraph
        graph = cls(source=data.get('source', SOURCE_PERCEPTION))
        graph.warnings = list(data.get('warnings', []))

        for raw in data.get('objects', []):
            name = raw.get('name')

            if not isinstance(name, str):
                continue

            if name in OBJECT_CATALOG:
                graph._add_catalog_object(name)
            else:
                graph.objects[name] = ObjectNode(
                    name=name,
                    graspable=bool(raw.get('graspable')),
                    receptacle=raw.get('receptacle')
                )

            support = raw.get('support')

            if isinstance(support, dict):
                target = support.get('target')
                relation = support.get('relation', 'on')

                if isinstance(target, str):
                    graph._ensure_surface(target)
                    graph._set_support(name, relation, target)

        for raw in data.get('surfaces', []):
            name = raw.get('name')

            if not isinstance(name, str) or name in graph.surfaces:
                continue

            graph.surfaces[name] = SurfaceNode(
                name=name,
                receptacle=raw.get('receptacle'),
                known=bool(raw.get('known', True))
            )

        robot = data.get('robot', {})

        if isinstance(robot, dict):
            holding = robot.get('holding')
            robot_at = robot.get('at')

            if isinstance(holding, str):
                graph.holding = holding

            if isinstance(robot_at, str):
                graph.robot_at = robot_at

        return graph

    def to_prompt_text(self):
        # type: () -> str
        lines = ['场景物体与当前关系：']

        for name in sorted(self.objects):
            node = self.objects[name]
            properties = []

            if node.graspable:
                properties.append('可抓取')
            else:
                properties.append('不可抓取')

            if node.receptacle == 'container':
                properties.append('容器（可作为 place 的 target，物体放入后为 in 关系）')
            elif node.receptacle == 'surface':
                properties.append('支撑面（可作为 place 的 target）')

            if self.holding == name:
                state = '正被机器人握持'
            else:
                support = self.supports.get(name)

                if support is not None:
                    relation, target = support
                    state = '在 {} {}（{}）'.format(
                        target,
                        '内' if relation == 'in' else '上',
                        relation
                    )
                else:
                    state = '位置未知'

            lines.append(
                '- {name}（{properties}）：{state}'.format(
                    name=name,
                    properties='，'.join(properties),
                    state=state
                )
            )

        if self.robot_at is None:
            location = '未靠近任何物体'
        else:
            location = '位于 {} 上方'.format(self.robot_at)

        lines.append(
            '机器人：{}，{}。'.format(
                '握持 {}'.format(self.holding)
                if self.holding is not None
                else '手空',
                location
            )
        )

        for warning in self.warnings:
            lines.append('感知告警：{}'.format(warning))

        return '\n'.join(lines)


def validate_actions(actions, graph):
    # type: (List[SkillAction], SceneGraph) -> ValidationReport
    """Validate a full action sequence on a cloned graph, applying effects.

    The first violation (if any) stops validation; otherwise the returned
    ``final_graph`` is the predicted world state after every action.
    """
    simulated = graph.clone()
    checks = []
    violation = None

    for index, action in enumerate(actions):
        result = simulated.check_preconditions(action)

        if result is not None:
            reason_code, detail = result
            violation = Violation(
                index=index,
                action=action,
                reason_code=reason_code,
                detail=detail
            )
            checks.append(False)
            break

        checks.append(True)
        simulated.apply(action)

    return ValidationReport(
        ok=violation is None,
        checks=checks,
        violation=violation,
        final_graph=simulated
    )
