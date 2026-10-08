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
from dataclasses import dataclass, replace
from typing import Dict, List, NamedTuple, Optional, Tuple

from agent_robot.skill_registry import SkillAction


# ---------------------------------------------------------------------------
# Object grounding (single source of truth, moved out of task_executor in M2)
# ---------------------------------------------------------------------------

# Legacy singleton objects of the fixed tabletop scene (M1-M4).
CANONICAL_OBJECTS = ('apple', 'cup', 'basket')

# Colors assignable to indexed object instances.
SCENE_COLORS = ('red', 'blue', 'green', 'yellow')

CHINESE_COLOR_WORDS = {
    '红': 'red',
    '蓝': 'blue',
    '绿': 'green',
    '黄': 'yellow',
}

CHINESE_COLOR_WORDS_INV = {
    value: key for key, value in CHINESE_COLOR_WORDS.items()
}

_COLOR_WORDS = {
    'red', 'blue', 'brown', 'green', 'yellow',
    'black', 'white', 'orange', 'purple', 'gray', 'grey'
} | set(CHINESE_COLOR_WORDS.keys())

_CHINESE_ALIASES = {
    '苹果': 'apple',
    '篮子': 'basket',
    '篮筐': 'basket',
    '杯子': 'cup',
    '水杯': 'cup',
    # M5 multi-instance types.
    '积木': 'block',
    '箱子': 'bin',
    '收纳箱': 'bin',
    '盒子': 'bin',
}


class GroundingError(ValueError):
    """An action parameter cannot be mapped to a catalogued object."""


class AmbiguousGroundingError(GroundingError):
    """Ambiguous reference with sorted candidate instance ids."""

    def __init__(self, reference, candidates):
        # type: (str, List[str]) -> None
        self.reference = reference
        self.candidates = sorted(candidates)
        message = (
            'Reference "{}" is ambiguous; candidates: {}. '
            'Use a full instance id to disambiguate.'.format(
                reference, ', '.join(self.candidates)
            )
        )
        super(AmbiguousGroundingError, self).__init__(message)


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
    """Return a copy of *action* with grounded canonical object names.

    Legacy M1-M4 entry point: *env_names* is a list of perceived raw
    names and grounding collapses to canonical singleton names. New M5
    code should pass a :class:`SceneGraph` (the dispatch is automatic).
    """
    if isinstance(env_names, SceneGraph):
        return ground_action_on_graph(action, env_names)

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
# M5 multi-instance grounding: resolve (unique) vs query (set)
# ---------------------------------------------------------------------------

def instance_type_and_color(name):
    # type: (str) -> Optional[Tuple[str, Optional[str]]]
    """Return ``(object_type, color)`` for an object node name.

    Multi-instance ids (``red_block_1``) carry an explicit color; legacy
    singleton ids (``apple``) have color ``None``. Returns ``None`` when
    the name is neither.
    """
    parsed = parse_instance_id(name)

    if parsed is not None:
        return parsed.object_type, parsed.color

    if name in OBJECT_TYPES:
        return name, None

    return None


def normalize_type_word(word):
    # type: (str) -> Optional[str]
    """Map a Chinese alias or English word to a canonical object type."""
    if not isinstance(word, str):
        return None
    text = word.strip().lower()

    if text in ('box', 'boxes'):
        return 'bin'

    if text in OBJECT_TYPES:
        return text

    if text in _CHINESE_ALIASES:
        canonical = _CHINESE_ALIASES[text]
        return canonical if canonical in OBJECT_TYPES else None

    # Plural English tokens such as "blocks".
    singular = text[:-1] if text.endswith('s') else text
    if singular in OBJECT_TYPES:
        return singular

    return None


def normalize_color_word(word):
    # type: (str) -> Optional[str]
    """Map a Chinese/English color word to a canonical color."""
    if not isinstance(word, str):
        return None
    text = word.strip().lower()

    if text in SCENE_COLORS:
        return text

    for chinese, canonical in CHINESE_COLOR_WORDS.items():
        if chinese in text:
            return canonical

    return None


def parse_object_reference(reference):
    # type: (str) -> Tuple[Optional[str], Optional[str]]
    """Parse a free-form reference into ``(object_type, color)``.

    Handles English tokens ("red block", "blue_bin_2" without the index)
    and Chinese phrases ("红色积木", "蓝箱子"). Full instance ids are
    resolved by exact match in :func:`resolve_instance`.
    """
    text = reference.strip().lower()

    # English / latin token path.
    if re.search(r'[a-z]', text):
        object_type = None
        color = None
        for token in re.findall(r'[a-z]+', text):
            if token in SCENE_COLORS:
                color = token
                continue
            normalized_type = normalize_type_word(token)
            if normalized_type is not None:
                object_type = normalized_type
        return object_type, color

    # Chinese phrase path.
    object_type = None
    for alias in sorted(_CHINESE_ALIASES, key=len, reverse=True):
        if alias in text:
            object_type = _CHINESE_ALIASES[alias]
            break

    color = normalize_color_word(text)
    return object_type, color


def _visible_object_ids(graph):
    # type: (SceneGraph) -> List[str]
    ids = []
    for name, node in graph.objects.items():
        if getattr(node, 'visible', True):
            ids.append(name)
    return ids


def query_instances(graph, object_type=None, color=None):
    # type: (SceneGraph, Optional[str], Optional[str]) -> List[str]
    """Return the sorted instance ids matching a (type, color) set query.

    Either argument may be a Chinese alias/color word. Both ``None``
    returns every visible instance. The result is always a list; this
    function never collapses a set to a single string.
    """
    wanted_type = normalize_type_word(object_type) if object_type \
        else None
    wanted_color = normalize_color_word(color) if color else None

    matches = []
    for name in _visible_object_ids(graph):
        identity = instance_type_and_color(name)
        if identity is None:
            continue
        actual_type, actual_color = identity
        if wanted_type is not None and actual_type != wanted_type:
            continue
        if wanted_color is not None and actual_color != wanted_color:
            continue
        matches.append(name)

    return sorted(matches)


def resolve_instance(reference, graph):
    # type: (str, SceneGraph) -> str
    """Resolve a reference to exactly one visible instance id.

    Exact full-id match wins; otherwise the reference is parsed into
    type/color attributes and matched against visible instances. Raises
    :class:`GroundingError` on zero matches and
    :class:`AmbiguousGroundingError` (with candidates) on several.
    """
    if not isinstance(reference, str) or not reference.strip():
        raise GroundingError('Empty object reference.')

    normalized = reference.strip().lower()

    if normalized in graph.objects:
        node = graph.objects[normalized]
        if getattr(node, 'visible', True):
            return normalized
        raise GroundingError(
            'Object "{}" is currently not visible.'.format(normalized)
        )

    wanted_type, wanted_color = parse_object_reference(normalized)

    if wanted_type is None and wanted_color is None:
        raise GroundingError(
            'Reference "{}" names no known type or color.'.format(
                reference
            )
        )

    # Preserve legacy singleton grounding when a perceived apple has no
    # indexed color-bearing identity.
    if wanted_type is not None and \
            wanted_type not in MULTI_INSTANCE_TYPES:
        wanted_color = None

    candidates = query_instances(
        graph, object_type=wanted_type, color=wanted_color
    )
    if (
        not candidates
        and wanted_type == 'apple'
        and 'apple' in graph.objects
        and graph.objects['apple'].visible
    ):
        candidates = ['apple']

    if not candidates:
        available = _visible_object_ids(graph)
        raise GroundingError(
            'No visible instance matches reference "{}" '
            '(type={}, color={}). Visible objects: {}.'.format(
                reference,
                wanted_type,
                wanted_color,
                ', '.join(sorted(available)) or 'none'
            )
        )

    if len(candidates) == 1:
        return candidates[0]

    raise AmbiguousGroundingError(reference, candidates)


def ground_action_on_graph(action, graph):
    # type: (SkillAction, SceneGraph) -> SkillAction
    """Return a copy of *action* with every reference resolved uniquely."""
    grounded = {}

    for param_name, value in action.params.items():
        grounded[param_name] = resolve_instance(value, graph)

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

# M5: every type that may appear in a scenario, including multi-instance
# types (colorable blocks and bins). Keyed by canonical type name.
OBJECT_TYPES = dict(OBJECT_CATALOG)
OBJECT_TYPES.update({
    'block': ObjectInfo(
        name='block', graspable=True, receptacle=None
    ),
    'bin': ObjectInfo(
        name='bin', graspable=False, receptacle='container'
    ),
})

# Types created as numbered color instances in M5 scenarios.
MULTI_INSTANCE_TYPES = ('apple', 'block', 'bin')


class InstanceId(NamedTuple):
    """Parsed ``<color>_<type>_<index>`` instance identity."""

    color: str
    object_type: str
    index: int


_INSTANCE_ID_RE = re.compile(
    r'^(?P<color>[a-z]+)_(?P<object_type>[a-z]+)_(?P<index>[0-9]+)$'
)


def build_instance_id(object_type, color, index):
    # type: (str, str, int) -> str
    """Build a canonical instance id ``<color>_<type>_<index>``."""
    if object_type not in MULTI_INSTANCE_TYPES:
        raise ValueError(
            'Unknown multi-instance object type: {}'.format(object_type)
        )

    if color not in SCENE_COLORS:
        raise ValueError('Unknown instance color: {}'.format(color))

    if not isinstance(index, int) or index < 1:
        raise ValueError('Instance index must be a positive integer.')

    return '{}_{}_{}'.format(color, object_type, index)


def parse_instance_id(name):
    # type: (str) -> Optional[InstanceId]
    """Parse an instance id; return ``None`` for any other name shape."""
    if not isinstance(name, str):
        return None

    match = _INSTANCE_ID_RE.match(name.strip())

    if match is None:
        return None

    object_type = match.group('object_type')

    if object_type not in MULTI_INSTANCE_TYPES:
        return None

    color = match.group('color')

    if color not in SCENE_COLORS:
        return None

    return InstanceId(
        color=color,
        object_type=object_type,
        index=int(match.group('index'))
    )


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

# M5: an object whose moved distance (meters) is at or below this value is
# considered stationary (sensor jitter). Larger moves invalidate robot_at.
POSITION_EPSILON = 0.02

_TYPE_CHINESE = {
    'apple': '苹果',
    'cup': '杯子',
    'basket': '篮子',
    'block': '积木',
    'bin': '箱子',
}


@dataclass
class ObjectNode:
    name: str
    graspable: bool
    receptacle: Optional[str]
    color: Optional[str] = None
    position: Optional[Tuple[float, float, float]] = None
    visible: bool = True
    bbox: Optional[Tuple[int, int, int, int]] = None
    confidence: Optional[float] = None


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
        # M5 episode identity and monotonic world version.
        self.scenario_id = None  # type: Optional[str]
        self.world_revision = 0  # type: int

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
        graph.initialize_from_environment(environment)
        return graph

    @classmethod
    def build_from_perception(cls, perception_result):
        """Build directly from the structured perception environment."""
        if not isinstance(perception_result, dict):
            raise ValueError('Perception result must be a JSON object.')
        graph = cls(source=SOURCE_PERCEPTION)
        graph.initialize_from_environment(perception_result)
        return graph

    def initialize_from_environment(self, environment):
        # type: (dict) -> None
        """Reset all belief state and ingest one environment payload.

        Used for the first observation of an episode. Unlike
        :meth:`update_observation` this clears objects, supports, holding
        and robot_at unconditionally.
        """
        self.objects = {}
        self.surfaces = {}
        self.supports = {}
        self.holding = None
        self.robot_at = None
        self.warnings = []
        self.scenario_id = self._read_scenario_id(environment)
        self.world_revision = self._read_revision(environment)
        self._merge_environment(environment, initial=True)

    def update_observation(self, environment):
        # type: (dict) -> bool
        """Merge a newer environment frame without losing action state.

        ``holding`` is never cleared; external position/support writes for
        the held object are ignored. ``robot_at`` is invalidated when its
        object moves beyond ``POSITION_EPSILON`` or changes support.
        Revision frames older than the current one are ignored; identical
        revisions are idempotent no-ops. Returns ``True`` when the frame
        was applied.
        """
        revision = environment.get('world_revision')

        if revision is None:
            # Legacy revision-less M4 frames: always merge, keep rev=0.
            self._merge_environment(environment, initial=False)
            return True

        revision = int(revision)

        if revision < self.world_revision:
            return False
        if revision == self.world_revision:
            return False

        scenario_id = self._read_scenario_id(environment)
        if scenario_id is not None:
            self.scenario_id = scenario_id

        self._merge_environment(environment, initial=False)
        self.world_revision = revision
        return True

    @staticmethod
    def _read_revision(environment):
        revision = environment.get('world_revision')
        return int(revision) if isinstance(revision, int) else 0

    @staticmethod
    def _read_scenario_id(environment):
        scenario_id = environment.get('scenario_id')
        return scenario_id if isinstance(scenario_id, str) else None

    def _merge_environment(self, environment, initial):
        # type: (dict, bool) -> None
        incoming = {}

        for raw in environment.get('objects', []):
            if not isinstance(raw, dict):
                continue
            record = self._read_payload_object(raw)
            if record is None:
                continue
            incoming[record['id']] = record

        # Pass 1: node attributes (positions/visibility) so containers
        # already exist when support relations are applied.
        for instance_id, record in incoming.items():
            self._upsert_observed_node(instance_id, record)

        # Pass 2: support relations.
        for instance_id, record in incoming.items():
            if self.holding == instance_id:
                continue
            self._apply_observed_support(instance_id, record)

        if not initial:
            for instance_id, node in list(self.objects.items()):
                if instance_id not in incoming and node.visible:
                    self.objects[instance_id] = replace(
                        node, visible=False
                    )

    def _read_payload_object(self, raw):
        # type: (dict) -> Optional[dict]
        raw_id = raw.get('id')
        if not isinstance(raw_id, str) or not raw_id.strip():
            raw_id = raw.get('name')

        if not isinstance(raw_id, str) or not raw_id.strip():
            self.warnings.append('unknown_object:')
            return None

        parsed = parse_instance_id(raw_id)
        explicit_type = raw.get('type')

        if parsed is not None:
            instance_id = raw_id.strip()
            object_type = parsed.object_type
            color = parsed.color
            if (
                isinstance(explicit_type, str)
                and explicit_type in OBJECT_TYPES
                and explicit_type != object_type
            ):
                self.warnings.append(
                    'type_mismatch:{}'.format(instance_id)
                )
        elif (
            isinstance(explicit_type, str)
            and explicit_type in OBJECT_TYPES
        ):
            object_type = explicit_type
            if (
                object_type in MULTI_INSTANCE_TYPES
                and raw_id != object_type
            ):
                self.warnings.append(
                    'unindexed_instance:{}'.format(raw_id)
                )
                return None
            instance_id = object_type
            color = None
        else:
            canonical = ground_name(raw_id, [])
            if canonical is None:
                self.warnings.append(
                    'unknown_object:{}'.format(raw_id)
                )
                return None
            instance_id = canonical
            object_type = canonical
            color = None

        if color is None and isinstance(raw.get('color'), str):
            explicit_color = raw['color']
            if explicit_color in SCENE_COLORS:
                color = explicit_color

        position = self._read_position(raw.get('position'))
        visible = bool(raw.get('visible', True))
        location = raw.get('location')
        if not isinstance(location, str) or not location.strip():
            location = raw.get('support')
        relation = raw.get('relation')

        return {
            'id': instance_id,
            'type': object_type,
            'color': color,
            'position': position,
            'visible': visible,
            'bbox': self._read_bbox(raw.get('bbox')),
            'confidence': self._read_confidence(raw.get('confidence')),
            'location': (
                location.strip().lower()
                if isinstance(location, str) and location.strip()
                else None
            ),
            'relation': relation if isinstance(relation, str) else None,
        }

    @staticmethod
    def _read_bbox(raw_bbox):
        if not isinstance(raw_bbox, (list, tuple)) or len(raw_bbox) != 4:
            return None
        try:
            bbox = tuple(int(value) for value in raw_bbox)
        except (TypeError, ValueError):
            return None
        if bbox[2] <= bbox[0] or bbox[3] <= bbox[1]:
            return None
        return bbox

    @staticmethod
    def _read_confidence(raw_confidence):
        if raw_confidence is None:
            return None
        try:
            confidence = float(raw_confidence)
        except (TypeError, ValueError):
            return None
        if not 0.0 <= confidence <= 1.0:
            return None
        return confidence

    @staticmethod
    def _read_position(raw_position):
        if not isinstance(raw_position, (list, tuple)):
            return None
        if len(raw_position) != 3:
            return None
        try:
            return (
                float(raw_position[0]),
                float(raw_position[1]),
                float(raw_position[2]),
            )
        except (TypeError, ValueError):
            return None

    def _upsert_observed_node(self, instance_id, record):
        # type: (str, dict) -> None
        info = OBJECT_TYPES[record['type']]
        held = self.holding == instance_id

        previous = self.objects.get(instance_id)
        previous_position = (
            previous.position if previous is not None else None
        )

        node = ObjectNode(
            name=instance_id,
            graspable=info.graspable,
            receptacle=info.receptacle,
            color=record['color'],
            position=record['position'],
            visible=record['visible'],
            bbox=record['bbox'],
            confidence=record['confidence'],
        )

        if held:
            # The held object follows the gripper; external truth writes
            # about it must not overwrite execution state.
            warning = 'held_object_external_update:{}'.format(
                instance_id
            )
            if warning not in self.warnings:
                self.warnings.append(warning)
            node.visible = True
            if previous is not None:
                node.position = previous.position

        self.objects[instance_id] = node

        if (
            not held
            and self.robot_at == instance_id
            and self._moved_beyond_epsilon(
                previous_position, record['position']
            )
        ):
            self.robot_at = None

    def _apply_observed_support(self, instance_id, record):
        # type: (str, dict) -> None
        location = record['location']
        previous_support = self.supports.get(instance_id)

        if location is None:
            if previous_support is None:
                self.warnings.append(
                    'missing_support:{}'.format(instance_id)
                )
            return

        surface = self.surfaces.get(location)
        container = self.objects.get(location)

        if surface is not None:
            relation = record['relation'] or 'on'
        elif (
            container is not None
            and container.receptacle == 'container'
        ):
            relation = record['relation'] or 'in'
        else:
            self._ensure_surface(location)
            relation = record['relation'] or 'on'

        self._set_support(instance_id, relation, location)

        new_support = self.supports.get(instance_id)
        if self.robot_at == instance_id and \
                previous_support != new_support:
            self.robot_at = None

    @staticmethod
    def _moved_beyond_epsilon(old_position, new_position):
        if old_position is None or new_position is None:
            return False
        delta = sum(
            (old_position[index] - new_position[index]) ** 2
            for index in range(3)
        ) ** 0.5
        return delta > POSITION_EPSILON

    def apply_action_effect(self, action):
        # type: (SkillAction) -> None
        """Apply a verified action's effects (semantic alias of apply)."""
        self.apply(action)

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
        graph.scenario_id = self.scenario_id
        graph.world_revision = self.world_revision
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
            if not self.objects[action.object].visible:
                return (
                    OBJECT_NOT_IN_SCENE,
                    'move_to target "{}" is currently not visible.'
                    .format(action.object)
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

            if not obj.visible:
                return (
                    OBJECT_NOT_IN_SCENE,
                    'Cannot pick "{}": it is currently not visible.'
                    .format(action.object)
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

            if not getattr(target, 'visible', True):
                return (
                    OBJECT_NOT_IN_SCENE,
                    'place target "{}" is currently not visible.'.format(
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
                    'color': node.color,
                    'position': (
                        list(node.position)
                        if node.position is not None
                        else None
                    ),
                    'visible': node.visible,
                    'bbox': (
                        list(node.bbox) if node.bbox is not None else None
                    ),
                    'confidence': node.confidence,
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
            'scenario_id': self.scenario_id,
            'world_revision': self.world_revision,
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
        graph.scenario_id = (
            data['scenario_id']
            if isinstance(data.get('scenario_id'), str)
            else None
        )
        graph.world_revision = (
            int(data['world_revision'])
            if isinstance(data.get('world_revision'), int)
            else 0
        )

        # Pass 1: object nodes (so container support targets exist first).
        pending_supports = []

        for raw in data.get('objects', []):
            name = raw.get('name')

            if not isinstance(name, str):
                continue

            parsed = parse_instance_id(name)

            if name in OBJECT_CATALOG:
                info = OBJECT_CATALOG[name]
                node = ObjectNode(
                    name=name,
                    graspable=info.graspable,
                    receptacle=info.receptacle,
                    color=raw.get('color'),
                    position=cls._static_read_position(
                        raw.get('position')
                    ),
                    visible=bool(raw.get('visible', True)),
                    bbox=cls._read_bbox(raw.get('bbox')),
                    confidence=cls._read_confidence(
                        raw.get('confidence')
                    ),
                )
            elif parsed is not None:
                info = OBJECT_TYPES[parsed.object_type]
                node = ObjectNode(
                    name=name,
                    graspable=info.graspable,
                    receptacle=info.receptacle,
                    color=parsed.color,
                    position=cls._static_read_position(
                        raw.get('position')
                    ),
                    visible=bool(raw.get('visible', True)),
                    bbox=cls._read_bbox(raw.get('bbox')),
                    confidence=cls._read_confidence(
                        raw.get('confidence')
                    ),
                )
            else:
                node = ObjectNode(
                    name=name,
                    graspable=bool(raw.get('graspable')),
                    receptacle=raw.get('receptacle'),
                    color=raw.get('color'),
                    position=cls._static_read_position(
                        raw.get('position')
                    ),
                    visible=bool(raw.get('visible', True)),
                    bbox=cls._read_bbox(raw.get('bbox')),
                    confidence=cls._read_confidence(
                        raw.get('confidence')
                    ),
                )

            graph.objects[name] = node

            support = raw.get('support')
            if isinstance(support, dict):
                pending_supports.append(
                    (
                        name,
                        support.get('target'),
                        support.get('relation', 'on'),
                    )
                )

        for raw in data.get('surfaces', []):
            name = raw.get('name')

            if not isinstance(name, str) or name in graph.surfaces:
                continue

            graph.surfaces[name] = SurfaceNode(
                name=name,
                receptacle=raw.get('receptacle'),
                known=bool(raw.get('known', True))
            )

        # Pass 2: support relations; targets that are objects are not
        # re-registered as surfaces.
        for name, target, relation in pending_supports:
            if not isinstance(target, str):
                continue
            if target in graph.objects:
                graph._set_support(name, relation, target)
            else:
                graph._ensure_surface(target)
                graph._set_support(name, relation, target)

        robot = data.get('robot', {})

        if isinstance(robot, dict):
            holding = robot.get('holding')
            robot_at = robot.get('at')

            if isinstance(holding, str):
                graph.holding = holding

            if isinstance(robot_at, str):
                graph.robot_at = robot_at

        return graph

    @staticmethod
    def _static_read_position(raw_position):
        if not isinstance(raw_position, (list, tuple)):
            return None
        if len(raw_position) != 3:
            return None
        try:
            return (
                float(raw_position[0]),
                float(raw_position[1]),
                float(raw_position[2]),
            )
        except (TypeError, ValueError):
            return None

    def to_prompt_text(self):
        # type: () -> str
        lines = ['场景物体与当前关系：']

        for name in sorted(self.objects):
            node = self.objects[name]
            identity = instance_type_and_color(name)
            properties = []

            if identity is not None:
                type_label = _TYPE_CHINESE.get(
                    identity[0], identity[0]
                )
                if identity[1] is not None:
                    properties.append(
                        '{}/{}'.format(
                            '{}色'.format(
                                CHINESE_COLOR_WORDS_INV.get(
                                    identity[1], identity[1]
                                )
                            ),
                            type_label
                        )
                    )
                else:
                    properties.append(type_label)

            if node.graspable:
                properties.append('可抓取')
            else:
                properties.append('不可抓取')

            if node.receptacle == 'container':
                properties.append('容器（可作为 place 的 target，物体放入后为 in 关系）')
            elif node.receptacle == 'surface':
                properties.append('支撑面（可作为 place 的 target）')

            if not node.visible:
                properties.append('当前不可见')

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
