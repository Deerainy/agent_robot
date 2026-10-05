"""Structured robot skill protocol for EmbodiedPlan.

This module is intentionally free of ROS and PyBullet imports so that the
planner, the executor and future plan validators share one source of truth
for the high-level robot skill catalog.

Only three high-level skills are exposed to the LLM:

- ``move_to``: move the end effector above an object
- ``pick``: grasp and lift an object
- ``place``: put the held object onto/in a target

Gripper opening/closing exists only as an internal PyBullet primitive and
is therefore intentionally absent from :data:`SKILL_SPECS`.
"""

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional


class ActionSchemaError(ValueError):
    """A raw action does not match the structured skill protocol."""


class SkillExecutionError(RuntimeError):
    """A registered skill handler failed during execution."""


@dataclass(frozen=True)
class SkillSpec:
    """Declaration of one high-level robot skill."""

    name: str
    required: List[str] = field(default_factory=list)
    optional: List[str] = field(default_factory=list)
    description: str = ''

    @property
    def known_params(self):
        return tuple(self.required) + tuple(self.optional)


# The high-level skill catalog. Insertion order is the canonical order used
# in planner prompts.
SKILL_SPECS = {
    'move_to': SkillSpec(
        name='move_to',
        required=['object'],
        description='Move the robot end effector above the named object.'
    ),
    'pick': SkillSpec(
        name='pick',
        required=['object'],
        description='Grasp the named object and lift it from the surface.'
    ),
    'place': SkillSpec(
        name='place',
        required=['object', 'target'],
        description=(
            'Place the currently held object onto or into the named target.'
        )
    ),
}

SKILL_NAMES = tuple(SKILL_SPECS.keys())


@dataclass(frozen=True)
class SkillAction:
    """One validated structured action."""

    skill: str
    params: Dict[str, str] = field(default_factory=dict)

    @property
    def object(self):
        # type: () -> Optional[str]
        return self.params.get('object')

    @property
    def target(self):
        # type: () -> Optional[str]
        return self.params.get('target')

    def to_dict(self):
        # type: () -> Dict[str, str]
        data = {'skill': self.skill}
        data.update(self.params)
        return data

    def describe(self):
        # type: () -> str
        if self.target is not None:
            return '{skill}({obj} -> {target})'.format(
                skill=self.skill,
                obj=self.object,
                target=self.target
            )

        return '{skill}({obj})'.format(
            skill=self.skill,
            obj=self.object
        )


def _location_label(index):
    # type: (Optional[int]) -> str
    if index is None:
        return 'action'
    return 'actions[{}]'.format(index)


def parse_action(raw, index=None):
    # type: (Any, Optional[int]) -> SkillAction
    """Validate one raw action dictionary into a :class:`SkillAction`.

    Raises :class:`ActionSchemaError` for any protocol violation. Unknown
    skills (including the internal ``open_gripper`` / ``close_gripper``
    primitives) are rejected.
    """
    location = _location_label(index)

    if not isinstance(raw, dict):
        raise ActionSchemaError(
            '{} must be a JSON object, got {}.'.format(
                location, type(raw).__name__
            )
        )

    skill = raw.get('skill')

    if not isinstance(skill, str) or not skill.strip():
        raise ActionSchemaError(
            '{} is missing a non-empty "skill" string.'.format(location)
        )

    skill = skill.strip()

    if skill not in SKILL_SPECS:
        raise ActionSchemaError(
            '{} uses unknown skill "{}". Allowed skills: {}.'.format(
                location,
                skill,
                ', '.join(SKILL_NAMES)
            )
        )

    spec = SKILL_SPECS[skill]
    params = {}

    for param_name in spec.required:
        value = raw.get(param_name)

        if not isinstance(value, str) or not value.strip():
            raise ActionSchemaError(
                '{} skill "{}" requires a non-empty "{}" string.'.format(
                    location, skill, param_name
                )
            )

        params[param_name] = value.strip()

    for param_name in spec.optional:
        value = raw.get(param_name)

        if value is None:
            continue

        if not isinstance(value, str) or not value.strip():
            raise ActionSchemaError(
                '{} optional "{}" of skill "{}" must be a non-empty string.'
                .format(location, param_name, skill)
            )

        params[param_name] = value.strip()

    allowed = set(spec.known_params) | {'skill'}
    unexpected = sorted(set(raw.keys()) - allowed)

    if unexpected:
        raise ActionSchemaError(
            '{} skill "{}" has unknown parameter(s): {}.'.format(
                location, skill, ', '.join(unexpected)
            )
        )

    return SkillAction(skill=skill, params=params)


def parse_actions(raw_list):
    # type: (Any) -> List[SkillAction]
    """Validate a non-empty list of raw action dictionaries."""
    if not isinstance(raw_list, list):
        raise ActionSchemaError(
            '"actions" must be a list, got {}.'.format(
                type(raw_list).__name__
            )
        )

    if not raw_list:
        raise ActionSchemaError('"actions" must not be empty.')

    return [
        parse_action(raw, index=index)
        for index, raw in enumerate(raw_list)
    ]


class SkillRegistry(object):
    """Maps high-level skill names to backend handler callables.

    A handler has the signature ``handler(action, context)`` where *action*
    is a :class:`SkillAction` and *context* is an optional backend-owned
    object (simulation handles, execution state, ...).
    """

    def __init__(self):
        self._handlers = {}  # type: Dict[str, Callable]

    def register(self, skill_name, handler):
        # type: (str, Callable) -> None
        if skill_name not in SKILL_SPECS:
            raise ValueError(
                'Cannot register unknown skill "{}". Allowed: {}.'.format(
                    skill_name, ', '.join(SKILL_NAMES)
                )
            )

        if not callable(handler):
            raise TypeError(
                'Handler for skill "{}" must be callable.'.format(skill_name)
            )

        self._handlers[skill_name] = handler

    def has_handler(self, skill_name):
        # type: (str) -> bool
        return skill_name in self._handlers

    def registered_names(self):
        # type: () -> List[str]
        return list(self._handlers.keys())

    def execute(self, action, context=None):
        """Dispatch one validated action to its registered handler.

        Any handler exception is normalized into
        :class:`SkillExecutionError` so callers have one failure channel.
        """
        handler = self._handlers.get(action.skill)

        if handler is None:
            raise SkillExecutionError(
                'No handler is registered for skill "{}".'.format(
                    action.skill
                )
            )

        try:
            return handler(action, context)
        except SkillExecutionError:
            raise
        except Exception as error:
            raise SkillExecutionError(
                'Skill "{}" failed during execution: {}'.format(
                    action.skill, error
                )
            )


def skill_catalog_text():
    # type: () -> str
    """Render the skill catalog for the planner prompt."""
    lines = ['可用技能（actions 中只能出现以下 skill）：']

    examples = {
        'move_to': {'skill': 'move_to', 'object': 'apple'},
        'pick': {'skill': 'pick', 'object': 'apple'},
        'place': {'skill': 'place', 'object': 'apple', 'target': 'basket'},
    }

    for index, name in enumerate(SKILL_NAMES, start=1):
        spec = SKILL_SPECS[name]
        required = ', '.join(spec.required)
        lines.append(
            '{index}. {name}（必填参数：{required}）'.format(
                index=index, name=name, required=required
            )
        )
        lines.append('   含义：{}'.format(spec.description))
        lines.append(
            '   示例：{example}'.format(example=examples[name])
        )

    return '\n'.join(lines)
