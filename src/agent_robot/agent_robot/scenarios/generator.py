"""
Deterministic seeded generators for the five M5 scenario categories.

All randomness comes from a private ``random.Random(seed)`` instance; the
global random module is never touched, so generated specs are fully
reproducible. Every generated spec passes :func:`spec.validate`.
"""

import random

from agent_robot.scene_graph import (
    CHINESE_COLOR_WORDS,
    SCENE_COLORS,
    build_instance_id,
)
from agent_robot.scenarios.spec import (
    CATEGORY_CLASSIFICATION,
    CATEGORY_RECOVERY,
    CATEGORY_SAFE_REJECTION,
    CATEGORY_SEQUENTIAL,
    CATEGORY_TARGET_PICK,
    EVENT_OBJECT_SLIDE,
    FOOTPRINT_HALF_EXTENTS,
    MIN_OBJECT_GAP,
    REJECT_MISSING_TARGET,
    REJECT_NOT_GRASPABLE,
    REJECT_WRONG_CONTAINER,
    SCENARIO_CATEGORIES,
    TABLE_Y_RANGE,
    EventScript,
    ObjectInstance,
    SceneSpec,
    ScenarioGoal,
    validate,
)

GENERATOR_VERSION = 'm5-1'

# Rest heights (object center z) per type; table surface is z=0.
REST_Z = {
    'apple': 0.06,
    'cup': 0.06,
    'basket': 0.05,
    'block': 0.025,
    'bin': 0.05,
}

# Fixed rear/side container slots (x, y), assigned in shuffled order.
# Chosen so up to three bins plus one basket never overlap each other.
CONTAINER_SLOTS = (
    (0.72, 0.27),
    (0.72, 0.0),
    (0.72, -0.27),
    (0.40, 0.32),
)

# Region used when rejection-sampling loose object positions.
OBJECT_X_RANGE = (0.38, 0.62)
OBJECT_Y_RANGE = (-0.28, 0.20)

MAX_PLACEMENT_ATTEMPTS = 400

CHINESE_COLOR_NAMES = {
    value: key for key, value in CHINESE_COLOR_WORDS.items()
}


class GenerationError(RuntimeError):
    """Raised when a legal layout cannot be sampled for a seed."""


class _Placement:
    """A placed footprint without an instance identity yet."""

    def __init__(self, object_type, position):
        self.object_type = object_type
        self.position = position

    @property
    def footprint(self):
        return FOOTPRINT_HALF_EXTENTS[self.object_type]


class _LayoutBuilder:
    """Accumulates placed footprints and performs collision checks."""

    def __init__(self, rng):
        self._rng = rng
        slots = list(CONTAINER_SLOTS)
        rng.shuffle(slots)
        self._free_slots = slots
        self.placements = []

    def place_container(self, object_type):
        # type: (str) -> tuple
        while self._free_slots:
            x, y = self._free_slots.pop()
            if self._is_free(object_type, (x, y)):
                position = (x, y, REST_Z[object_type])
                self.placements.append(
                    _Placement(object_type, position)
                )
                return position
        raise GenerationError('No free container slot left.')

    def sample_object(self, object_type):
        # type: (str) -> tuple
        z = REST_Z[object_type]
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            x = self._rng.uniform(*OBJECT_X_RANGE)
            y = self._rng.uniform(*OBJECT_Y_RANGE)
            if self._is_free(object_type, (x, y)):
                position = (x, y, z)
                self.placements.append(
                    _Placement(object_type, position)
                )
                return position
        raise GenerationError(
            'Could not place {} within {} attempts.'.format(
                object_type, MAX_PLACEMENT_ATTEMPTS
            )
        )

    def sample_relocation(self, object_type, old_position):
        # type: (str, tuple) -> tuple
        for _ in range(MAX_PLACEMENT_ATTEMPTS):
            x = self._rng.uniform(*OBJECT_X_RANGE)
            y = self._rng.uniform(*TABLE_Y_RANGE)
            if (
                abs(x - old_position[0]) < 0.12
                and abs(y - old_position[1]) < 0.12
            ):
                continue
            if self._is_free(object_type, (x, y)):
                return (x, y, old_position[2])
        raise GenerationError('Could not sample a relocation spot.')

    def _is_free(self, object_type, xy):
        half_x, half_y = FOOTPRINT_HALF_EXTENTS[object_type]
        x, y = xy
        for placed in self.placements:
            other_hx, other_hy = placed.footprint
            gap_x = abs(x - placed.position[0])
            gap_y = abs(y - placed.position[1])
            if (
                gap_x < half_x + other_hx + MIN_OBJECT_GAP
                and gap_y < half_y + other_hy + MIN_OBJECT_GAP
            ):
                return False
        return True


def _color_word(color):
    # type: (str) -> str
    return '{}色'.format(CHINESE_COLOR_NAMES[color])


def _instance(object_type, position, instance_id=None, color=None):
    # type: (str, tuple, str, str) -> ObjectInstance
    return ObjectInstance(
        instance_id=instance_id or object_type,
        object_type=object_type,
        color=color,
        position=tuple(position),
    )


def _number_blocks(blocks_by_color):
    # type: (dict) -> list
    """Build block instances with per-color indices sorted by position."""
    instances = []
    for color in sorted(blocks_by_color):
        ordered = sorted(
            blocks_by_color[color],
            key=lambda position: (position[0], position[1])
        )
        for index, position in enumerate(ordered, start=1):
            instances.append(
                _instance(
                    'block',
                    position,
                    instance_id=build_instance_id(
                        'block', color, index
                    ),
                    color=color,
                )
            )
    return instances


def _bin_instance(color, position):
    # type: (str, tuple) -> ObjectInstance
    return _instance(
        'bin',
        position,
        instance_id=build_instance_id('bin', color, 1),
        color=color,
    )


def _build_spec(category, seed, command, objects, goal, events=(),
                scenario_id=None):
    spec = SceneSpec(
        scenario_id=scenario_id or '{}_seed{}'.format(category, seed),
        category=category,
        seed=seed,
        command=command,
        objects=tuple(objects),
        events=tuple(events),
        goal=ScenarioGoal(category=category, params=goal),
    )
    validate(spec)
    return spec


# ---------------------------------------------------------------------------
# Category generators
# ---------------------------------------------------------------------------

def _generate_classification(rng, seed, scenario_id):
    layout = _LayoutBuilder(rng)
    blocks_by_color = {'red': []}

    red_count = rng.randint(2, 3)
    other_colors = list(
        color for color in SCENE_COLORS if color != 'red'
    )
    rng.shuffle(other_colors)
    chosen_colors = other_colors[:rng.randint(1, 2)]
    for color in chosen_colors:
        blocks_by_color[color] = []
    block_counts = {'red': red_count}
    for color in chosen_colors:
        block_counts[color] = rng.randint(1, 2)

    bins = []
    for color in sorted(blocks_by_color):
        bins.append(_bin_instance(color, layout.place_container('bin')))

    for color, count in block_counts.items():
        blocks_by_color[color].extend(
            layout.sample_object('block') for _ in range(count)
        )

    pairs = [
        {'color': color, 'bin_id': '{}_bin_1'.format(color)}
        for color in sorted(blocks_by_color)
    ]
    command_parts = [
        '把{}积木放进{}箱子'.format(
            _color_word(pair['color']),
            _color_word(pair['color'])
        )
        for pair in pairs
    ]

    return _build_spec(
        CATEGORY_CLASSIFICATION,
        seed,
        '，'.join(command_parts) + '。',
        _number_blocks(blocks_by_color) + bins,
        {'pairs': pairs},
        scenario_id=scenario_id,
    )


def _generate_target_pick(rng, seed, scenario_id):
    layout = _LayoutBuilder(rng)
    blocks_by_color = {}
    singletons = []
    bins = []

    use_apple_target = rng.random() < 0.3
    include_cup = rng.random() < 0.4
    use_bin_container = rng.random() < 0.5

    if use_apple_target:
        bin_color = rng.choice(SCENE_COLORS)
        bin_obj = _bin_instance(
            bin_color, layout.place_container('bin')
        )
        bins.append(bin_obj)

        distractor_colors = [
            color for color in SCENE_COLORS if color != bin_color
        ]
        rng.shuffle(distractor_colors)
        counts = [
            (distractor_colors[0], 2),
            (distractor_colors[1], rng.randint(1, 2)),
        ]
        for color, count in counts:
            blocks_by_color[color] = [
                layout.sample_object('block') for _ in range(count)
            ]

        apple_position = layout.sample_object('apple')
        singletons.append(_instance('apple', apple_position))
        if include_cup:
            singletons.append(
                _instance('cup', layout.sample_object('cup'))
            )

        target_id = 'apple'
        container_id = bin_obj.instance_id
        command = '把苹果放进{}箱子。'.format(
            _color_word(bin_color)
        )
    else:
        # A uniquely colored block is the target.
        target_color = rng.choice(SCENE_COLORS)
        other_colors = [
            color for color in SCENE_COLORS if color != target_color
        ]
        rng.shuffle(other_colors)

        if use_bin_container:
            bin_color = rng.choice(SCENE_COLORS)
            bin_obj = _bin_instance(
                bin_color, layout.place_container('bin')
            )
            bins.append(bin_obj)
            container_id = bin_obj.instance_id
            container_word = '{}箱子'.format(
                _color_word(bin_color)
            )
        else:
            basket = _instance(
                'basket', layout.place_container('basket')
            )
            singletons.append(basket)
            container_id = 'basket'
            container_word = '篮子'

        block_counts = {
            target_color: 1,
            other_colors[0]: rng.randint(1, 2),
            other_colors[1]: rng.randint(1, 2),
        }
        for color, count in block_counts.items():
            blocks_by_color[color] = [
                layout.sample_object('block') for _ in range(count)
            ]

        if include_cup:
            singletons.append(
                _instance('cup', layout.sample_object('cup'))
            )

        target_id = build_instance_id('block', target_color, 1)
        command = '把{}积木放进{}。'.format(
            _color_word(target_color), container_word
        )

    return _build_spec(
        CATEGORY_TARGET_PICK,
        seed,
        command,
        _number_blocks(blocks_by_color) + singletons + bins,
        {'target_id': target_id, 'container_id': container_id},
        scenario_id=scenario_id,
    )


def _generate_sequential(rng, seed, scenario_id):
    layout = _LayoutBuilder(rng)

    block_color = rng.choice(SCENE_COLORS)
    block_count = rng.randint(1, 2)
    bin_color = rng.choice(SCENE_COLORS)

    basket = _instance('basket', layout.place_container('basket'))
    bin_obj = _bin_instance(bin_color, layout.place_container('bin'))

    cup_obj = _instance('cup', layout.sample_object('cup'))
    apple_obj = _instance('apple', layout.sample_object('apple'))
    blocks_by_color = {
        block_color: [
            layout.sample_object('block') for _ in range(block_count)
        ]
    }

    command = '先把杯子放进篮子，再把苹果放进{}箱子。'.format(
        _color_word(bin_color)
    )
    goal = {
        'first': {'object_id': 'cup', 'container_id': 'basket'},
        'second': {
            'object_id': 'apple',
            'container_id': bin_obj.instance_id,
        },
    }

    return _build_spec(
        CATEGORY_SEQUENTIAL,
        seed,
        command,
        [cup_obj, apple_obj, basket, bin_obj]
        + _number_blocks(blocks_by_color),
        goal,
        scenario_id=scenario_id,
    )


def _generate_recovery(rng, seed, scenario_id):
    layout = _LayoutBuilder(rng)
    blocks_by_color = {}
    target_objects = []

    use_block_target = rng.random() < 0.4

    if use_block_target:
        target_color = rng.choice(SCENE_COLORS)
        other_colors = [
            color for color in SCENE_COLORS if color != target_color
        ]
        rng.shuffle(other_colors)
        distractor_color = other_colors[0]
    else:
        colors = list(SCENE_COLORS)
        rng.shuffle(colors)
        distractor_color = colors[0]

    bin_color = rng.choice(SCENE_COLORS)
    bin_obj = _bin_instance(bin_color, layout.place_container('bin'))

    if use_block_target:
        blocks_by_color[target_color] = [
            layout.sample_object('block')
        ]
        blocks_by_color[distractor_color] = [
            layout.sample_object('block')
        ]
        blocks = _number_blocks(blocks_by_color)
        target_id = build_instance_id('block', target_color, 1)
        target_obj = next(
            obj for obj in blocks if obj.instance_id == target_id
        )
        target_word = '{}积木'.format(_color_word(target_color))
    else:
        blocks_by_color[distractor_color] = [
            layout.sample_object('block')
        ]
        blocks = _number_blocks(blocks_by_color)
        target_obj = _instance(
            'apple', layout.sample_object('apple')
        )
        target_objects.append(target_obj)
        target_id = 'apple'
        target_word = '苹果'

    cup_obj = _instance('cup', layout.sample_object('cup'))

    new_position = layout.sample_relocation(
        target_obj.object_type, target_obj.position
    )
    event = EventScript(
        event=EVENT_OBJECT_SLIDE,
        target=target_id,
        after_action_index=1,
        new_position=new_position,
        new_support='table',
    )
    command = '把{}放进{}箱子。'.format(
        target_word, _color_word(bin_color)
    )

    return _build_spec(
        CATEGORY_RECOVERY,
        seed,
        command,
        blocks + target_objects + [cup_obj, bin_obj],
        {'target_id': target_id, 'container_id': bin_obj.instance_id},
        events=[event],
        scenario_id=scenario_id,
    )


def _generate_safe_rejection(rng, seed, subtype, scenario_id):
    layout = _LayoutBuilder(rng)
    blocks_by_color = {}
    singletons = []
    bins = []

    def add_blocks(color, count):
        blocks_by_color.setdefault(color, []).extend(
            layout.sample_object('block') for _ in range(count)
        )

    if subtype == REJECT_MISSING_TARGET:
        bins.append(
            _bin_instance('red', layout.place_container('bin'))
        )
        add_blocks('red', 1)
        add_blocks('blue', 1)
        add_blocks('green', 1)
        missing_color = 'yellow'
        command = '把{}积木放进红色箱子。'.format(
            _color_word(missing_color)
        )
        goal = {
            'subtype': subtype,
            'object_type': 'block',
            'color': missing_color,
            'container_id': 'red_bin_1',
        }
    elif subtype == REJECT_WRONG_CONTAINER:
        singletons.append(
            _instance('basket', layout.place_container('basket'))
        )
        singletons.append(
            _instance('cup', layout.sample_object('cup'))
        )
        add_blocks('red', 2)
        add_blocks('blue', 1)
        command = '把红色积木放进杯子里。'
        goal = {
            'subtype': subtype,
            'object_id': 'red_block_1',
            'container_id': 'cup',
        }
    else:
        bins.append(
            _bin_instance('red', layout.place_container('bin'))
        )
        bins.append(
            _bin_instance('blue', layout.place_container('bin'))
        )
        add_blocks('green', 2)
        command = '把红色箱子放进蓝色箱子。'
        goal = {
            'subtype': subtype,
            'object_id': 'red_bin_1',
            'container_id': 'blue_bin_1',
        }

    return _build_spec(
        CATEGORY_SAFE_REJECTION,
        seed,
        command,
        _number_blocks(blocks_by_color) + singletons + bins,
        goal,
        scenario_id=scenario_id,
    )


def generate(category, seed, scenario_id=None, **params):
    # type: (str, int, str, object) -> SceneSpec
    """
    Generate one validated :class:`SceneSpec` deterministically.

    ``seed`` must be an integer. ``scenario_id`` overrides the default
    ``<category>_seed<seed>`` id (used by the fixed suite). For
    safe_rejection pass ``subtype`` explicitly; it otherwise cycles with
    the seed.
    """
    if category not in SCENARIO_CATEGORIES:
        raise ValueError('Unknown category: {}'.format(category))
    if not isinstance(seed, int):
        raise ValueError('seed must be an integer.')

    rng = random.Random(seed)

    if category == CATEGORY_SAFE_REJECTION:
        subtype = params.get('subtype')
        if subtype is None:
            subtypes = (
                REJECT_MISSING_TARGET,
                REJECT_WRONG_CONTAINER,
                REJECT_NOT_GRASPABLE,
            )
            subtype = subtypes[seed % len(subtypes)]
        return _generate_safe_rejection(
            rng, seed, subtype, scenario_id
        )

    generators = {
        CATEGORY_CLASSIFICATION: _generate_classification,
        CATEGORY_TARGET_PICK: _generate_target_pick,
        CATEGORY_SEQUENTIAL: _generate_sequential,
        CATEGORY_RECOVERY: _generate_recovery,
    }
    return generators[category](rng, seed, scenario_id)


def generate_scenario(category, seed, scenario_id=None, **params):
    # type: (str, int, str, object) -> SceneSpec
    """Compatibility entry point for callers using the scenario name."""
    return generate(
        category,
        seed,
        scenario_id=scenario_id,
        **params
    )
