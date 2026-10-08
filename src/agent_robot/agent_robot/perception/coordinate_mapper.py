"""Configurable affine mapping from tabletop pixels to robot coordinates."""

import math
from dataclasses import dataclass


OBJECT_REST_HEIGHT = {
    'apple': 0.06,
    'cup': 0.06,
    'basket': 0.05,
    'block': 0.025,
    'bin': 0.05,
}


@dataclass(frozen=True)
class CoordinateMapper:
    """Map a calibrated image ROI onto an axis-aligned tabletop rectangle."""

    world_x: tuple = (0.38, 0.76)
    world_y: tuple = (-0.28, 0.28)
    image_roi: tuple = (0.0, 0.0, 1.0, 1.0)
    image_y_positive_toward_camera: bool = True

    def pixel_to_world(self, pixel, image_size, object_type=None):
        if (
            not isinstance(pixel, (list, tuple))
            or not isinstance(image_size, (list, tuple))
            or len(pixel) != 2
            or len(image_size) != 2
        ):
            raise ValueError('Pixel and image_size must each have two values.')
        image_width, image_height = image_size
        if (
            not math.isfinite(float(image_width))
            or not math.isfinite(float(image_height))
            or image_width <= 0
            or image_height <= 0
        ):
            raise ValueError('Image dimensions must be positive.')
        try:
            u, v = float(pixel[0]), float(pixel[1])
        except (TypeError, ValueError):
            raise ValueError('Pixel coordinates must be numeric.')
        if not math.isfinite(u) or not math.isfinite(v):
            raise ValueError('Pixel coordinates must be finite.')
        left, top, right, bottom = self.image_roi
        coordinates = (
            left,
            top,
            right,
            bottom,
            self.world_x[0],
            self.world_x[1],
            self.world_y[0],
            self.world_y[1],
        )
        if not all(math.isfinite(float(value)) for value in coordinates):
            raise ValueError('Coordinate mapping ranges must be finite.')
        if (
            not 0.0 <= left < right <= 1.0
            or not 0.0 <= top < bottom <= 1.0
            or self.world_x[1] <= self.world_x[0]
            or self.world_y[1] <= self.world_y[0]
        ):
            raise ValueError(
                'image_roi and world ranges must have positive extents.'
            )
        x0, y0 = left * image_width, top * image_height
        x1, y1 = right * image_width, bottom * image_height
        if x1 <= x0 or y1 <= y0:
            raise ValueError('image_roi must have positive width and height.')
        u_fraction = min(1.0, max(0.0, (u - x0) / (x1 - x0)))
        v_fraction = min(1.0, max(0.0, (v - y0) / (y1 - y0)))
        x = self.world_x[0] + u_fraction * (
            self.world_x[1] - self.world_x[0]
        )
        if self.image_y_positive_toward_camera:
            v_fraction = 1.0 - v_fraction
        y = self.world_y[0] + v_fraction * (
            self.world_y[1] - self.world_y[0]
        )
        z = OBJECT_REST_HEIGHT.get(object_type, 0.0)
        return [x, y, z]

    def map_bbox(self, bbox, image_size, object_type=None):
        bbox = clamp_bbox(bbox, image_size)
        center = (
            (float(bbox[0]) + float(bbox[2])) / 2.0,
            (float(bbox[1]) + float(bbox[3])) / 2.0,
        )
        return self.pixel_to_world(center, image_size, object_type)


def clamp_bbox(bbox, image_size):
    """Validate and clamp an ``[x1,y1,x2,y2]`` box to image bounds."""
    if (
        not isinstance(bbox, (list, tuple))
        or len(bbox) != 4
        or not isinstance(image_size, (list, tuple))
        or len(image_size) != 2
    ):
        raise ValueError('bbox and image_size must have lengths 4 and 2.')
    width, height = image_size
    if (
        not math.isfinite(float(width))
        or not math.isfinite(float(height))
        or width <= 0
        or height <= 0
    ):
        raise ValueError('Image dimensions must be finite and positive.')
    try:
        coordinates = [float(value) for value in bbox]
    except (TypeError, ValueError):
        raise ValueError('Bounding-box coordinates must be numeric.')
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError('Bounding-box coordinates must be finite.')
    x1, y1, x2, y2 = coordinates
    if x2 <= x1 or y2 <= y1:
        raise ValueError('Bounding box must have positive width and height.')
    clamped = [
        max(0.0, min(float(width), x1)),
        max(0.0, min(float(height), y1)),
        max(0.0, min(float(width), x2)),
        max(0.0, min(float(height), y2)),
    ]
    if clamped[2] <= clamped[0] or clamped[3] <= clamped[1]:
        raise ValueError('Bounding box does not intersect the image.')
    return clamped


def validate_object_distances(objects, allowed_containment=(), gap=0.005):
    """Return pairs whose simulation footprints are too close.

    Circular enclosing radii from each simulator footprint are deliberately
    conservative for rectangular and compound shapes.
    """
    allowed = {
        tuple(sorted(pair)) for pair in allowed_containment
    }
    conflicts = []
    for index, left in enumerate(objects):
        left_id = _object_id(left)
        left_xy = _object_position(left)
        left_radius = _object_radius(left)
        for right in objects[index + 1:]:
            right_id = _object_id(right)
            if tuple(sorted((left_id, right_id))) in allowed:
                continue
            right_xy = _object_position(right)
            distance = math.hypot(
                left_xy[0] - right_xy[0],
                left_xy[1] - right_xy[1],
            )
            minimum = left_radius + _object_radius(right) + gap
            if distance < minimum:
                conflicts.append({
                    'objects': [left_id, right_id],
                    'distance': distance,
                    'minimum_distance': minimum,
                })
    return conflicts


def correct_simulation_placements(objects, allowed_containment=()):
    """Return simulation-only positions adjusted to avoid initial overlap."""
    from agent_robot.scenarios.spec import (
        FOOTPRINT_HALF_EXTENTS,
    )

    adjusted = [dict(item) for item in objects]
    by_id = {_object_id(item): item for item in adjusted}
    containment = [tuple(pair) for pair in allowed_containment]
    for item in adjusted:
        if item.get('type') in ('basket', 'bin'):
            container_box = item.get('bbox')
            if not container_box:
                continue
            for other in adjusted:
                if other is item or other.get('type') in (
                    'basket', 'bin', 'table'
                ):
                    continue
                other_box = other.get('bbox')
                if not other_box:
                    continue
                center_x = (other_box[0] + other_box[2]) / 2.0
                center_y = (other_box[1] + other_box[3]) / 2.0
                if (
                    container_box[0] < center_x < container_box[2]
                    and container_box[1] < center_y < container_box[3]
                ):
                    containment.append(
                        (_object_id(other), _object_id(item))
                    )

    intended = {
        tuple(sorted(pair)) for pair in containment
    }
    radius_by_id = {
        _object_id(item): _object_radius(item)
        for item in adjusted
        if item.get('type') in FOOTPRINT_HALF_EXTENTS
    }
    original_positions = {
        _object_id(item): list(item['position'])
        for item in adjusted
    }
    initial_conflicts = validate_object_distances(
        [
            item for item in adjusted
            if item.get('type') in radius_by_id
        ],
        containment,
    )
    ordered = sorted(
        (
            item for item in adjusted
            if item.get('type') in FOOTPRINT_HALF_EXTENTS
        ),
        key=lambda item: (
            item.get('type') not in ('basket', 'bin'),
            -radius_by_id[_object_id(item)],
            _object_id(item),
        ),
    )
    placed = []
    adjustments = []
    # The fixed PyBullet table is centered at x=0.5 with a 1.0 x 1.0
    # tabletop. Keep each conservative circular footprint on that surface.
    x_min, x_max = 0.0, 1.0
    y_min, y_max = -0.5, 0.5

    for item in ordered:
        item_id = _object_id(item)
        object_type = item['type']
        x_radius = _object_radius(item)
        intended_container = next(
            (
                container_id
                for object_id, container_id in containment
                if object_id == item_id
            ),
            None,
        )
        source = list(original_positions[item_id])
        source[2] = _simulation_rest_height(object_type)
        if intended_container is not None:
            container = by_id[intended_container]
            container_x, container_y = container['position'][:2]
            half_x, half_y = FOOTPRINT_HALF_EXTENTS[
                container['type']
            ]
            item_half_x, item_half_y = FOOTPRINT_HALF_EXTENTS[object_type]
            wall_clearance = 0.012
            x = min(
                container_x + half_x - wall_clearance - item_half_x,
                max(
                    container_x - half_x + wall_clearance + item_half_x,
                    source[0],
                ),
            )
            y = min(
                container_y + half_y - wall_clearance - item_half_y,
                max(
                    container_y - half_y + wall_clearance + item_half_y,
                    source[1],
                ),
            )
            source = [x, y, _contained_rest_height(object_type)]

        chosen = _nearest_valid_position(
            source,
            item_id,
            object_type,
            x_radius,
            placed,
            intended,
            x_min,
            x_max,
            y_min,
            y_max,
        )
        if chosen is None:
            raise ValueError(
                'Unable to place {} without initial simulation collision.'
                .format(item_id)
            )
        item['position'] = [chosen[0], chosen[1], source[2]]
        if (
            abs(chosen[0] - original_positions[item_id][0]) > 0.001
            or abs(chosen[1] - original_positions[item_id][1]) > 0.001
            or abs(source[2] - original_positions[item_id][2]) > 0.001
        ):
            adjustments.append({
                'object': item_id,
                'perception_position': original_positions[item_id],
                'simulation_position': item['position'],
            })
        placed.append(item)

    warnings = []
    if initial_conflicts:
        warnings.append('initial placement collision detected')
    conflicts = validate_object_distances(
        [item for item in adjusted if item.get('type') in radius_by_id],
        containment,
    )
    if conflicts:
        raise ValueError(
            'Initial placement correction left collisions: {}'.format(
                conflicts
            )
        )
    return {
        'objects': adjusted,
        'adjustments': adjustments,
        'warnings': warnings,
        'containment': [list(pair) for pair in containment],
    }


def _object_id(item):
    return getattr(item, 'instance_id', None) or item.get(
        'id', item.get('name', '')
    )


def _object_position(item):
    position = (
        getattr(item, 'position') if not isinstance(item, dict)
        else item.get('position')
    )
    if (
        not isinstance(position, (list, tuple))
        or len(position) != 3
    ):
        raise ValueError(
            'Object {} must have a three-dimensional position.'.format(
                _object_id(item)
            )
        )
    try:
        coordinates = [float(value) for value in position]
    except (TypeError, ValueError):
        raise ValueError(
            'Object {} position must be numeric.'.format(_object_id(item))
        )
    if not all(math.isfinite(value) for value in coordinates):
        raise ValueError(
            'Object {} position must be finite.'.format(_object_id(item))
        )
    return coordinates


def _object_radius(item):
    from agent_robot.scenarios.spec import FOOTPRINT_HALF_EXTENTS

    object_type = (
        item.object_type if not isinstance(item, dict) else item['type']
    )
    half_x, half_y = FOOTPRINT_HALF_EXTENTS[object_type]
    return math.hypot(half_x, half_y)


def _contained_rest_height(object_type):
    heights = {
        'apple': 0.075,
        'cup': 0.08,
        'block': 0.055,
    }
    return heights[object_type]


def _simulation_rest_height(object_type):
    heights = {
        'apple': 0.045,
        'cup': 0.05,
        'basket': 0.05,
        'block': 0.025,
        'bin': 0.05,
    }
    return heights[object_type]


def _nearest_valid_position(
    source,
    object_id,
    object_type,
    radius,
    placed,
    allowed_containment,
    x_min,
    x_max,
    y_min,
    y_max,
):
    allowed = {
        tuple(sorted(pair)) for pair in allowed_containment
    }

    def required_distance(other):
        clearance = (
            0.07
            if object_type in ('apple', 'cup', 'block')
            and other.get('type') in ('basket', 'bin')
            else 0.0
        )
        return radius + _object_radius(other) + 0.005 + clearance

    if (
        x_min + radius <= source[0] <= x_max - radius
        and y_min + radius <= source[1] <= y_max - radius
        and all(
            tuple(sorted((object_id, _object_id(other)))) in allowed
            or math.hypot(
                source[0] - other['position'][0],
                source[1] - other['position'][1],
            ) >= required_distance(other)
            for other in placed
        )
    ):
        return [source[0], source[1], source[2]]

    best = None
    for step_x in range(0, 101):
        for sign_x in (1, -1) if step_x else (1,):
            x = round(source[0] + sign_x * step_x * 0.01, 6)
            if x < x_min + radius or x > x_max - radius:
                continue
            for step_y in range(0, 81):
                for sign_y in (1, -1) if step_y else (1,):
                    y = round(source[1] + sign_y * step_y * 0.01, 6)
                    if y < y_min + radius or y > y_max - radius:
                        continue
                    candidate = [x, y, source[2]]
                    collides = False
                    for other in placed:
                        other_id = _object_id(other)
                        if tuple(sorted((object_id, other_id))) in allowed:
                            continue
                        minimum = required_distance(other)
                        if math.hypot(
                            x - other['position'][0],
                            y - other['position'][1],
                        ) < minimum:
                            collides = True
                            break
                    if collides:
                        continue
                    displacement = (
                        (x - source[0]) ** 2 + (y - source[1]) ** 2
                    )
                    key = (displacement, x, y)
                    if best is None or key < best[0]:
                        best = (key, candidate)
        if best is not None and step_x * 0.01 > math.sqrt(best[0][0]):
            break
    return best[1] if best is not None else None
