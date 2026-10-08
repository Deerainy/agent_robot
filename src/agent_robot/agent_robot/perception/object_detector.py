"""OpenCV-only tabletop object detector MVP for the supported object set."""

import cv2
import numpy as np

SUPPORTED_OBJECTS = ('apple', 'basket', 'cup', 'table')

_APPLE_COLORS = {
    'red': ((0, 70, 40), (12, 255, 255)),
    'green': ((30, 45, 35), (95, 255, 255)),
}


def _largest_color_components(mask, minimum_area, maximum_area):
    kernel = np.ones((3, 3), dtype=np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=2)
    mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=1)
    contours, _ = cv2.findContours(
        mask,
        cv2.RETR_EXTERNAL,
        cv2.CHAIN_APPROX_SIMPLE,
    )
    components = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if not minimum_area <= area <= maximum_area:
            continue
        perimeter = cv2.arcLength(contour, True)
        if perimeter <= 0:
            continue
        circularity = 4.0 * np.pi * area / (perimeter * perimeter)
        x, y, width, height = cv2.boundingRect(contour)
        if min(width, height) < 12 or max(width, height) > 110:
            continue
        if (
            circularity < 0.42
            or max(width, height) > 2.0 * min(width, height)
        ):
            continue
        components.append({
            'bbox': [x, y, x + width, y + height],
            'area': float(area),
            'confidence': min(0.99, 0.60 + circularity * 0.35),
        })
    return sorted(components, key=lambda item: item['area'], reverse=True)


def _detect_apples(image, hsv):
    height, width = image.shape[:2]
    minimum_area = max(90.0, height * width * 0.0012)
    maximum_area = height * width * 0.08
    detections = []

    for color, (lower, upper) in _APPLE_COLORS.items():
        mask = cv2.inRange(
            hsv,
            np.array(lower, dtype=np.uint8),
            np.array(upper, dtype=np.uint8),
        )
        if color == 'red':
            second = cv2.inRange(
                hsv,
                np.array([170, 70, 40], dtype=np.uint8),
                np.array([180, 255, 255], dtype=np.uint8),
            )
            mask = cv2.bitwise_or(mask, second)
        for component in _largest_color_components(
            mask, minimum_area, maximum_area
        ):
            center_y = (component['bbox'][1] + component['bbox'][3]) / 2.0
            if center_y < height * 0.18:
                continue
            component.update({
                'name': 'apple',
                'type': 'apple',
                'color': color,
            })
            detections.append(component)

    detections.sort(key=lambda item: (item['bbox'][0], item['bbox'][1]))
    for index, item in enumerate(detections, start=1):
        item['id'] = '{}_apple_{}'.format(item['color'], index)
    return detections


def detect_generated_scene(image_path):
    """Detect the solid-color objects used by the PyBullet dataset renderer."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Could not read image: {}'.format(image_path))
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    height, width = image.shape[:2]
    min_area = max(55.0, height * width * 0.00015)
    max_small_size = max(28, int(min(height, width) * 0.14))
    detections = []
    color_ranges = {
        'red': (((0, 80, 35), (12, 255, 255)),
                ((170, 80, 35), (180, 255, 255))),
        'green': (((30, 65, 35), (95, 255, 255)),),
        'blue': (((95, 140, 35), (135, 255, 255)),),
        'yellow': (((18, 75, 35), (35, 255, 255)),),
    }

    for color, ranges in color_ranges.items():
        mask = None
        for lower, upper in ranges:
            part = cv2.inRange(
                hsv,
                np.array(lower, dtype=np.uint8),
                np.array(upper, dtype=np.uint8),
            )
            mask = part if mask is None else cv2.bitwise_or(mask, part)
        mask = cv2.morphologyEx(
            mask, cv2.MORPH_CLOSE, np.ones((3, 3), dtype=np.uint8)
        )
        contours, _ = cv2.findContours(
            mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        for contour in contours:
            area = cv2.contourArea(contour)
            if area < min_area or area > height * width * 0.2:
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            size = max(box_width, box_height)
            if min(box_width, box_height) <= 0:
                continue
            perimeter = cv2.arcLength(contour, True)
            circularity = (
                4.0 * np.pi * area / (perimeter * perimeter)
                if perimeter else 0.0
            )
            if size <= max_small_size:
                if circularity < 0.52:
                    continue
                object_type = 'cup' if color == 'blue' else (
                    'apple' if color in ('red', 'green') else None
                )
                if object_type is None:
                    continue
            else:
                object_type = 'bin'
            detections.append({
                'type': object_type,
                'name': object_type,
                'color': color,
                'bbox': [x, y, x + box_width, y + box_height],
                'confidence': min(0.99, 0.60 + circularity * 0.35),
            })

    brown_mask = cv2.inRange(
        hsv,
        np.array([4, 80, 25], dtype=np.uint8),
        np.array([24, 255, 180], dtype=np.uint8),
    )
    brown_mask = cv2.morphologyEx(
        brown_mask, cv2.MORPH_CLOSE, np.ones((5, 5), dtype=np.uint8)
    )
    contours, _ = cv2.findContours(
        brown_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < max(500.0, height * width * 0.001):
            continue
        x, y, box_width, box_height = cv2.boundingRect(contour)
        if max(box_width, box_height) <= max_small_size:
            continue
        perimeter = cv2.arcLength(contour, True)
        circularity = (
            4.0 * np.pi * area / (perimeter * perimeter)
            if perimeter else 0.0
        )
        detections.append({
            'type': 'basket',
            'name': 'basket',
            'color': 'brown',
            'bbox': [x, y, x + box_width, y + box_height],
            'confidence': min(0.95, 0.60 + circularity * 0.3),
        })

    basket_boxes = [
        item['bbox'] for item in detections if item['type'] == 'basket'
    ]
    if basket_boxes:
        detections = [
            item for item in detections
            if item['type'] != 'bin'
            or not any(
                min(item['bbox'][2], basket[2])
                > max(item['bbox'][0], basket[0])
                and min(item['bbox'][3], basket[3])
                > max(item['bbox'][1], basket[1])
                for basket in basket_boxes
            )
        ]

    detections.sort(
        key=lambda item: (
            item['type'], item['color'], item['bbox'][0], item['bbox'][1]
        )
    )
    for item in detections:
        kind = item['type']
        if kind == 'apple':
            color_index = sum(
                1 for prior in detections
                if prior is not item
                and prior['type'] == kind
                and prior['color'] == item['color']
                and (prior['bbox'][0], prior['bbox'][1])
                < (item['bbox'][0], item['bbox'][1])
            ) + 1
            item['id'] = '{}_{}_{}'.format(
                item['color'], kind, color_index
            )
        elif kind == 'bin':
            color_index = sum(
                1 for prior in detections
                if prior is not item
                and prior['type'] == kind
                and prior['color'] == item['color']
                and (prior['bbox'][0], prior['bbox'][1])
                < (item['bbox'][0], item['bbox'][1])
            ) + 1
            item['id'] = '{}_bin_{}'.format(item['color'], color_index)
        elif kind == 'cup':
            item['id'] = 'cup'
        else:
            item['id'] = 'basket'
    return detections


def _window_candidates(image, hsv, width_fraction, height_fraction):
    height, width = image.shape[:2]
    window_width = max(20, int(width * width_fraction))
    window_height = max(20, int(height * height_fraction))
    step = max(4, min(width, height) // 45)
    edges = cv2.Canny(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY), 55, 135)

    hue = hsv[:, :, 0]
    saturation = hsv[:, :, 1]
    value = hsv[:, :, 2]
    brown = (
        (hue >= 4) & (hue <= 32)
        & (saturation >= 38) & (value >= 35) & (value <= 245)
    ).astype(np.uint8)
    edge_integral = cv2.integral(
        (edges > 0).astype(np.uint8), sdepth=cv2.CV_64F
    )
    brown_integral = cv2.integral(brown, sdepth=cv2.CV_64F)

    best = None
    for y in range(int(height * 0.12), height - window_height, step):
        for x in range(0, width - window_width, step):
            x2 = x + window_width
            y2 = y + window_height
            area = float(window_width * window_height)
            edge_count = (
                edge_integral[y2, x2] - edge_integral[y, x2]
                - edge_integral[y2, x] + edge_integral[y, x]
            )
            brown_count = (
                brown_integral[y2, x2] - brown_integral[y, x2]
                - brown_integral[y2, x] + brown_integral[y, x]
            )
            brown_fraction = brown_count / area
            edge_fraction = edge_count / area
            if brown_fraction < 0.24 or edge_fraction < 0.055:
                continue
            score = brown_fraction * min(edge_fraction, 0.35)
            if best is None or score > best[0]:
                best = (score, [x, y, x2, y2], brown_fraction, edge_fraction)
    return best


def _detect_basket(image, hsv):
    candidates = [
        _window_candidates(image, hsv, width_fraction, height_fraction)
        for width_fraction, height_fraction in (
            (0.32, 0.45),
            (0.36, 0.48),
            (0.40, 0.48),
            (0.34, 0.40),
        )
    ]
    candidates = [candidate for candidate in candidates if candidate]
    if not candidates:
        return None
    score, bbox, brown_fraction, edge_fraction = max(
        candidates, key=lambda item: item[0]
    )
    confidence = min(0.90, 0.45 + score * 1.8)
    return {
        'id': 'basket',
        'name': 'basket',
        'type': 'basket',
        'color': 'brown',
        'bbox': bbox,
        'confidence': confidence,
    }


def _detect_cup(image, hsv, exclude_boxes, basket_bbox=None):
    height, width = image.shape[:2]
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 45, 125)
    neutral_bright = (
        (hsv[:, :, 1] < 160) & (hsv[:, :, 2] > 115)
    ).astype(np.uint8)
    window_width = max(20, int(width * 0.20))
    window_height = max(24, int(height * 0.34))
    step = max(3, min(width, height) // 60)
    best = None

    for y in range(int(height * 0.10), height - window_height, step):
        for x in range(0, width - window_width, step):
            x2, y2 = x + window_width, y + window_height
            window_area = float(window_width * window_height)
            if any(
                max(0, min(x2, box[2]) - max(x, box[0]))
                * max(0, min(y2, box[3]) - max(y, box[1]))
                / window_area > overlap_threshold
                for box, overlap_threshold in exclude_boxes
            ):
                continue
            if (x + x2) / 2.0 < width * 0.17:
                continue
            center_x = (x + x2) / 2.0
            if (
                basket_bbox is not None
                and basket_bbox[0] - width * 0.06
                <= center_x
                <= basket_bbox[2] + width * 0.06
            ):
                apple_centers = [
                    (
                        (box[0] + box[2]) / 2.0,
                        (box[1] + box[3]) / 2.0,
                    )
                    for box, _ in exclude_boxes
                ]
                if apple_centers and min(
                    abs(center_x - apple_center[0])
                    for apple_center in apple_centers
                ) > width * 0.30:
                    continue
            neutral_fraction = float(
                np.mean(neutral_bright[y:y2, x:x2])
            )
            edge_fraction = float(np.mean(edges[y:y2, x:x2] > 0))
            if neutral_fraction < 0.38 or edge_fraction < 0.045:
                continue
            score = neutral_fraction * min(edge_fraction, 0.25)
            if best is None or score > best[0]:
                best = (score, [x, y, x2, y2], neutral_fraction)

    if best is None:
        return None
    score, bbox, neutral_fraction = best
    return {
        'id': 'cup',
        'name': 'cup',
        'type': 'cup',
        'color': 'white',
        'bbox': bbox,
        'confidence': min(0.82, 0.40 + score * 2.2),
    }


def detect_objects(image):
    """Detect supported objects in BGR; boxes use [x1, y1, x2, y2]."""
    if image is None or not isinstance(image, np.ndarray):
        raise ValueError('Expected an image as a NumPy array.')
    if image.ndim != 3 or image.shape[2] != 3:
        raise ValueError('Expected a three-channel BGR image.')
    height, width = image.shape[:2]
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    apples = _detect_apples(image, hsv)
    detections = list(apples)

    basket = _detect_basket(image, hsv)
    exclude = [(item['bbox'], 0.05) for item in apples]
    if basket is not None:
        detections.append(basket)

    cup = _detect_cup(
        image,
        hsv,
        exclude,
        basket['bbox'] if basket is not None else None,
    )
    if cup is not None:
        detections.append(cup)

    detections.append({
        'id': 'table',
        'name': 'table',
        'type': 'table',
        'color': None,
        'bbox': [0, 0, width, height],
        'confidence': 0.50,
    })
    return detections


def detect_image(image_path):
    """Read and detect supported objects from an image file."""
    image = cv2.imread(image_path, cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError('Could not read image: {}'.format(image_path))
    return detect_objects(image)
