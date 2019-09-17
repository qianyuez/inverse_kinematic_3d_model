import numpy as np
import keras.backend as K
import config


def _rotate(x, y, z, rotation, axis, sin, cos):
    if axis == 'x':
        x_ = x
        y_ = cos(rotation) * y - sin(rotation) * z
        z_ = sin(rotation) * y + cos(rotation) * z
    elif axis == 'y':
        x_ = cos(rotation) * x + sin(rotation) * z
        y_ = y
        z_ = -sin(rotation) * x + cos(rotation) * z
    elif axis == 'z':
        x_ = cos(rotation) * x - sin(rotation) * y
        y_ = sin(rotation) * x + cos(rotation) * y
        z_ = z
    else:
        return None
    return x_, y_, z_


def rotate_tensor(x, y, z, rotation, axis):
    return _rotate(x, y, z, rotation, axis, K.sin, K.cos)


def rotate_vector(x, y, z, rotation, axis):
    return _rotate(x, y, z, rotation, axis, np.sin, np.cos)


def ratio_to_rotation(rotation_ratio, rotation_range):
    return rotation_range[:, 0] + rotation_ratio * (rotation_range[:, 1] - rotation_range[:, 0])


def rotate_to_target(length1, length2, rotations, rotate, second_node_rotate_axis):
    rotation_x = rotations[:, 0]
    rotation_y = rotations[:, 1]
    rotation_z = rotations[:, 2]
    second_node_rotation = rotations[:, 3]

    x = length2 if second_node_rotate_axis == 'y' else 0
    y = length2 if second_node_rotate_axis == 'x' else 0
    z = 0

    x, y, z = rotate(x, y, z, second_node_rotation, second_node_rotate_axis)
    x += length1 if second_node_rotate_axis == 'y' else 0
    y += length1 if second_node_rotate_axis == 'x' else 0

    x, y, z = rotate(x, y, z, rotation_x, 'x')
    x, y, z = rotate(x, y, z, rotation_y, 'y')
    x, y, z = rotate(x, y, z, rotation_z, 'z')
    return x, y, z


def random_target_positions(target_type, batch_size, eps=0.1):
    rotations = []
    if target_type == 'left_hand':
        length1 = config.d1
        length2 = config.d2
        rotation_ranges = config.left_hand_rotation_range

        axis = 'y'
    elif target_type == 'right_hand':
        length1 = -config.d1
        length2 = -config.d2
        rotation_ranges = config.right_hand_rotation_range
        axis = 'y'
    elif target_type == 'left_foot':
        length1 = -config.l1
        length2 = -config.l2
        rotation_ranges = config.left_foot_rotation_range
        axis = 'x'
    elif target_type == 'right_foot':
        length1 = -config.l1
        length2 = -config.l2
        rotation_ranges = config.right_foot_rotation_range
        axis = 'x'
    else:
        return None
    for _ in range(batch_size):
        random_rotation = [np.random.uniform(rotation_range[0] - eps, rotation_range[1] + eps)
                           for rotation_range in rotation_ranges]
        rotations.append(random_rotation)
    x, y, z = rotate_to_target(length1, length2, np.array(rotations), rotate_vector, axis)
    positions = np.stack([x, y, z], axis=-1)
    return positions


def normalize_vector(vec, l):
    vec = np.array(vec) * config.scale
    magnitude = np.linalg.norm(vec)
    if magnitude > l:
        vec = vec / magnitude * l
    return vec
