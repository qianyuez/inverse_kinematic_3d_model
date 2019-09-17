import numpy as np

scale = 0.1
d1 = 29.3 * scale
d2 = 29.4 * scale
l1 = 46.5 * scale
l2 = 48.4 * scale

hand_target_radius = 1.0 * (d1 + d2)
foot_target_radius = 1.0 * (l1 + l2)

left_hand_rotation_range = np.array([[-0.2 * np.pi, 0.1 * np.pi],
                                    [-0.6 * np.pi, 0.1 * np.pi],
                                    [-0.5 * np.pi, 0.5 * np.pi],
                                    [-0.75 * np.pi, 0]])

right_hand_rotation_range = np.array([[-0.2 * np.pi, 0.1 * np.pi],
                                      [-0.1 * np.pi, 0.6 * np.pi],
                                      [-0.5 * np.pi, 0.5 * np.pi],
                                      [0, 0.75 * np.pi]])

left_foot_rotation_range = np.array([[-0.5 * np.pi, 0.1 * np.pi],
                                     [0, 0],
                                     [0, 0.15 * np.pi],
                                     [0, 0.65 * np.pi]])

right_foot_rotation_range = np.array([[-0.5 * np.pi, 0.1 * np.pi],
                                      [0, 0],
                                      [-0.15 * np.pi, 0],
                                      [0, 0.65 * np.pi]])