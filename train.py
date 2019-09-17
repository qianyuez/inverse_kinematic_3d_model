import numpy as np
import config
from inverse_kinematic_model import InverseKinematicModel
from transform import random_target_positions


def train(model, epochs, batch_size, path):
    loss = []
    for e in range(epochs):
        left_hand_targets = random_target_positions('left_hand', batch_size)
        right_hand_targets = random_target_positions('right_hand', batch_size)
        left_foot_targets = random_target_positions('left_foot', batch_size)
        right_foot_targets = random_target_positions('right_foot', batch_size)
        targets = [np.array(left_hand_targets),
                   np.array(right_hand_targets),
                   np.array(left_foot_targets),
                   np.array(right_foot_targets)]
        l = model.train(targets)
        loss.append(l)
        if (e + 1) % 100 == 0:
            print('episode: {}/{}'.format(e, epochs))
            print(np.mean(loss, axis=0))
            loss = []
        if e % 1000 == 0:
            model.save_model(path)


if __name__ == '__main__':
    EPOCHS = 10000000
    BATCH_SIZE = 32
    MODEL_PATH = './model/inverse_kinematic.h5'
    model = InverseKinematicModel(config.d1,
                                  config.d2,
                                  config.l1,
                                  config.l2,
                                  config.left_hand_rotation_range,
                                  config.right_hand_rotation_range,
                                  config.left_foot_rotation_range,
                                  config.right_foot_rotation_range
                                  )
    train(model, EPOCHS, BATCH_SIZE, MODEL_PATH)
