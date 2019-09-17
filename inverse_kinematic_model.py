from keras import Model
from keras.layers import Input, Dense
from keras.optimizers import Adam
from transform import rotate_to_target, ratio_to_rotation, rotate_tensor
import keras.backend as K


class InverseKinematicModel():
    def __init__(self,
                 d1,
                 d2,
                 l1,
                 l2,
                 left_hand_rotation_range,
                 right_hand_rotation_range,
                 left_foot_rotation_range,
                 right_foot_rotation_range):
        self.d1 = d1
        self.d2 = d2
        self.l1 = l1
        self.l2 = l2
        self.left_hand_rotation_range = left_hand_rotation_range
        self.right_hand_rotation_range = right_hand_rotation_range
        self.left_foot_rotation_range = left_foot_rotation_range
        self.right_foot_rotation_range = right_foot_rotation_range
        self.model = self._build_model()
        self.model.compile(optimizer=Adam(lr=0.00001), loss=[
            self.get_custom_mse(d1, d2, left_hand_rotation_range, 'y'),
            self.get_custom_mse(-d1, -d2, right_hand_rotation_range, 'y'),
            self.get_custom_mse(-l1, -l2, left_foot_rotation_range, 'x'),
            self.get_custom_mse(-l1, -l2, right_foot_rotation_range, 'x')
        ])

    def predict_rotations(self, positions):
        rotations = self.model.predict(positions)
        rotations = [r[0] for r in rotations]
        left_arm_rotation, right_arm_rotation, left_leg_rotation, right_leg_rotation = rotations

        left_arm_rotation = ratio_to_rotation(left_arm_rotation, self.left_hand_rotation_range)
        right_arm_rotation = ratio_to_rotation(right_arm_rotation, self.right_hand_rotation_range)
        left_leg_rotation = ratio_to_rotation(left_leg_rotation, self.left_foot_rotation_range)
        right_leg_rotation = ratio_to_rotation(right_leg_rotation, self.right_foot_rotation_range)

        return left_arm_rotation, right_arm_rotation, left_leg_rotation, right_leg_rotation

    def train(self, positions):
        loss = self.model.train_on_batch(positions, positions)
        return loss

    def save_model(self, path):
        self.model.save_weights(path)

    def load_model(self, path):
        self.model.load_weights(path)

    def _build_model(self):
        left_arm_target = Input(shape=(3,))
        right_arm_target = Input(shape=(3,))
        left_leg_target = Input(shape=(3,))
        right_leg_target = Input(shape=(3,))

        units = 150
        activation = 'tanh'
        d1 = Dense(units, activation=activation)(left_arm_target)
        d2 = Dense(units, activation=activation)(right_arm_target)
        d3 = Dense(units, activation=activation)(left_leg_target)
        d4 = Dense(units, activation=activation)(right_leg_target)

        left_arm_rotation = Dense(4, activation='sigmoid')(d1)
        right_arm_rotation = Dense(4, activation='sigmoid')(d2)
        left_leg_rotation = Dense(4, activation='sigmoid')(d3)
        right_leg_rotation = Dense(4, activation='sigmoid')(d4)

        model = Model(inputs=[left_arm_target, right_arm_target, left_leg_target, right_leg_target],
                      outputs=[left_arm_rotation, right_arm_rotation, left_leg_rotation, right_leg_rotation])
        return model

    def get_custom_mse(self, length1, length2, rotation_ranges, axis):
        def custom_func(y_true, y_pred):
            rotations = ratio_to_rotation(y_pred, rotation_ranges)
            x, y, z = rotate_to_target(length1, length2, rotations, rotate_tensor, axis)
            pos = K.stack([x, y, z], axis=-1)
            return K.mean(K.square(pos - y_true))
        return custom_func
