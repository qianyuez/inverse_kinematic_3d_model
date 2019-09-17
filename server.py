import tornado.web
import tornado.ioloop
import tornado.escape
import config
import json
import argparse
from inverse_kinematic_model import InverseKinematicModel
from transform import normalize_vector


class IndexHandler(tornado.web.RequestHandler):
    def get(self):
        self.render('./web/index.html', port=ARGS.port)

class UpdateHandler(tornado.web.RequestHandler):
    # def set_default_headers(self):
    #     self.set_header("Access-Control-Allow-Origin", "*")
    #     self.set_header("Access-Control-Allow-Headers", "x-requested-with")
    #     self.set_header('Access-Control-Allow-Methods', 'POST, GET, OPTIONS')
    #     # self.set_header("Access-Control-Allow-Headers", "Access-Control-Allow-Headers,"
    #     #                                                " Origin,Accept, X-Requested-With, Content-Type, "
    #     #                                                "Access-Control-Request-Method, Access-Control-Request-Headers")

    def post(self):
        data = tornado.escape.json_decode(self.request.body)
        left_hand_pos = data['leftHandPos']
        right_hand_pos = data['rightHandPos']
        left_foot_pos = data['leftFootPos']
        right_foot_pos = data['rightFootPos']
        positions = [normalize_vector([left_hand_pos], config.hand_target_radius),
                     normalize_vector([right_hand_pos], config.hand_target_radius),
                     normalize_vector([left_foot_pos], config.foot_target_radius),
                     normalize_vector([right_foot_pos], config.foot_target_radius)]
        left_hand_rotation, right_hand_rotation, left_foot_rotation, right_foot_rotation = model.predict_rotations(positions)
        rotations = {
            'leftHandRotation': left_hand_rotation.tolist(),
            'rightHandRotation': right_hand_rotation.tolist(),
            'leftFootRotation': left_foot_rotation.tolist(),
            'rightFootRotation': right_foot_rotation.tolist()
        }
        self.write(json.dumps(rotations))

    # def options(self):
    #     self.set_status(204)
    #     self.finish()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--port', type=int, default=8000,
        help='local server port, default 8000'
    )
    ARGS = parser.parse_args()

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
    model.load_model(MODEL_PATH)
    settings = {
        "static_path": "./web"
    }
    application = tornado.web.Application([
        (r"/", IndexHandler),
        (r"/update", UpdateHandler)
    ], **settings)
    application.listen(ARGS.port)
    print('Inverse kinematic model application at http://localhost:{}'.format(ARGS.port))
    tornado.ioloop.IOLoop.current().start()
