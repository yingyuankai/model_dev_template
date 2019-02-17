import importlib
import tensorflow as tf
from service.law_server import *

FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_string('config', 'cfg_fasttext', 'path to configuration file')


def main(unused_args):
    config = importlib.import_module('configs.{}'.format(FLAGS.config))
    predictor = config.Predictor(config)
    obj = LawPredictServer(config.law_predict_port, predictor)
    obj.process()


if __name__ == '__main__':
    tf.app.run()
