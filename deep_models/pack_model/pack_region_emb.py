#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import logging
import traceback
import importlib
from utils.dirs import create_dirs
import tensorflow as tf
from tensorflow.python.framework import graph_util


class PackModel:
    def __init__(self, config, gpu_config):
        self.config = config
        self.model = self.config.Model(self.config)
        self.sess = tf.Session(config=gpu_config)
        self.saver = tf.train.Saver()

    def _load_model(self):
        try:
            checkpoint_state = tf.train.get_checkpoint_state(self.config.checkpoint_dir)
            if not (checkpoint_state and checkpoint_state.model_checkpoint_path):
                tf.logging.info('No model to decode yet at %s', self.config.checkpoint_dir)
                return False, 'No model to decode yet at %s' % self.config.checkpoint_dir

            checkpoint_path = os.path.join(
                self.config.checkpoint_dir,
                "model.best"
                # 'model.ckpt-36986'
            )
            self.saver.restore(self.sess, checkpoint_path)
        except Exception as e:
            logging.error(traceback.format_exc())
            return False, e
        return True, "load model success!"

    def load(self):
        flag, msg = self._load_model()
        if flag is False:
            return flag, msg
        return True, "load success!"

    def _save_model_pb(self):
        create_dirs([self.config.model_output_path])

        with self.sess as sess:
            graph_def = tf.get_default_graph().as_graph_def()
            output_graph_def = \
                graph_util.convert_variables_to_constants(sess, graph_def, [
                    "fact_input",
                    "dropout",
                    "is_training_1",
                    "output/xw_plus_b",
                    "output/xw_plus_b_1",
                    "output/xw_plus_b_2",
                ])

            with tf.gfile.GFile(os.path.join(self.config.model_output_path, "{}.pb".format(self.config.exp_name)), "wb") as f:
                f.write(output_graph_def.SerializeToString())

    def _load_model_pb(self):
        with tf.Session() as sess:
            with tf.gfile.FastGFile(os.path.join(self.config.model_output_path, "{}.pb".format(self.config.exp_name)), "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def)
            return graph

    def process(self):
        if self.config.io_mode == "save":
            self.load()
            self._save_model_pb()
        elif self.config.io_mode == "load":
            self._load_model_pb()
        else:
            raise ValueError("invalid io mode, please input save or load.")
