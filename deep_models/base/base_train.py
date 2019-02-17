import os
import tensorflow as tf
from tensorflow.contrib import slim
from utils.logger import Logger


class BaseTrain:
    def __init__(self, model, config):
        self.model = model
        self.config = config

    def train(self):
        # for i in range(self.config.num_epochs):
        try:
            # print("epoch {}\n".format(i))
            # self.train_dataset = self.config.DataGenerator('train', 1, self.config.batch_size,
            #                                                self.config)
            self.train_epoch()
        except tf.errors.OutOfRangeError as e:
            tf.logging.warn(e)

    def eval(self):
        # for i in range(self.config.num_epochs):
        try:
            # print("epoch {}\n".format(i))
            # self.valid_dataset = self.config.DataGenerator('valid', 1,
            #                                                self.config.batch_size * 10, self.config)
            self.eval_epoch()
        except tf.errors.OutOfRangeError as e:
            tf.logging.warn(e)

    def build_sv(self):
        if self.config.pre_train_ckpt_dir != '':
            restore_vars = self.get_restore_vars()
            restore_saver = tf.train.Saver(restore_vars)

            def restore_fn(sess):
                restore_saver.restore(sess, os.path.join(self.config.pre_train_ckpt_dir, "ckpt", "model.best"))

            sv = tf.train.Supervisor(
                logdir=self.config.checkpoint_dir,
                saver=self.model.saver,
                summary_op=None,
                summary_writer=None,
                save_model_secs=self.config.checkpoint_secs,
                global_step=self.model.global_step_tensor,
                init_fn=restore_fn)
        else:
            sv = tf.train.Supervisor(
                logdir=self.config.checkpoint_dir,
                saver=self.model.saver,
                summary_op=None,
                summary_writer=None,
                save_model_secs=self.config.checkpoint_secs,
                global_step=self.model.global_step_tensor)
        return sv

    def get_restore_vars(self):
        reader = tf.train.NewCheckpointReader(os.path.join(self.config.pre_train_ckpt_dir, "ckpt", "model.best"))
        var_to_shape_map = reader.get_variable_to_shape_map()
        exclude_scopes = ["global_step"]
        all_vars = slim.get_variables_to_restore(exclude=exclude_scopes)
        var_restored = []
        for var in all_vars:
            var_name = var.op.name
            var_shape = var.get_shape().as_list()
            if var_name in var_to_shape_map and var_to_shape_map[var_name] == var_shape:
                var_restored.append(var)

        return var_restored

    def train_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def train_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError

    def eval_epoch(self):
        """
        implement the logic of epoch:
        -loop over the number of iterations in the config and call the train step
        -add any summaries you want using the summary
        """
        raise NotImplementedError

    def eval_step(self):
        """
        implement the logic of the train step
        - run the tensorflow session
        - return any metrics you need to summarize
        """
        raise NotImplementedError
