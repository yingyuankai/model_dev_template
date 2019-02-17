import tensorflow as tf
from tensorflow.contrib import slim
from deep_models.base.base_feature import BaseFeature


class Feature(BaseFeature):
    def __init__(self, config):
        super(Feature, self).__init__(config)

    def feature(self, inputs, is_training, task=""):
        """
        textcnn feature
        :param inputs: sentence embedding
        :param is_training: True
        :return:
        """
        with tf.variable_scope("textcnn_feature{}".format("_" + task if task != "" else "")):
            net_outputs = []
            total_output = self._config.output_num_per_filter * len(self._config.filter_sizes)
            for i, filter_size in enumerate(self._config.filter_sizes):
                with tf.name_scope("conv_pool_{}".format(filter_size)):
                    with slim.arg_scope([slim.conv2d], padding='VALID', normalizer_fn=slim.batch_norm,
                                        normalizer_params={'is_training': is_training}):
                        net = slim.conv2d(inputs,
                                          self._config.output_num_per_filter,
                                          [filter_size, self._config.embed_size],
                                          scope='conv_0_{}'.format(filter_size))
                        net = slim.conv2d(net,
                                          self._config.output_num_per_filter, [filter_size, 1],
                                          scope='conv_1_{}'.format(filter_size))
                    net = tf.reduce_max(net, axis=1, name='max_pool')
                    net_outputs.append(net)

            net_c = tf.concat(net_outputs, -1)
            net_flatted = tf.reshape(net_c, [-1, total_output])
            return net_flatted