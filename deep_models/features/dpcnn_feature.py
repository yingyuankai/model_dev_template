import tensorflow as tf
from tensorflow.contrib import slim
from deep_models.base.base_feature import BaseFeature
from utils.nn_utils import region_embedding, conv_block


class Feature(BaseFeature):
    def __init__(self, config):
        super(Feature, self).__init__(config)

    def feature(self, inputs, is_training=True):
        """
        dpcnn feature
        :param inputs: sentence embedding
        :param is_training: True
        :return:
        """
        with tf.variable_scope("dpcnn_feature"):
            # region embedding
            inputs = region_embedding(inputs,
                                      self._config.channel_size,
                                      self._config.filter_size,
                                      self._config.embed_size,
                                      is_training)
            # first two layers of conv
            net = conv_block(inputs,
                             name="conv_block_0",
                             channel_size=self._config.channel_size,
                             max_pooling=False, is_training=is_training)
            # repeat conv_block n times
            for i in range(self._config.num_repeats):
                net = conv_block(net,
                                 name="conv_block_{}".format(i + 1),
                                 channel_size=self._config.channel_size,
                                 max_pooling=False, is_training=is_training)
            # max pooling
            with tf.variable_scope("max_pool_final"):
                with slim.arg_scope([slim.max_pool2d], kernel_size=[net.get_shape().as_list()[1], 1], stride=1,
                                    padding='VALID'):
                    net = slim.max_pool2d(net)
            net_flatted = tf.reshape(net, [-1, self._config.channel_size])
            return net_flatted
