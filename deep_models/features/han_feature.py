import tensorflow as tf
from tensorflow.contrib import slim
from deep_models.base.base_feature import BaseFeature
from utils.nn_utils import bi_lstm, attention_multihop


class Feature(BaseFeature):
    def __init__(self, config):
        super(Feature, self).__init__(config)

    def feature(self, inputs, keep_prob, is_training=True):
        """
        han feature
        :param inputs: sentence embedding
        :param is_training: True
        :return:
        """
        with tf.variable_scope("han_feature"):
            embedded_sent_words = [tf.squeeze(x) for x in tf.split(inputs, self._config.num_sens, axis=1)]
            word_atts = []
            # word level
            for i in range(self._config.num_sens):
                sent = embedded_sent_words[i]
                reuse_flag = True if i > 0 else False
                sent = tf.reshape(sent, [-1, self._config.sent_len, self._config.embed_size])
                words_encoded, word_encoded2 = \
                    bi_lstm(sent, 'word_level', self._config.hidden_size, keep_prob, reuse_flag)
                # word attention
                word_att = attention_multihop(words_encoded, "word_level", self._config.num_hops, reuse=reuse_flag)
                word_att = slim.batch_norm(word_att, scale=True, epsilon=1e-5, is_training=is_training)
                word_att = slim.dropout(word_att, keep_prob=keep_prob)
                # word_attention=tf.concat([word_attention,word_encodeded2],axis=1) # TODO
                word_atts.append(word_att)

            # sent level
            sent_encoder_input = tf.stack(word_atts, axis=1)
            sent_encoded, sent_encoded2 = bi_lstm(sent_encoder_input, "sent_level", self._config.hidden_size * 2, keep_prob)
            sent_att = attention_multihop(sent_encoded, "sent_level", self._config.num_hops)
            sent_att = slim.batch_norm(sent_att, scale=True, epsilon=1e-5, is_training=is_training)
            sent_att = slim.dropout(sent_att, keep_prob=keep_prob)
            net_flatted = tf.reshape(sent_att, shape=[-1, self._config.hidden_size * 4])
            return net_flatted
