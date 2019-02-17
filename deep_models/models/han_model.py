import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib import rnn
from deep_models.base.base_model import BaseModel
from utils.nn_utils import *



class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.initializer = tf.truncated_normal_initializer(stddev=1e-4)
        self.build_model()
        self.init_saver()

    def _add_placeholders(self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.facts = tf.placeholder(tf.int32, shape=[None, self.config.max_seq_len], name="fact_input")
        self.laws = tf.placeholder(tf.int32, shape=[None, None], name="law_input")
        self.accusations = tf.placeholder(tf.int32, shape=[None, None], name='accu_input')
        self.imprisonments = tf.placeholder(tf.int32, shape=[None], name='impris_input')
        self.embedding_placeholder = tf.placeholder(tf.float32,
                                                    [self.config.data_obj.vocab.num_ids(), self.config.embed_size])
        self.keep_prob = tf.placeholder(tf.float32, name='keep_prob')
        self.seq_len = self.facts.shape[-1]

    def build_model(self):
        self._add_placeholders()
        self.seq_len = self.facts.shape[-1]
        # embedding
        with tf.variable_scope('embedding'), tf.device('/cpu:0'):
            # self.embedding = tf.get_variable('words',
            #                                  shape=[self.config.data_obj.vocab.num_ids(), self.config.embed_size],
            #                                  initializer=self.initializer)
            self.embedding = tf.Variable(
                tf.constant(0., shape=[self.config.data_obj.vocab.num_ids(), self.config.embed_size]),
                trainable=False, name='chars')
            self.embedding_init = self.assign_pretrain_embedding(self.embedding_placeholder)
        # network
        self.logits_law, self.logits_accu, self.logits_impris = self.inference()

        # loss
        self.loss_l2 = l2_loss(self.config.l2_lambda)

        loss_law = loss_multi_label(self.logits_law, self.laws)
        self.loss_law = loss_law + self.loss_l2

        loss_accu = loss_multi_label(self.logits_accu, self.accusations)
        self.loss_accu = loss_accu + self.loss_l2

        loss_impris = loss_single_label(self.logits_impris, self.imprisonments)
        self.loss_impris = loss_impris + self.loss_l2

        self.loss_all = self.config.task_loss_weights[0] * loss_law + \
                        self.config.task_loss_weights[1] * loss_accu + \
                        self.config.task_loss_weights[2] * loss_impris + self.loss_l2

        # accuracy
        with tf.variable_scope("score"):
            # self.accuracy_law = cal_multi_label_accuracy(self.logits_law, self.laws, 'law')
            self.micro_f1_law, self.macro_f1_law, self.weighted_f1_law = \
                cal_multi_label_score(self.logits_law, self.laws)
            # self.accuracy_accu = cal_multi_label_accuracy(self.logits_accu, self.accusations, 'accu')
            self.micro_f1_accu, self.macro_f1_accu, self.weighted_f1_accu = \
                cal_multi_label_score(self.logits_accu, self.accusations)
            self.accuracy_impris = cal_single_label_accuracy(self.logits_impris, self.imprisonments, 'impris')

        tf.summary.scalar("loss_law", self.loss_law)
        tf.summary.scalar("loss_accu", self.loss_accu)
        tf.summary.scalar("loss_impris", self.loss_impris)
        tf.summary.scalar("loss_l2", self.loss_l2)
        tf.summary.scalar("loss_all", self.loss_all)
        tf.summary.scalar("micro_f1_law", self.micro_f1_law)
        tf.summary.scalar("macro_f1_law", self.macro_f1_law)
        tf.summary.scalar("weighted_f1_law", self.weighted_f1_law)
        tf.summary.scalar("score_law", (self.macro_f1_law + self.micro_f1_law) / 2.)
        tf.summary.scalar("micro_f1_accu", self.micro_f1_accu)
        tf.summary.scalar("macro_f1_accu", self.macro_f1_accu)
        tf.summary.scalar("weighted_f1_accu", self.weighted_f1_accu)
        tf.summary.scalar("score_accu", (self.macro_f1_accu + self.micro_f1_accu) / 2.0)
        tf.summary.scalar("accuracy_impris", self.accuracy_impris)
        tf.summary.histogram('loss_law_dist', self.loss_law)
        tf.summary.histogram('loss_accu_dist', self.loss_accu)
        tf.summary.histogram('loss_impris_dist', self.loss_impris)

        if self.config.mode == 'train':
            # train op
            self.train_all_op = self.train(self.loss_all)
            self.train_law_op = self.train(self.loss_law)
            self.train_accu_op = self.train(self.loss_accu)
            self.train_impris_op = self.train(self.loss_impris)

        self.summaries = tf.summary.merge_all()

    def inference(self):
        """
        :return:
        """
        inputs = tf.nn.embedding_lookup(self.embedding, self.facts)
        # embedded_sent_words = [tf.squeeze(x) for x in tf.split(embedded_words, self.config.num_sens, axis=1)]
        # word_atts = []
        # # word level
        # for i in range(self.config.num_sens):
        #     sent = embedded_sent_words[i]
        #     reuse_flag = True if i > 0 else False
        #     sent = tf.reshape(sent, [-1, self.config.sent_len, self.config.embed_size])
        #     words_encoded, word_encoded2 = \
        #         self.bi_lstm(sent, 'word_level', self.config.hidden_size, reuse_flag)
        #     # word attention
        #     word_att = self.attention_multihop(words_encoded, "word_level", reuse=reuse_flag)
        #     word_att = slim.batch_norm(word_att, scale=True, epsilon=1e-5, is_training=self.is_training)
        #     word_att = slim.dropout(word_att, keep_prob=self.keep_prob)
        #     # word_attention=tf.concat([word_attention,word_encodeded2],axis=1) # TODO
        #     word_atts.append(word_att)
        #
        # # sent level
        # sent_encoder_input = tf.stack(word_atts, axis=1)
        # sent_encoded, sent_encoded2 = self.bi_lstm(sent_encoder_input, "sent_level", self.config.hidden_size * 2)
        # sent_att = self.attention_multihop(sent_encoded, "sent_level")
        # sent_att = slim.batch_norm(sent_att, scale=True, epsilon=1e-5, is_training=self.is_training)
        # sent_att = slim.dropout(sent_att, keep_prob=self.keep_prob)
        # net_flatted = tf.reshape(sent_att, shape=[-1, self.config.hidden_size * 4])
        feature_obj = self.config.Feature(self.config)
        net_flatted = feature_obj.feature(inputs, self.keep_prob, self.is_training)

        with tf.variable_scope("output"):
            with slim.arg_scope([slim.fully_connected],
                                normalizer_fn=slim.batch_norm,
                                normalizer_params={'is_training': self.is_training}):
                net_law = slim.fully_connected(net_flatted, num_outputs=1024, scope='liner_law_0')
                net_law = slim.dropout(net_law, keep_prob=self.keep_prob)
                net_accu = slim.fully_connected(net_flatted, num_outputs=1024, scope='liner_accu_0')
                net_accu = slim.dropout(net_accu, keep_prob=self.keep_prob)
                net_impris = slim.fully_connected(net_flatted, num_outputs=1024, scope='liner_impris_0')
                net_impris = slim.dropout(net_impris, keep_prob=self.keep_prob)
            with slim.arg_scope([slim.fully_connected], activation_fn=None):
                logits_law = slim.fully_connected(net_law, num_outputs=len(self.config.data_obj.law2id), scope='law')
                logits_accu = slim.fully_connected(net_accu, num_outputs=len(self.config.data_obj.accu2id), scope='accu')
                logits_impris = slim.fully_connected(net_impris, num_outputs=len(self.config.data_obj.imprisonment2id), scope='impris')

        return logits_law, logits_accu, logits_impris

    def bi_lstm(self, inputs, level, hidden_size, reuse=False):
        # batch_size = inputs.get_shape().as_list()[0]
        with tf.variable_scope('bi_lstm_{}'.format(level), reuse=reuse):
            fw_cell = rnn.GRUCell(hidden_size)
            fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=self.keep_prob)
            bw_cell = rnn.GRUCell(hidden_size)
            bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=self.keep_prob)
            # init_fw_state = tf.get_variable("init_fw_stata", shape=[batch_size, hidden_size])
            # init_bw_state = tf.get_variable("init_bw_stata", shape=[batch_size, hidden_size])
            outputs, hidden_states = tf.nn.bidirectional_dynamic_rnn(fw_cell,
                                                                     bw_cell,
                                                                     inputs,
                                                                     dtype=tf.float32)
                                                                     # initial_state_fw=init_fw_state,
                                                                     # initial_state_bw=init_bw_state)
        outputs = tf.concat(outputs, axis=2)
        hidden_states = tf.concat([hidden_states[0], hidden_states[1]], axis=1)
        return outputs, hidden_states

    def attention_multihop(self, inputs, level, reuse=False,
                           initializer=tf.random_normal_initializer(stddev=0.1)):
        """

        :param inputs:
        :param level:
        :param reuse:
        :return:
        """
        num_units = inputs.get_shape().as_list()[-1] / self.config.num_hops  # get last dimension
        attention_rep_list=[]
        for i in range(self.config.num_hops):
            with tf.variable_scope("attention_{}_{}".format(i, level), reuse=reuse):
                v_attention = tf.get_variable("u_attention_{}".format(level),
                                              shape=[num_units],
                                              initializer=initializer)
                # 1.one-layer MLP
                u = tf.layers.dense(inputs, num_units, activation=tf.nn.tanh, use_bias=True)
                # 2.compute weight by compute simility of u and attention vector v
                score = tf.multiply(u, v_attention)  # [batch_size,seq_length,num_units]
                weight = tf.reduce_sum(score, axis=2, keep_dims=True)  # [batch_size,seq_length,1]
                # 3.weight sum
                attention_rep = tf.reduce_sum(tf.multiply(u, weight), axis=1)  # [batch_size,num_units]. TODO here we not use original input_sequences but transformed version of input: u.
                attention_rep_list.append(attention_rep)

        attention_representation = tf.concat(attention_rep_list, axis=-1)
        return attention_representation

    def assign_pretrain_embedding(self, pretrain_embedding):
        with tf.variable_scope("assign_pretrain_embedding"):
            embedding_init = tf.assign(self.embedding, pretrain_embedding)
            return embedding_init

    def train(self, loss):
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        if update_ops:
            updates = tf.group(*update_ops)
            loss = control_flow_ops.with_dependencies([updates], loss)

        learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step_tensor,
                                                   self.config.decay_steps, self.config.decay_rate)
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = slim.learning.create_train_op(loss, optimizer, self.global_step_tensor,
                                                 clip_gradient_norm=self.config.clip_gradients)
        tf.summary.scalar('lr', learning_rate)
        return train_op

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

