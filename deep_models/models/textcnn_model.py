import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.layers import optimize_loss
from deep_models.base.base_model import BaseModel
from utils.nn_utils import *
from utils.math_utils import softmax, shift_to_one


class Model(BaseModel):
    def __init__(self, config):
        super(Model, self).__init__(config)
        self.initializer = tf.truncated_normal_initializer(stddev=1e-4)
        self._make_loss_weight()
        self.build_model()
        self.init_saver()

    def _make_loss_weight(self):
        # for single label
        self.loss_weight_accu = self._make_single_loss_weight(self.config.loss_weight_accu)
        self.loss_weight_law = self._make_single_loss_weight(self.config.loss_weight_law)
        self.loss_weight_impris = self._make_single_loss_weight(self.config.loss_weight_impris)

        # for double multi label
        # self.loss_weight_accu_double = self._make_double_loss_weight(self.config.loss_weight_accu)
        # self.loss_weight_law_double = self._make_double_loss_weight(self.config.loss_weight_law)

    def _make_double_loss_weight(self, loss_weight_list):
        loss_weight = []
        sum_weight = sum(loss_weight_list)
        loss_weight_list_ = [sum_weight - lw for lw in loss_weight_list]
        for i in range(len(loss_weight_list)):
            tmp_loss_weight = softmax([math.log(sum_weight / max(loss_weight_list[i], 1)),
                                       math.log(sum_weight / max(loss_weight_list_[i], 1))])
            loss_weight.append(tmp_loss_weight.tolist())
        return tf.constant(loss_weight, dtype=tf.float32)

    def _make_single_loss_weight(self, loss_weight_list):
        loss_weight = []
        sum_weight = sum(loss_weight_list)
        for i in range(len(loss_weight_list)):
            loss_weight.append(math.log(sum_weight / max(loss_weight_list[i], 1)))
        # loss_weight = softmax(loss_weight)
        loss_weight = shift_to_one(loss_weight)
        return tf.constant(loss_weight, dtype=tf.float32)


    def _add_placeholders(self):
        self.is_training = tf.placeholder(tf.bool, name="is_training")
        self.facts = tf.placeholder(tf.int32, shape=[None, None], name="fact_input")
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
            self.embedding = tf.get_variable('words',
                                             shape=[self.config.data_obj.vocab.num_ids(), self.config.embed_size],
                                             initializer=self.initializer)
            # self.embedding = tf.Variable(
            #     tf.constant(0., shape=[self.config.data_obj.vocab.num_ids(), self.config.embed_size]),
            #     trainable=False, name='words')
            self.embedding_init = self.assign_pretrain_embedding(self.embedding_placeholder)
        # network
        self.logits_law, self.logits_accu, self.logits_impris = self.inference()

        # loss
        self.loss_l2 = l2_loss(self.config.l2_lambda)

        loss_law = loss_double_multi_label(self.logits_law, self.laws,
                                           len(self.config.data_obj.law2id),
                                           loss_weights_single=self.loss_weight_law)
                                           # , loss_weights_double=self.loss_weight_law_double)
        self.loss_law = loss_law + self.loss_l2

        loss_accu = loss_double_multi_label(self.logits_accu, self.accusations,
                                            len(self.config.data_obj.accu2id),
                                            loss_weights_single=self.loss_weight_accu)
                                            # ,loss_weights_double=self.loss_weight_accu_double)
        self.loss_accu = loss_accu + self.loss_l2

        loss_impris = loss_single_label(self.logits_impris, self.imprisonments,
                                        loss_weight=self.loss_weight_impris)
        self.loss_impris = loss_impris + self.loss_l2

        self.loss_all = loss_law + loss_accu + loss_impris + self.loss_l2

        # accuracy
        with tf.variable_scope("f1"):
            # self.accuracy_law = cal_multi_label_accuracy(self.logits_law, self.laws, 'law')
            self.micro_f1_law, self.macro_f1_law, self.weighted_f1_law = \
                cal_double_multi_label_score(self.logits_law, self.laws, len(self.config.data_obj.law2id))
            # self.accuracy_accu = cal_multi_label_accuracy(self.logits_accu, self.accusations, 'accu')
            self.micro_f1_accu, self.macro_f1_accu, self.weighted_f1_accu = \
                cal_double_multi_label_score(self.logits_accu, self.accusations, len(self.config.data_obj.accu2id))
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
        self.embedded_words = tf.nn.embedding_lookup(self.embedding, self.facts)
        self.sentence_embedding_expanded = tf.expand_dims(self.embedded_words, -1)
        net_outputs = []
        total_output = self.config.output_num_per_filter * len(self.config.filter_sizes)
        for i, filter_size in enumerate(self.config.filter_sizes):
            with tf.name_scope("conv_pool_{}".format(filter_size)):
                with slim.arg_scope([slim.conv2d], padding='VALID', normalizer_fn=slim.batch_norm,
                                    normalizer_params={'is_training': self.is_training}):
                    net = slim.conv2d(self.sentence_embedding_expanded,
                                      self.config.output_num_per_filter,
                                      [filter_size, self.config.embed_size],
                                      scope='conv_0_{}'.format(filter_size))
                    net = slim.conv2d(net,
                                      self.config.output_num_per_filter, [filter_size, 1],
                                      scope='conv_1_{}'.format(filter_size))
                net = tf.reduce_max(net, axis=1, name='max_pool')
                net_outputs.append(net)

        net_c = tf.concat(net_outputs, -1)
        net_flatted = tf.reshape(net_c, [-1, total_output])

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
                logits_law = slim.fully_connected(net_law, num_outputs=2 * len(self.config.data_obj.law2id), scope='law')
                logits_accu = slim.fully_connected(net_accu, num_outputs=2 * len(self.config.data_obj.accu2id), scope='accu')
                logits_impris = slim.fully_connected(net_impris, num_outputs=len(self.config.data_obj.imprisonment2id), scope='impris')

        return logits_law, logits_accu, logits_impris

    def get_weight_bias(self, input_dim, output_dim, name=""):
        W = tf.get_variable(name="W_{}".format(name), shape=[input_dim, output_dim], initializer=self.initializer)
        b = tf.get_variable(name="b_{}".format(name), shape=[output_dim], initializer=self.initializer)
        return W, b

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
        learning_rate = tf.maximum(self.config.min_lr, learning_rate)
        # with tf.control_dependencies(update_ops):
        optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
        train_op = slim.learning.create_train_op(loss, optimizer, self.global_step_tensor,
                                                 clip_gradient_norm=self.config.clip_gradients)
        # train_op = optimize_loss(loss, global_step=self.global_step_tensor,
        #                          learning_rate=learning_rate, optimizer="Adam",
        #                          clip_gradients=self.config.clip_gradients)
        tf.summary.scalar('lr', learning_rate)
        return train_op

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

