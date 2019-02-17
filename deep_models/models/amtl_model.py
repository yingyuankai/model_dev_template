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
        self.loss_weight_death = self._make_single_loss_weight(self.config.loss_weight_death)

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
        self.facts = tf.placeholder(tf.int32, shape=[None, self.config.max_seq_len], name="fact_input")
        self.laws = tf.placeholder(tf.int32, shape=[None, None], name="law_input")
        self.accusations = tf.placeholder(tf.int32, shape=[None, None], name='accu_input')
        self.imprisonments = tf.placeholder(tf.int32, shape=[None], name='impris_input')
        # self.money = tf.placeholder(tf.int32, shape=[None], name='money_input')
        self.death = tf.placeholder(tf.int32, shape=[None], name='death_input')
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
        self.logits_law, self.logits_accu, self.logits_impris, self.logits_death, loss_diff = self.inference()

        # loss
        self.loss_l2 = l2_loss(self.config.l2_lambda)
        # law loss
        self.loss_law = loss_multi_label(self.logits_law, self.laws,
                                         loss_weights_single=self.loss_weight_law if self.config.use_label_loss_weight else None)
        self.loss_law = self.config.task_loss_weights[0] * self.loss_law
        # accu loss
        self.loss_accu = loss_multi_label(self.logits_accu, self.accusations,
                                          loss_weights_single=self.loss_weight_accu if self.config.use_label_loss_weight else None)
        self.loss_accu = self.config.task_loss_weights[1] * self.loss_accu
        # impris loss
        self.loss_impris = loss_single_label(self.logits_impris, self.imprisonments,
                                             loss_weight=self.loss_weight_impris if self.config.use_label_loss_weight else None)
        self.loss_impris = self.config.task_loss_weights[2] * self.loss_impris
        # death loss
        self.loss_death = loss_single_label(self.logits_death, self.death,
                                            loss_weight=self.loss_weight_death if self.config.use_label_loss_weight else None)
        self.loss_death = self.config.task_loss_weights[3] * self.loss_death
        if self.config.adv:
            self.loss_all = self.loss_law + \
                            self.loss_accu + \
                            self.loss_impris + \
                            self.loss_death + \
                            self.config.diff_loss_lambad * loss_diff + \
                            self.loss_l2
        else:
            self.loss_all = self.loss_law + \
                            self.loss_accu + \
                            self.loss_impris + \
                            self.loss_death + \
                            self.loss_l2

        # accuracy
        with tf.variable_scope("score"):
            # self.accuracy_law = cal_multi_label_accuracy(self.logits_law, self.laws, 'law')
            self.micro_f1_law, self.macro_f1_law, self.weighted_f1_law = \
                cal_multi_label_score(self.logits_law, self.laws)
            # self.accuracy_accu = cal_multi_label_accuracy(self.logits_accu, self.accusations, 'accu')
            self.micro_f1_accu, self.macro_f1_accu, self.weighted_f1_accu = \
                cal_multi_label_score(self.logits_accu, self.accusations)
            self.accuracy_impris = cal_single_label_accuracy(self.logits_impris, self.imprisonments, 'impris')
            self.accuracy_death = cal_single_label_accuracy(self.logits_death, self.death, 'death')

        tf.summary.scalar("loss_law", self.loss_law)
        tf.summary.scalar("loss_accu", self.loss_accu)
        tf.summary.scalar("loss_impris", self.loss_impris)
        tf.summary.scalar("loss_death", self.loss_death)
        tf.summary.scalar("loss_diff", loss_diff)
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
        tf.summary.scalar("accuracy_death", self.accuracy_death)

        if self.config.mode == 'train':
            # train op
            self.train_all_op = self.train(self.loss_all)

        self.summaries = tf.summary.merge_all()

    def inference(self):
        """

        :return:
        """
        inputs = tf.nn.embedding_lookup(self.embedding, self.facts)
        inputs = tf.expand_dims(inputs, axis=-1)
        feature_obj = self.config.Feature(self.config.textcnn_config)
        net_flatted_law = feature_obj.feature(inputs, self.is_training, "law")
        net_flatted_accu = feature_obj.feature(inputs, self.is_training, "accu")
        net_flatted_impris = feature_obj.feature(inputs, self.is_training, "impris")
        net_flatted_share = feature_obj.feature(inputs, self.is_training, "share")

        loss_diff_law = diff_loss(net_flatted_share, net_flatted_law)
        loss_diff_accu = diff_loss(net_flatted_share, net_flatted_accu)
        loss_diff_impris = diff_loss(net_flatted_share, net_flatted_impris)
        loss_diff = self.config.task_loss_weights[0] * loss_diff_law + \
                    self.config.task_loss_weights[1] * loss_diff_accu + \
                    self.config.task_loss_weights[2] * loss_diff_impris

        if self.config.adv:
            net_flatted_law = tf.concat([net_flatted_law, net_flatted_share], axis=1)
            net_flatted_accu = tf.concat([net_flatted_accu, net_flatted_share], axis=1)
            net_flatted_impris = tf.concat([net_flatted_impris, net_flatted_share], axis=1)

        logits_law = project_output(net_flatted_law,
                                    "law",
                                    len(self.config.data_obj.law2id),
                                    self.keep_prob,
                                    self.is_training)
        logits_accu = project_output(net_flatted_accu,
                                     "accu",
                                     len(self.config.data_obj.accu2id),
                                     self.keep_prob,
                                     self.is_training)
        logits_impris = project_output(tf.concat([net_flatted_impris, tf.nn.relu(logits_law), tf.nn.relu(logits_accu)], axis=-1) if self.config.use_prior else net_flatted_impris,
                                       "impris",
                                       len(self.config.data_obj.imprisonment2id),
                                       self.keep_prob,
                                       self.is_training)
        logits_death = project_output(tf.concat([net_flatted_impris, tf.nn.relu(logits_law), tf.nn.relu(logits_accu)], axis=-1) if self.config.use_prior else net_flatted_impris,
                                     "death",
                                     3,
                                     self.keep_prob,
                                     self.is_training)
        return logits_law, logits_accu, logits_impris, logits_death, loss_diff

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
                                                 clip_gradient_norm=self.config.clip_gradients,
                                                 summarize_gradients=True)
        tf.summary.scalar('lr', learning_rate)
        return train_op

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

