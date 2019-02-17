import math
import tensorflow as tf
import tensorflow.contrib.slim as slim
from tensorflow.python.ops import control_flow_ops
from tensorflow.contrib.layers import optimize_loss
from deep_models.base.base_model import BaseModel
from utils.nn_utils import *
from utils.math_utils import softmax, shift_to_one
from deep_models import library

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
        self.dropout = tf.placeholder(tf.float32, name='dropout')
        self.seq_len = self.facts.shape[-1]

    def build_model(self):
        self._add_placeholders()
        self.seq_len = self.facts.shape[-1]

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

        self.loss_all = loss_law + loss_accu + loss_impris + self.loss_l2

        # accuracy
        with tf.variable_scope("f1"):
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

        if self.config.mode == 'train':
            # train op
            self.train_all_op = self.train(self.loss_all)
            self.train_law_op = self.train(self.loss_law)
            self.train_accu_op = self.train(self.loss_accu)
            self.train_impris_op = self.train(self.loss_impris)
        self.summaries = tf.summary.merge_all()

    def inference(self):
        fn_dict = {'reduce_max': tf.reduce_max,
                   'reduce_sum': tf.reduce_sum,
                   'concat': tf.reshape}
        region_merge_fn = fn_dict.get(self.config.region_merge_fn, tf.reduce_max)

        # Layers
        assert (self.config.variant in ['WC', 'CW', 'win_pool', 'scalar', 'multi_region'])
        if self.config.variant == 'WC':
            L = library.WordContextRegionEmbeddingLayer
        elif self.config.variant == 'CW':
            L = library.ContextWordRegionEmbeddingLayer
        elif self.config.variant == 'win_pool':
            L = library.WindowPoolEmbeddingLayer
        elif self.config.variant == 'scalar':
            L = library.ScalarRegionEmbeddingLayer
        if self.config.variant == 'multi_region':
            L = library.MultiRegionEmbeddingLayer
            assert (type(self.config.region_sizes) is list)
            assert (len(self.config.region_sizes) * self.config.embed_size == self.config.hidden_depth)
            self._region_emb_layer = L(self.config.data_obj.vocab.num_ids(),
                                       self.config.embed_size,
                                       self.config.region_sizes,
                                       region_merge_fn=region_merge_fn)
        else:
            assert (type(self.config.region_size) is int)
            assert (self.config.embed_size == self.config.hidden_depth)
            self._region_emb_layer = L(self.config.data_obj.vocab.num_ids(),
                                       self.config.embed_size,
                                       self.config.region_size,
                                       region_merge_fn=region_merge_fn)
        self.embedding_init = self._region_emb_layer.assign_pretrain_embedding(self.embedding_placeholder)
        self._vsum_layer = library.WeightedVSumLayer()
        fc_layer_law = library.FCLayer(self.config.hidden_depth, len(self.config.data_obj.law2id), name='law')
        fc_layer_accu = library.FCLayer(self.config.hidden_depth, len(self.config.data_obj.accu2id), name='accu')
        fc_layer_impris = library.FCLayer(self.config.hidden_depth, len(self.config.data_obj.imprisonment2id), name='impris')
        with tf.variable_scope("output"):
            if self.config.mode == 'multi_region':
                logits_law = self.logits_multi_regioin(fc_layer=fc_layer_law)
                logits_accu = self.logits_multi_regioin(fc_layer=fc_layer_accu)
                logits_impris = self.logits_multi_regioin(fc_layer=fc_layer_impris)
            else:
                logits_law = self.logits(fc_layer=fc_layer_law)
                logits_accu = self.logits(fc_layer=fc_layer_accu)
                logits_impris = self.logits(fc_layer=fc_layer_impris)
        return logits_law, logits_accu, logits_impris


    def train(self, loss):
        learning_rate = tf.train.exponential_decay(self.config.learning_rate, self.global_step_tensor,
                                                   self.config.decay_steps, self.config.decay_rate)
        train_op = optimize_loss(loss, global_step=self.global_step_tensor,
                                 learning_rate=learning_rate, optimizer="Adam",
                                 clip_gradients=self.config.clip_gradients)
        tf.summary.scalar('lr', learning_rate)
        return train_op

    def init_saver(self):
        # here you initialize the tensorflow saver that will be used in saving the checkpoints.
        self.saver = tf.train.Saver(max_to_keep=self.config.max_to_keep)

    def logits(self, fc_layer):
        """logits"""

        region_emb = self._region_emb_layer(self.facts)

        # Mask padding elements (id > 0)
        region_radius = int(self.config.region_size / 2)
        # trimed_seq = self.sequence[:, region_radius : self.sequence.get_shape()[1] - region_radius]
        trimed_seq = self.facts[..., region_radius: self.facts.get_shape()[1] - region_radius]

        def mask(x):
            """mask
            """
            return tf.cast(tf.greater(tf.cast(x, tf.int32), tf.constant(0)), tf.float32)

        weight = tf.map_fn(mask, trimed_seq, dtype=tf.float32, back_prop=False)
        weight = tf.expand_dims(weight, -1)
        # End mask

        h = self._vsum_layer((region_emb, weight))
        h = fc_layer(h)
        return h

    def logits_multi_regioin(self, fc_layer):
        """logits"""

        multi_region_emb = self._region_emb_layer(self.facts)
        assert (len(multi_region_emb) == len(self.config.region_sizes))

        def mask(x):
            """mask
            """
            return tf.cast(tf.greater(tf.cast(x, tf.int32), tf.constant(0)), tf.float32)

        h = []
        for i, region_emb in enumerate(multi_region_emb):
            # Mask padding elements (id > 0)
            region_radius = int(self.config.region_sizes[i] / 2)
            trimed_seq = self.facts[..., region_radius: self.facts.get_shape()[1] - region_radius]
            weight = tf.map_fn(mask, trimed_seq, dtype=tf.float32, back_prop=False)
            weight = tf.expand_dims(weight, -1)
            # End mask
            h.append(self._vsum_layer((region_emb, weight)))

        h = tf.concat(h, 1)
        h = self.fc_layer(h)
        return h