import os
import numpy as np
from tqdm import tqdm
import tensorflow as tf
from utils.loss_utils import running_avg_loss
from utils.nn_utils import list_to_multi_hot, list_to_double_multi_hot
from tflearn.data_utils import pad_sequences, to_categorical
from deep_models.base.base_train import BaseTrain


class Trainer(BaseTrain):
    def __init__(self, model, config, gpu_config):
        super(Trainer, self).__init__(model, config)
        self.train_dataset = self.config.DataGenerator('train', self.config.num_epochs, self.config.batch_size, config)
        self.sv = self.build_sv()
        self.sess = self.sv.prepare_or_wait_for_session(config=gpu_config)
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "train"), self.sess.graph)
        # self.model.load(self.sess)
        if self.config.use_pretrain_embedding and \
                self.model.has_ckpt(self.sess) is False and \
                self.config.pre_train_ckpt_dir == '':
            print("load pretrain embedding...")
            pretrain_embedding = self.config.data_obj.load_pretrain_embedding()
            self.sess.run(self.model.embedding_init, feed_dict={self.model.embedding_placeholder: pretrain_embedding})

    def train_epoch(self):
        avg_loss_all, avg_loss_law, avg_loss_accu, avg_loss_impris = 0., 0., 0., 0.
        avg_acc_impris = 0.
        avg_macro_f1_law, avg_micro_f1_law, avg_weighted_f1_law = 0., 0., 0.
        avg_macro_f1_accu, avg_micro_f1_accu, avg_weighted_f1_accu = 0., 0., 0.

        while not self.sv.should_stop():
            loss_law, loss_accu, loss_impris, loss_l2, loss_all, \
            macro_f1_law, micro_f1_law, weighted_f1_law, \
            macro_f1_accu, micro_f1_accu, weighted_f1_accu, \
            acc_impris, summaries = self.train_step()
            global_steps = self.model.global_step_tensor.eval(self.sess)

            avg_loss_law = running_avg_loss(avg_loss_law, loss_law)
            avg_loss_accu = running_avg_loss(avg_loss_accu, loss_accu)
            avg_loss_impris = running_avg_loss(avg_loss_impris, loss_impris)
            avg_loss_all = running_avg_loss(avg_loss_all, loss_all)

            avg_macro_f1_law = running_avg_loss(avg_macro_f1_law, macro_f1_law)
            avg_micro_f1_law = running_avg_loss(avg_micro_f1_law, micro_f1_law)
            avg_weighted_f1_law = running_avg_loss(avg_weighted_f1_law, weighted_f1_law)

            avg_macro_f1_accu = running_avg_loss(avg_macro_f1_accu, macro_f1_accu)
            avg_micro_f1_accu = running_avg_loss(avg_micro_f1_accu, micro_f1_accu)
            avg_weighted_f1_accu = running_avg_loss(avg_weighted_f1_accu, weighted_f1_accu)

            avg_acc_impris = running_avg_loss(avg_acc_impris, acc_impris)
            self.summary_writer.add_summary(summaries, global_step=global_steps)
            if global_steps % 100 == 0:
                print("global_steps:{}\tloss_all:{:.3f}\tloss_law:{:.3f}\tloss_accu:{:.3f}\tloss_impris:{:.3f}"
                      "\tmacro_f1_law:{:.3f}\tmicro_f1_law:{:.3f}\tweighted_f1_law:{:.3f}"
                      "\tavg_macro_f1_accu:{:.3f}\tavg_micro_f1_accu:{:.3f}\tavg_weighted_f1_accu:{:.3f}\t"
                      "acc_impris:{}".
                      format(global_steps, avg_loss_all, avg_loss_law, avg_loss_accu, avg_loss_impris,
                             avg_macro_f1_law, avg_micro_f1_law, avg_weighted_f1_law,
                             avg_macro_f1_accu, avg_micro_f1_accu, avg_weighted_f1_accu,
                             avg_acc_impris))
                self.summary_writer.flush()
        self.sv.Stop()

    def train_step(self):
        facts_batch, laws_batch, accus_batch, impris_batch, money_batch, death_batch = self.train_dataset.next_batch(self.sess)
        facts = pad_sequences(facts_batch, maxlen=self.config.max_seq_len,
                              value=self.config.data_obj.vocab.word_to_id(self.config.PAD_TOKEN))
        laws = list_to_multi_hot(laws_batch, len(self.config.data_obj.law2id))
        accus = list_to_multi_hot(accus_batch, len(self.config.data_obj.accu2id))
        impris = impris_batch
        money = money_batch
        death = death_batch

        # if self.config.train_tasks:
        train_ops = [getattr(self.model, "train_{}_op".format(train_task)) for train_task in self.config.train_tasks]

        to_reture = [self.model.loss_law, self.model.loss_accu, self.model.loss_impris,
                     self.model.loss_l2, self.model.loss_all,
                     self.model.macro_f1_law,
                     self.model.micro_f1_law,
                     self.model.weighted_f1_law,
                     self.model.macro_f1_accu,
                     self.model.micro_f1_accu,
                     self.model.weighted_f1_accu,
                     self.model.accuracy_impris,
                     self.model.summaries]
        to_reture.extend(train_ops)
        feed_dict = {self.model.facts: facts,
                     self.model.laws: laws,
                     self.model.accusations: accus,
                     self.model.imprisonments: impris,
                     self.model.death: death,
                     self.model.is_training: True,
                     self.model.keep_prob: self.config.keep_prob}
        return self.sess.run(to_reture, feed_dict=feed_dict)[: len(to_reture) - len(train_ops)]
