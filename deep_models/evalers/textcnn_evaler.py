import os
import numpy as np
import time
from tqdm import tqdm
import tensorflow as tf
from utils.loss_utils import running_avg_loss
from utils.nn_utils import list_to_multi_hot, list_to_double_multi_hot
from tflearn.data_utils import pad_sequences, to_categorical
from deep_models.base.base_train import BaseTrain


class Evaler(BaseTrain):
    def __init__(self, model, config, gpu_config):
        super(Evaler, self).__init__(model, config)
        self.valid_dataset = self.config.DataGenerator('valid', self.config.num_epochs, self.config.batch_size * 10, config)
        self.sess = tf.Session(config=gpu_config)
        self.summary_writer = tf.summary.FileWriter(os.path.join(self.config.summary_dir, "valid"), self.sess.graph)

    def eval_epoch(self):
        avg_loss_all, avg_loss_law, avg_loss_accu, avg_loss_impris = 0., 0., 0., 0.
        best_avg_loss_all, best_avg_loss_law, best_avg_loss_accu, best_avg_loss_impris = \
            float('inf'), float('inf'), float('inf'), float('inf')
        avg_acc_impris = 0.
        avg_macro_f1_law, avg_micro_f1_law, avg_weighted_f1_law = 0., 0., 0.
        avg_macro_f1_accu, avg_micro_f1_accu, avg_weighted_f1_accu = 0., 0., 0.
        self.sess.run(tf.local_variables_initializer())
        coord = tf.train.Coordinator()
        threads = tf.train.start_queue_runners(sess=self.sess, coord=coord)
        try:
            while not coord.should_stop():
                time.sleep(self.config.eval_interval_secs)
                self.model.load_ckpt(self.sess)

                loss_law, loss_accu, loss_impris, loss_l2, loss_all, \
                macro_f1_law, micro_f1_law, weighted_f1_law, \
                macro_f1_accu, micro_f1_accu, weighted_f1_accu, \
                acc_impris, summaries = self.eval_step()

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

                print("global_steps:{}\tloss_all:{:.3f}\tloss_law:{:.3f}\tloss_accu:{:.3f}\tloss_impris:{:.3f}"
                      "\tmacro_f1_law:{:.3f}\tmicro_f1_law:{:.3f}\tweighted_f1_law:{:.3f}"
                      "\tavg_macro_f1_accu:{:.3f}\tavg_micro_f1_accu:{:.3f}\tavg_weighted_f1_accu:{:.3f}\t"
                      "acc_impris:{}".
                      format(global_steps, avg_loss_all, avg_loss_law, avg_loss_accu, avg_loss_impris,
                             avg_macro_f1_law, avg_micro_f1_law, avg_weighted_f1_law,
                             avg_macro_f1_accu, avg_micro_f1_accu, avg_weighted_f1_accu,
                             avg_acc_impris))
                self.summary_writer.flush()
                if global_steps == 0:
                    continue

                if "impris" in self.config.train_tasks and best_avg_loss_impris > avg_loss_impris:
                    tf.logging.info('Found new best model (%f vs. %f)', running_avg_loss, best_avg_loss_all)
                    best_avg_loss_impris = avg_loss_impris
                    self.model.saver.save(self.sess, self.config.checkpoint_dir + '/model.best', write_state=False)
                elif "law" in self.config.train_tasks and best_avg_loss_law > avg_loss_law:
                    tf.logging.info('Found new best model (%f vs. %f)', running_avg_loss, best_avg_loss_all)
                    best_avg_loss_law = avg_loss_law
                    self.model.saver.save(self.sess, self.config.checkpoint_dir + '/model.best', write_state=False)
                elif "accu" in self.config.train_tasks and best_avg_loss_accu > avg_loss_accu:
                    tf.logging.info('Found new best model (%f vs. %f)', running_avg_loss, best_avg_loss_all)
                    best_avg_loss_accu = avg_loss_accu
                    self.model.saver.save(self.sess, self.config.checkpoint_dir + '/model.best', write_state=False)
                elif "all" in self.config.train_tasks and best_avg_loss_all > avg_loss_all:
                    tf.logging.info('Found new best model (%f vs. %f)', running_avg_loss, best_avg_loss_all)
                    best_avg_loss_all = avg_loss_all
                    self.model.saver.save(self.sess, self.config.checkpoint_dir + '/model.best', write_state=False)
        except tf.errors.OutOfRangeError as e:
            tf.logging.info('\nDone Training, step limit reached.')
        finally:
            coord.request_stop()
        coord.join(threads)

    def eval_step(self):
        facts_batch, laws_batch, accus_batch, impris_batch, money_batch, death_batch = self.valid_dataset.next_batch(self.sess)
        facts = pad_sequences(facts_batch, maxlen=self.config.max_seq_len,
                                   value=self.config.data_obj.vocab.word_to_id(self.config.PAD_TOKEN))
        laws = list_to_multi_hot(laws_batch, len(self.config.data_obj.law2id))
        accus = list_to_multi_hot(accus_batch, len(self.config.data_obj.accu2id))
        impris = impris_batch
        death = death_batch

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

        feed_dict = {self.model.facts: facts,
                     self.model.laws: laws,
                     self.model.accusations: accus,
                     self.model.imprisonments: impris,
                     self.model.death: death,
                     self.model.is_training: False,
                     self.model.keep_prob: 1.}
        return self.sess.run(to_reture, feed_dict=feed_dict)[: len(to_reture)]
