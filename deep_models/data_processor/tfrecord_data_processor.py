import os
import codecs
import json
import tensorflow as tf
from tqdm import tqdm
import pickle
import numpy as np
import math
from random import random, uniform, shuffle
from utils.strutil import is_num
# from gensim.models.wrappers.fasttext import FastText
from deep_models.base.base_data_process import BaseDataProcess
from utils.dirs import create_dirs
from utils.data_new import Data
from utils.math_utils import softmax


class DataProcessor(BaseDataProcess):
    def __init__(self, config):
        super(DataProcessor, self).__init__(config)
        self._data = Data(config)
        self._loss_weight_law = [0] * len(self._data.law2id)
        self._loss_weight_accu = [0] * len(self._data.accu2id)
        self._loss_weight_impris = [0] * len(self._data.imprisonment2id)
        self._loss_weight_death = [0] * 3
        if self._config.rebalance:
            self.raw_loss_weight_accu, self.raw_loss_weight_law, self.raw_loss_weight_impris, self.raw_loss_weight_death = \
                pickle.load(open(self._config.raw_loss_weight_path, "rb"))
        # self._load_pretrain_embedding()
        # load synonyms
        if os.path.isfile(self._config.synonym_path):
            self.synonyms = pickle.load(open(self._config.synonym_path, "rb"))
        else:
            self.synonyms = {}

    def _load_pretrain_embedding(self):
        self.model = FastText.load(self._config.pretrain_word2vec_embedding)

    def _get_most_similar_words(self, word):
        try:
            nn_words = self.model.most_similar(word)
        except KeyError as e:
            return None
        return nn_words

    def augment_data(self, fact, word_pos, n_time):
        """
        Augment data:

        1) delete word with probability 0.1;
        2) replace word with similarity word with probability 0.5
        3) retain word with probability 0.4
        :param fact:
        :return:
        """
        fact_augmented = []
        for i in range(n_time):
            new_fact = []
            for i in range(len(fact)):
                word = fact[i]
                pos = word_pos[i]
                rand = random()
                if rand < 0.1:
                    continue
                elif rand < 0.5:
                    new_fact.append(word)
                else:
                    if pos.find('v') == -1 or len(word) == 1:
                        new_fact.append(word)
                        continue
                    synonym = self._get_synonym(word)
                    new_fact.append(synonym)
            fact_augmented.append(new_fact)
        return fact_augmented

    def _get_synonym(self, word):
        if word not in self.synonyms:
            return word
            # most_similar_words = self._get_most_similar_words(word)
            # if most_similar_words is None:
            #     return word
            # new_most_similar_words = []
            # for itm, prob in most_similar_words:
            #     if len(set(word) & set(itm)) != 0 and len(word) == len(itm):
            #         new_most_similar_words.append((itm, prob))
            # if len(new_most_similar_words) == 0:
            #     return word
            # self.synonyms[word] = new_most_similar_words
        synonym = self._random_pick(self.synonyms[word])
        return synonym

    def _random_pick(self, word_prob):
        words, probs = zip(*word_prob)
        probs = softmax(np.array(probs)).tolist()
        x = uniform(0, 1)
        cum_prob = 0.
        result = None
        for item, prob in zip(words, probs):
            cum_prob += prob
            if x < cum_prob:
                result = item
                break
        return result

    def update_loss_weight(self, loss_weight, labels):
        if isinstance(labels, list):
            for label in labels:
                loss_weight[label] += 1
        else:
            loss_weight[labels] += 1

    def _json_to_tfrecord(self, filename_from, filenmae_to):
        """

        :param filename_from:
        :param filenmae_to:
        :param mode:
        :return:
        """
        print('start convert tfrecord for {}'.format(filename_from))
        create_dirs([self._config.data_base_path])
        # writer = tf.python_io.TFRecordWriter(filenmae_to)
        writer = tf.python_io.TFRecordWriter(filenmae_to)
        examples = []
        with codecs.open(filename_from, 'r', encoding='utf8') as inf:
            for line in tqdm(inf):
                line_json = json.loads(line)
                fact_seg = ' '.join(line_json.get('fact_seg')).split()
                fact_seg_ids = self._data.get_word_ids(fact_seg)
                meta = line_json.get('meta')
                # relevant_articles
                relevant_articles = self._data.get_label(meta, 'relevant_articles')
                self.update_loss_weight(self._loss_weight_law, relevant_articles)
                # accusation
                accusation = self._data.get_label(meta, 'accusation')
                self.update_loss_weight(self._loss_weight_accu, accusation)
                # term_of_imprisonment
                term_of_imprisonment = self._data.get_label(meta, 'term_of_imprisonment')
                self.update_loss_weight(self._loss_weight_impris, term_of_imprisonment)
                # death
                death = self._data.get_label(meta, 'death')
                self.update_loss_weight(self._loss_weight_death, death)
                # money
                punish_of_money = meta.get('punish_of_money')
                punish_of_money = 1 if punish_of_money > 0 else 0

                criminals = meta.get('criminals')
                # build example
                example = tf.train.Example(
                    features=tf.train.Features(
                        feature={
                            'fact': tf.train.Feature(int64_list=tf.train.Int64List(value=fact_seg_ids)),
                            'relevant_articles': tf.train.Feature(int64_list=tf.train.Int64List(value=relevant_articles)),
                            'accusation': tf.train.Feature(int64_list=tf.train.Int64List(value=accusation)),
                            'term_of_imprisonment': tf.train.Feature(int64_list=tf.train.Int64List(value=term_of_imprisonment)),
                            'money': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[punish_of_money])),
                            'death': tf.train.Feature(
                                int64_list=tf.train.Int64List(value=[death])),
                        }
                    )
                )
                examples.append(example.SerializeToString())
                if len(examples) > 1000000:
                    shuffle(examples)
                    print("save examples...")
                    for example in tqdm(examples):
                        writer.write(example)
                    examples = []
            if len(examples) > 0:
                shuffle(examples)
                print("save last examples...")
                for example in tqdm(examples):
                    writer.write(example)
        writer.close()

    def _get_copy_count(self, raw_loss_weight, labels):
        example_max = max(raw_loss_weight)
        copy_counts = []
        for label in labels:
            copy_counts.append(int(example_max / raw_loss_weight[label]))
        copy_count = sum(copy_counts) / len(copy_counts)
        return int(copy_count)

    def _train_rebalance(self, filename_from, filenmae_to, mode="train"):
        """

        :param filename_from:
        :param filenmae_to:
        :param mode:
        :return:
        """
        print('start convert tfrecord for {}'.format(filename_from))
        create_dirs([self._config.data_base_path])
        writer = tf.python_io.TFRecordWriter(
            os.path.join(filenmae_to, "data.{}{}.tfrecord".format(mode, ".augmented")))
        # writer_law = tf.python_io.TFRecordWriter(os.path.join(filenmae_to, "data.train{}.tfrecord".format(".balance.law")))
        # writer_accu = tf.python_io.TFRecordWriter(os.path.join(filenmae_to, "data.train{}.tfrecord".format(".balance.accu")))
        # writer_impris = tf.python_io.TFRecordWriter(os.path.join(filenmae_to, "data.train{}.tfrecord".format(".balance.impris")))
        examples = []
        with codecs.open(filename_from, 'r', encoding='utf8') as inf:
            for line in tqdm(inf):
                line_json = json.loads(line)
                fact_chars = line_json.get('fact')
                fact_seg = ' '.join(line_json.get('fact_seg')).split()
                fact_len = len("".join(fact_seg))
                if fact_len < 50:
                    continue
                word_pos = ' '.join(line_json.get('word_pos')).split()
                # fact_seg_ids = self._data.get_word_ids(fact_seg)
                meta = line_json.get('meta')
                # relevant_articles
                relevant_articles = self._data.get_label(meta, 'relevant_articles')
                # accusation
                accusation = self._data.get_label(meta, 'accusation')
                # term_of_imprisonment
                term_of_imprisonment = self._data.get_label(meta, 'term_of_imprisonment')
                # money
                punish_of_money = meta.get('punish_of_money')
                punish_of_money = 1 if punish_of_money > 0 else 0
                # death
                death = self._data.get_label(meta, 'death')
                criminals = meta.get('criminals')

                copy_count_law = self._get_copy_count(self.raw_loss_weight_law, relevant_articles)
                copy_count_accu = self._get_copy_count(self.raw_loss_weight_accu, accusation)
                copy_count_impris = self._get_copy_count(self.raw_loss_weight_impris, term_of_imprisonment)

                copy_count = math.floor(math.sqrt((copy_count_accu + copy_count_impris + copy_count_law) / 3.))
                fact_augmented = self.augment_data(fact_seg, word_pos, copy_count - 1)
                fact_augmented.append(fact_seg)
                for fact in fact_augmented:
                    fact_seg_ids = self._data.get_word_ids(fact)
                    self.update_loss_weight(self._loss_weight_law, relevant_articles)
                    self.update_loss_weight(self._loss_weight_accu, accusation)
                    self.update_loss_weight(self._loss_weight_impris, term_of_imprisonment)
                    self.update_loss_weight(self._loss_weight_death, death)
                    # build example
                    example = tf.train.Example(
                        features=tf.train.Features(
                            feature={
                                'fact': tf.train.Feature(int64_list=tf.train.Int64List(value=fact_seg_ids)),
                                'relevant_articles': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=relevant_articles)),
                                'accusation': tf.train.Feature(int64_list=tf.train.Int64List(value=accusation)),
                                'term_of_imprisonment': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=term_of_imprisonment)),
                                'money': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[punish_of_money])),
                                'death': tf.train.Feature(
                                    int64_list=tf.train.Int64List(value=[death])),
                            }
                        )
                    )
                    examples.append(example.SerializeToString())
                    # writer.write(example.SerializeToString())
                if len(examples) > 1000000:
                    shuffle(examples)
                    print("save examples...")
                    for example in tqdm(examples):
                        writer.write(example)
                    examples = []
            if len(examples) > 0:
                shuffle(examples)
                print("save last examples...")
                for example in tqdm(examples):
                    writer.write(example)
        writer.close()
        print("save synonyms")
        pickle.dump(self.synonyms, open(self._config.synonym_path, "wb"))
        #         for i in range(copy_count_law):
        #             self.update_loss_weight(self._loss_weight_law, relevant_articles)
        #             writer_law.write(example.SerializeToString())
        #         for i in range(copy_count_accu):
        #             self.update_loss_weight(self._loss_weight_accu, accusation)
        #             writer_accu.write(example.SerializeToString())
        #         for i in range(copy_count_impris):
        #             self.update_loss_weight(self._loss_weight_impris, term_of_imprisonment)
        #             writer_impris.write(example.SerializeToString())
        # writer_law.close()
        # writer_accu.close()
        # writer_impris.close()

    def make_loss_weight(self):
        law_sum = sum(self._loss_weight_law)
        accu_sum = sum(self._loss_weight_accu)
        impris_sum = sum(self._loss_weight_impris)
        # # for law
        # for i in range(len(self._loss_weight_law)):
        #     self._loss_weight_law[i] = float(law_sum) / float(max(self._loss_weight_law[i], 1))
        # # for accu
        # for i in range(len(self._loss_weight_accu)):
        #     self._loss_weight_accu[i] = float(accu_sum) / float(max(self._loss_weight_accu[i], 1))
        # # for impors
        # for i in range(len(self._loss_weight_impris)):
        #     self._loss_weight_impris[i] = float(impris_sum) / float(max(self._loss_weight_impris[i], 1))

        pickle.dump((self._loss_weight_accu, self._loss_weight_law, self._loss_weight_impris, self._loss_weight_death),
                    open(os.path.join(self._config.data_base_path,
                                      "loss_weight{}.pkl".format("_balance" if self._config.rebalance else '')),
                         "wb"))

    def process(self, modes):
        # make tfrecord
        for mode in modes:
            if mode == 'train':
                if self._config.rebalance:
                    self._train_rebalance(self._config.seg_train_path, self._config.data_base_path)
                else:
                    self._json_to_tfrecord(self._config.seg_train_path, self._config.train_data_path)
            elif mode == 'valid':
                if self._config.rebalance:
                    self._train_rebalance(self._config.seg_valid_path, self._config.data_base_path, mode=mode)
                else:
                    self._json_to_tfrecord(self._config.seg_valid_path, self._config.valid_data_path)
            else:
                if self._config.rebalance:
                    self._train_rebalance(self._config.seg_test_path, self._config.data_base_path, mode=mode)
                else:
                    self._json_to_tfrecord(self._config.seg_test_path, self._config.test_data_path)
        # make loss weight
        if "train" in modes:
            self.make_loss_weight()
