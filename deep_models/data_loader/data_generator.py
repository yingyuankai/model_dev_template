import numpy as np
import json
import tensorflow as tf
from utils.data_new import Data

class DataGenerator:
    def __init__(self, mode, num_epochs, batch_size, config):
        self._config = config
        self._data = Data(config)
        self.fact_batch, self.relevant_articles_batch, self.accusation_batch, self.term_of_imprisonment_batch, self.money_batch, self.death_batch = \
            self.input_pipeline(mode, num_epochs, batch_size)

    def _build_dataset(self, mode):
        # to be continue
        def _parse_function(example_proto):
            """

            :param line:
            :return:
            """
            features = {
                'fact': tf.VarLenFeature(tf.int64),
                'relevant_articles': tf.VarLenFeature(tf.int64),
                'accusation': tf.VarLenFeature(tf.int64),
                'term_of_imprisonment': tf.VarLenFeature(tf.int64)
            }
            parsed_features = tf.parse_single_example(example_proto, features=features)
            return parsed_features

            # return parsed_features['fact'], parsed_features['relevant_articles'], \
            #        parsed_features['accusation'], parsed_features['term_of_imprisonment']

        filenames = self._choice_filenames(mode)
        dataset = tf.data.TFRecordDataset(filenames)
        dataset = dataset.map(_parse_function)
        dataset = dataset.shuffle(buffer_size=100)
        dataset = dataset.batch(self._config.batch_size)

        iterator = dataset.make_initializable_iterator()
        iterator_initializer = iterator.initializer
        next_batch = iterator.get_next()

        return iterator_initializer, next_batch

    def read_and_decode(self, filename_queue):
        """
        read and decode
        :param filename_queue:
        :return:
        """
        features = {
            'fact': tf.VarLenFeature(tf.int64),
            'relevant_articles': tf.VarLenFeature(tf.int64),
            'accusation': tf.VarLenFeature(tf.int64),
            'term_of_imprisonment': tf.VarLenFeature(tf.int64),
            'money': tf.VarLenFeature(tf.int64),
            'death': tf.VarLenFeature(tf.int64)
        }
        # read and parse data
        reader = tf.TFRecordReader()
        _, serialized_example = reader.read(filename_queue)
        parsed_features = tf.parse_single_example(serialized_example, features=features)
        return parsed_features['fact'], parsed_features['relevant_articles'], \
               parsed_features['accusation'], parsed_features['term_of_imprisonment'], \
               parsed_features['money'], parsed_features['death']

    def input_pipeline(self, mode, num_epochs, batch_size, min_after_dequeue=10000):
        """
        input pipeline
        :param filenames:
        :param batch_size:
        :param num_epochs:
        :param min_after_dequeue:

        define how big a buffer we will randomly sample from,
        bigger means better shuffling but slower start up and memory used.

        :return:
        """
        filenames = self._choice_filenames(mode)
        tf.logging.warn("fetch data from {}".format(" ".join(filenames)))
        # determine the maximum we will prefetch
        capacity = min_after_dequeue + 3 * batch_size
        # queue of input files
        filename_queue = tf.train.string_input_producer(filenames, num_epochs=num_epochs, shuffle=True)
        fact, relevant_articles, accusation, term_of_imprisonment, money, death = self.read_and_decode(filename_queue)
        fact_batch, relevant_articles_batch, accusation_batch, term_of_imprisonment_batch, money_batch, death_batch = \
            tf.train.shuffle_batch([fact, relevant_articles, accusation, term_of_imprisonment, money, death],
                                   batch_size=batch_size,
                                   capacity=capacity, min_after_dequeue=min_after_dequeue)

        fact_batch = \
            tf.sparse_tensor_to_dense(fact_batch,
                                      default_value=self._config.data_obj.vocab.word_to_id(self._config.PAD_TOKEN))
        relevant_articles_batch = tf.sparse_tensor_to_dense(relevant_articles_batch, default_value=-1)
        accusation_batch = tf.sparse_tensor_to_dense(accusation_batch, default_value=-1)
        term_of_imprisonment_batch = tf.squeeze(tf.sparse_tensor_to_dense(term_of_imprisonment_batch))
        money_batch = tf.squeeze(tf.sparse_tensor_to_dense(money_batch))
        death_batch = tf.squeeze(tf.sparse_tensor_to_dense(death_batch))

        return fact_batch, relevant_articles_batch, accusation_batch, term_of_imprisonment_batch, money_batch, death_batch

    def _choice_filenames(self, mode):
        """
        choice filenames according mode and train_sources
        :param mode:
        :return:
        """
        if mode == 'train':
            train_sources = self._config.train_sources
            filenames = [getattr(self._config, "{}_data_aug_path".format(train_source) if train_source != "train" else "{}_data_path".format(train_source))
                         for train_source in train_sources]
        else:
            filenames = [getattr(self._config, "{}_data_path".format(mode))]

        return filenames

    def next_batch(self, sess):
        return sess.run([self.fact_batch, self.relevant_articles_batch,
                         self.accusation_batch, self.term_of_imprisonment_batch, self.money_batch, self.death_batch])
