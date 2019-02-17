import os
import sys
import json
import numpy as np
import pickle
from fastText import load_model, util

class Data():
    def __init__(self, config):
        self._config = config
        self.law2id, self.accu2id, self.id2law, self.id2accu = self._load()
        self.imprisonment2id, self.id2imprisonment = self._build_imprisonment_vocab()
        self.vocab = Vocab(config)
        # self.pretrain_embedding = self._load_pretrain_embedding()

    def _load(self):
        f = open(self._config.raw_law_path, 'r', encoding='utf8')
        law = {}
        lawname = {}
        line = f.readline()
        while line:
            lawname[len(law)] = line.strip()
            law[line.strip()] = len(law)
            line = f.readline()
        f.close()

        f = open(self._config.raw_accu_path, 'r', encoding='utf8')
        accu = {}
        accuname = {}
        line = f.readline()
        while line:
            accuname[len(accu)] = line.strip()
            accu[line.strip()] = len(accu)
            line = f.readline()
        f.close()

        return law, accu, lawname, accuname

    def _build_imprisonment_vocab(self):
        imprisonment2id, id2imprisonment= {}, {}
        for i in range(0, 301):
            imprisonment2id[str(i)] = i
        imprisonment2id['-1'] = len(imprisonment2id)
        imprisonment2id['-2'] = len(imprisonment2id)
        id2imprisonment = {id: imprisonment for imprisonment, id in imprisonment2id.items()}
        return imprisonment2id, id2imprisonment

    def load_pretrain_embedding(self):
        """
        load pretrain embedding from fasttext model and
        add <un_tkn>, <pad> embeddings
        :return:
        """
        model = load_model(self._config.pretrain_embedding)
        output_word_size = len(model.get_output_matrix())
        if self.vocab.num_ids() == output_word_size + 2:
            pretrain_embedding = model.get_output_matrix()
        else:
            pretrain_embedding = model.get_input_matrix()

        pretrain_embedding = np.row_stack((pretrain_embedding, np.zeros(self._config.embed_size)))
        pretrain_embedding = np.row_stack((pretrain_embedding, np.zeros(self._config.embed_size)))
        return pretrain_embedding

    def get_class_num(self, kind):
        if kind == 'law':
            return len(self.law2id)
        if kind == 'accu':
            return len(self.accu2id)

    def get_name(self, index, kind):
        if kind == 'law':
            return self.id2law[index]

        if kind == 'accu':
            return self.id2accu[index]

        if kind == 'impri':
            return self.id2imprisonment[index]

    def get_time(self, time):
        # 将刑期用分类模型来做
        v = int(time['imprisonment'])

        if time['death_penalty']:
            return str(-2)
            # return 0
        if time['life_imprisonment']:
            return str(-1)
            # return 1
        else:
            return str(v)

    def get_death(self, time):
        if time['death_penalty']:
            return 2
        if time['life_imprisonment']:
            return 1
        else:
            return 0

    def get_label(self, d, kind):
        if kind == 'relevant_articles':
            return [self.law2id[str(law_itm)] for law_itm in d['relevant_articles']]
        if kind == 'accusation':
            return [self.accu2id[str(acc_itm)] for acc_itm in d['accusation']]
        if kind == 'term_of_imprisonment':
            return [self.imprisonment2id.get(self.get_time(d['term_of_imprisonment']))]
        if kind == 'death':
            return self.get_death(d['term_of_imprisonment'])

        # return label

    def get_word_ids(self, words):
        """

        :param words:
        :return:
        """
        if isinstance(words, list):
            return [self.vocab.word_to_id(word) for word in words]
        elif isinstance(words, str):
            return [self.vocab.word_to_id(word) for word in words.split()]
        else:
            raise Exception('words type error, place input list or str.')

class Vocab():
    """Vocabulary class for mapping words and ids."""

    def __init__(self, config):
        self._config = config
        self._word_to_id = {}
        self._id_to_word = {}
        self._count = 0
        self._load_or_build_vocab()

    def _load_or_build_vocab(self):
        if not os.path.isfile(self._config.words_vocab_path):
            model = load_model(self._config.pretrain_embedding)
            words = model.get_words()
            self._word_to_id = {word: model.get_word_id(word) for word in words}
            self._word_to_id[self._config.UNKNOWN_TOKEN] = len(self._word_to_id)
            self._word_to_id[self._config.PAD_TOKEN] = len(self._word_to_id)
            self._id_to_word = {id: word for word, id in self._word_to_id.items()}
            self._count = len(self._id_to_word)
            pickle.dump(self._word_to_id, open(self._config.words_vocab_path, "wb"))
        else:
            self._word_to_id = pickle.load(open(self._config.words_vocab_path, "rb"))
            self._id_to_word = {id: word for word, id in self._word_to_id.items()}
            self._count = len(self._id_to_word)

    def check_vocab(self, word):
        if word not in self._word_to_id:
            return None
        return self._word_to_id[word]

    def word_to_id(self, word):
        if word not in self._word_to_id:
            return self._word_to_id[self._config.UNKNOWN_TOKEN]
        return self._word_to_id[word]

    def id_to_word(self, word_id):
        if word_id not in self._id_to_word:
            raise ValueError('id not found in vocab: %d.' % word_id)
        return self._id_to_word[word_id]

    def num_ids(self):
        return self._count