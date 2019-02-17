#!/usr/bin/env python

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
import os
import importlib
import codecs
import tensorflow as tf
from fastText import train_supervised
from utils.dirs import create_dirs
from utils.display_utils import print_results
from shallow_models.base.base_train import BaseTrainer


class Trainer(BaseTrainer):
    def __init__(self, config):
        super(Trainer, self).__init__(config)

    def _build_train_dataset(self, task):
        """
        build train or train_valid or train_valid_test dataset
        :return:
        """
        print("build train dataset: {}".format(str(self._config.train_sources)))
        train_path = self._config.train_data_path.format(self._config.data_base_path, task,
                                                         int(self._config.multi_labels_in_one_line))
        if len(self._config.train_sources) == 1 and 'train' in self._config.train_sources:
            return train_path
        else:
            train_path = os.path.join(self._config.data_base_path, 'tmp')
            with codecs.open(train_path, 'w', encoding='utf8') as ouf:
                for train_source in self._config.train_sources:
                    path_tmp = os.path.join(self._config.data_base_path, task,
                                            "data_{}.{}".format(int(self._config.multi_labels_in_one_line),
                                                                train_source))
                    with codecs.open(path_tmp, 'r', encoding='utf8') as inf:
                        for line in inf:
                            ouf.write(line)
            return train_path


    def _train_single(self, task):
        print("start train for {}".format(task))
        model_path = os.path.join(self._config.model_output_path, task, str(int(self._config.multi_labels_in_one_line)))
        create_dirs([model_path])
        train_path = self._build_train_dataset(task)

        model = train_supervised(input=train_path,
                                 lr=self._config.lr,
                                 dim=self._config.dim,
                                 ws=self._config.ws,
                                 epoch=self._config.epoch,
                                 minCount=self._config.min_count,
                                 minCountLabel=self._config.min_count_label,
                                 minn=self._config.minn,
                                 maxn=self._config.maxn,
                                 neg=self._config.neg,
                                 wordNgrams=self._config.word_ngrams,
                                 loss=self._config.loss,
                                 bucket=self._config.bucket,
                                 thread=self._config.thread,
                                 lrUpdateRate=self._config.lr_update_rate,
                                 t=self._config.t,
                                 label=self._config.label,
                                 verbose=self._config.verbose,
                                 pretrainedVectors=self._config.pretrained_vectors
                                 )

        model.save_model(os.path.join(model_path, "fasttext.bin"))

        if self._config.is_save_ftz:
            model.quantize(input=train_path,
                           qout=self._config.qout,
                           cutoff=self._config.cutoff,
                           retrain=self._config.retrain,
                           epoch=self._config.epoch,
                           lr=self._config.lr,
                           thread=self._config.thread,
                           verbose=self._config.verbose,
                           dsub=self._config.dsub,
                           qnorm=self._config.qnorm
                           )
            model.save_model(os.path.join(model_path, "fasttext.ftz"))
        os.system('rm {}'.format(train_path))

    def train(self):
        for task in self._config.tasks:
            self._train_single(task)
