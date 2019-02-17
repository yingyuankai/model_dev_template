import os
import sys
from tqdm import tqdm
import jieba
import json
from random import random
import codecs
from shallow_models.base.base_data_process import BaseDataProcess
from utils.dirs import create_dirs
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.strutil import is_num, refine_text, splitSentence
from utils.io_utils import read_next
from utils.dirs import create_dirs


class DataProcessor(BaseDataProcess):
    def __init__(self, config):
        super(DataProcessor, self).__init__(config)
        # load law vocab
        jieba.dt.tmp_dir = './jieba_tmp'
        jieba.load_userdict(self._config.law_vocab_path)
        # load stopwords
        self._stopwords = self._read_stop_words()

    def _read_stop_words(self):
        """

        :return:
        """
        stopwords = set()
        for word in read_next(self._config.stopwords_path):
            stopwords.add(word)
        return stopwords

    def _json_to_fasttext(self, filename_from, filenmae_to):
        """
        convert raw data that has json format 
        to new format as input of fasttext
        :param filename_from: 
        :param filenmae_to: 
        :return: 
        """
        print('start cut raw data from {}'.format(filename_from))
        with codecs.open(filename_from, "r", encoding='utf8') as inf:
            with codecs.open(filenmae_to, 'w') as ouf:
                for line in tqdm(inf):
                    if line is None:
                        continue
                    line = line.strip()
                    line_json = json.loads(line)
                    new_line = self._seg_text(line_json)
                    ouf.write(new_line)

    def _seg_text(self, line_json):
        """
        
        :param line_json: 
        :param taks: 
        :return: 
        """
        fact = line_json.get("fact")
        sen_list = splitSentence(fact, object_end=True)
        facts = []
        for sen in sen_list:
            fact_jiebaed = jieba.cut(sen)
            fact_jiebaed_refined = refine_text(fact_jiebaed, self._stopwords)
            if len(fact_jiebaed_refined) == 0:
                continue
            facts.append(' '.join(fact_jiebaed_refined))
        line_json["fact_seg"] = facts

        accusations = line_json.get('meta').get('accusation')
        acc_seg = []
        for acc in accusations:
            acc_jiebaed = jieba.cut(acc)
            acc_jiebaed_refined = refine_text(acc_jiebaed, self._stopwords)
            if len(acc_jiebaed_refined) == 0:
                continue
            acc_seg.append(' '.join(acc_jiebaed_refined))
        line_json['meta']['accusation_seg'] = acc_seg

        return json.dumps(line_json, ensure_ascii=False) + '\n'

    def seg_and_split_bigdata(self, filename_from):
        train_file_obj = codecs.open(self._config.seg_train_path, "w", encoding='utf8')
        valid_file_obj = codecs.open(self._config.seg_valid_path, "w", encoding='utf8')
        test_file_obj = codecs.open(self._config.seg_test_path, "w", encoding='utf8')
        with codecs.open(filename_from, "r", encoding='utf8') as inf:
            for line in tqdm(inf):
                if line is None:
                    continue
                line = line.strip()
                line_json = json.loads(line)
                new_line = self._seg_text(line_json)
                rand_n = random()
                if rand_n <= 0.9:
                    train_file_obj.write(new_line)
                elif rand_n <= 0.95:
                    valid_file_obj.write(new_line)
                else:
                    test_file_obj.write(new_line)

        train_file_obj.close()
        valid_file_obj.close()
        test_file_obj.close()


    def process(self, modes):
        """
        modes: valid or test or train or split
        :param modes:
        :return:
        """
        for mode in modes:
            # for small data
            if mode == 'valid':
                self._json_to_fasttext(self._config.raw_valid_path, self._config.seg_valid_path)
            elif mode == 'test':
                self._json_to_fasttext(self._config.raw_test_path, self._config.seg_test_path)
            elif mode == 'train':
                self._json_to_fasttext(self._config.raw_train_path, self._config.seg_train_path)
            else:  # for big data
                self.seg_and_split_bigdata(self._config.big_raw_path)

