import os
import sys
from tqdm import tqdm
import json
import codecs
from shallow_models.base.base_data_process import BaseDataProcess
from utils.dirs import create_dirs
from utils.strutil import is_num, refine_text
from utils.io_utils import read_next
from utils import data


class DataProcessor(BaseDataProcess):
    def __init__(self, config):
        super(DataProcessor, self).__init__(config)
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

    def _json_to_fasttext(self, filename_from, filenmae_to, mode='w'):
        """
        convert raw data that has json format 
        to new format as input of fasttext
        :param filename_from: 
        :param filenmae_to: 
        :return: 
        """
        print('start convert for {}'.format(filename_from))
        output_file_objs = {}
        with codecs.open(filename_from, "r", encoding='utf8') as inf:
            for line in tqdm(inf):
                if line is None:
                    continue
                line = line.strip()
                line_json = json.loads(line)
                for task in self._config.tasks:
                    # build file obj
                    task_data_path = os.path.join(self._config.data_base_path, task)
                    create_dirs([task_data_path])
                    if task not in output_file_objs:
                        output_file_objs[task] = codecs.open(filenmae_to.format(
                            self._config.data_base_path, task, int(self._config.multi_labels_in_one_line)),
                            mode, encoding='utf8')

                    # convert format
                    new_lines = self._build_lines_for_task(line_json, task)
                    for new_line in new_lines:
                        output_file_objs[task].write(new_line)

        for task in self._config.tasks:
            output_file_objs[task].close()

    def _build_lines_for_task(self, line_json, task):
        """
        
        :param line_json: 
        :param taks: 
        :return: 
        """
        # fact = line_json.get("fact")
        # fact_jiebaed = jieba.cut(fact)
        # fact_jiebaed_refined = refine_text(fact_jiebaed, self._stopwords)
        fact_jiebaed_refined = ' '.join(line_json.get('fact_seg'))
        if len(fact_jiebaed_refined) == 0:
            return
        # labels = line_json.get('meta').get(task)
        labels = data.getlabel(line_json.get('meta'), task)
        if self._config.multi_labels_in_one_line:
            labels_ = map(lambda x: '__label__{}'.format(x), labels)
            new_line = '{} {}\n'.format(' '.join(list(labels_)), fact_jiebaed_refined)
            new_lines = [new_line]
        else:
            new_lines = list(map(lambda x: '__label__{} {}\n'.format(x, fact_jiebaed_refined), labels))

        return new_lines

    def process(self, modes):
        for mode in modes:
            if mode == 'train':
                self._json_to_fasttext(self._config.seg_train_path, self._config.train_data_path)
            elif mode == 'valid':
                self._json_to_fasttext(self._config.seg_valid_path, self._config.valid_data_path)
            else:
                self._json_to_fasttext(self._config.seg_test_path, self._config.test_data_path)