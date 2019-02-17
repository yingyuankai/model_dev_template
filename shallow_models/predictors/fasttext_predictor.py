import sys
import os
import jieba
from fastText import load_model
from shallow_models.base.base_predict import BasePredict
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.io_utils import read_next
from utils.strutil import refine_text


class Predictor(BasePredict):
    def __init__(self, config):
        super(Predictor, self).__init__(config)
        self.batch_size = 1
        # load law vocab
        jieba.dt.tmp_dir = './tmp'
        jieba.load_userdict(self._config.law_vocab_path)
        # load stopwords
        self._stopwords = self._read_stop_words()
        self._models = self._load_models()
    
    def _read_stop_words(self):
        """

        :return:
        """
        stopwords = set()
        for word in read_next(self._config.stopwords_path):
            stopwords.add(word)
        return stopwords

    def _load_single_model(self, task):
        """

        :param task:
        :return:
        """
        model_path = os.path.join(self._config.model_output_path, task, str(int(self._config.multi_labels_in_one_line)))
        print("Eval for {}".format(model_path))
        model = load_model(os.path.join(model_path, "fasttext.{}".format(self._config.use_bin_or_ftz)))
        return model

    def _load_models(self):
        models = {}
        for task in self._config.tasks:
            if task not in models:
                models[task] = self._load_single_model(task)
        return models
        
    def predict(self, content):
        result = []
        for fact in content:
            # fact_jiebaed = jieba.cut(fact)
            # fact_jiebaed_refined = refine_text(fact_jiebaed, self._stopwords)
            fact_jiebaed_refined = str(' '.join(fact)).split()
            if len(fact_jiebaed_refined) == 0:
                continue
            # result_tmp = {'term_of_imprisonment': -3}
            result_tmp = {}
            for task in self._config.tasks:
                pred_labels, pred_prob = self._models[task].predict(" ".join(fact_jiebaed_refined), self._config.return_n)
                pred_labels = list(map(lambda s: s.replace(self._config.label, ""), pred_labels))
                if task not in result_tmp:
                    if task == 'term_of_imprisonment':
                        result_tmp[task] = int(pred_labels[0])
                    else:
                        result_tmp[task] = [int(pre_label) + 1 for pre_label in pred_labels]

            result.append(result_tmp)
        return result
