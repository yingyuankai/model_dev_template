import sys
import os
import jieba
import jieba.posseg as pseg
import numpy as np
import tensorflow as tf
from tflearn.data_utils import pad_sequences
from deep_models.base.base_predict import BasePredict

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from utils.io_utils import read_next
from utils.strutil import refine_text
from utils.math_utils import softmax


class Predictor(BasePredict):
    def __init__(self, config, gpu_config):
        super(Predictor, self).__init__(config)
        self.batch_size = self._config.batch_size
        self._data = self._config.Data(config)
        # load law vocab
        jieba.dt.tmp_dir = './jieba_tmp'
        jieba.load_userdict(self._config.law_vocab_path)
        jieba.enable_parallel(8)
        # load stopwords
        self._stopwords = self._read_stop_words()
        # load model
        flag, msg = self._load_models()
        if flag is False:
            print(msg)
            sys.exit(1)

        # extract ops
        self._extract_ops()
        # build sess
        self._sess = tf.Session(graph=self._graph, config=gpu_config)

    def _read_stop_words(self):
        """

        :return:
        """
        stopwords = set()
        for word in read_next(self._config.stopwords_path):
            stopwords.add(word)
        return stopwords

    def _load_models(self):
        try:
            with tf.gfile.FastGFile(os.path.join(self._config.model_output_path, "textcnn.pb"), "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name=self._config.op_prefix)
                self._graph = graph
        except Exception as e:
            print(e)
            return False, e
        return True, "load model success!"

    def _extract_ops(self):
        self.fact_input = self._graph.get_tensor_by_name("{}/fact_input:0".format(self._config.op_prefix))
        self.law = self._graph.get_tensor_by_name("{}/output/law/BiasAdd:0".format(self._config.op_prefix))
        self.accu = self._graph.get_tensor_by_name("{}/output/accu/BiasAdd:0".format(self._config.op_prefix))
        self.impris = self._graph.get_tensor_by_name("{}/output/impris/BiasAdd:0".format(self._config.op_prefix))
        self.keep_prob = self._graph.get_tensor_by_name("{}/keep_prob:0".format(self._config.op_prefix))
        self.is_training = self._graph.get_tensor_by_name("{}/is_training_1:0".format(self._config.op_prefix))

    def _build_batch(self, content):
        fact_ids = []
        for fact in content:
            # for online
            # fact_jiebaed = pseg.cut(fact)
            # fact_jiebaed_refined, pos = refine_text(fact_jiebaed, self._stopwords)

            # for offline
            fact_jiebaed_refined = str(' '.join(fact)).split()
            if len(fact_jiebaed_refined) == 0:
                continue
            # to ids
            fact_seg_ids = self._data.get_word_ids(fact_jiebaed_refined)
            fact_ids.append(fact_seg_ids)
        return pad_sequences(fact_ids, maxlen=self._config.max_seq_len,
                             value=self._data.vocab.word_to_id(self._config.PAD_TOKEN))

    def choice_label_by_compare(self, logits):
        """

        :param logits:
        :return:
        """
        logits_reshaped = np.reshape(logits, newshape=[logits.shape[0], 2, int(logits.shape[-1] / 2)])
        law_logits_argmax = np.argmax(logits_reshaped, axis=1)
        indexs = np.where(law_logits_argmax == 0)
        labels = dict()
        for index in zip(indexs[0], indexs[1]):
            row, col = index
            if row not in labels:
                labels[row] = []
            labels[row].append(col)
        return labels

    def choice_label_by_max(self, logits):
        logits_reshaped = np.reshape(logits, newshape=[2, int(logits.shape[-1] / 2)])
        for i in range(int(logits.shape[-1] / 2)):
            logits_reshaped[:, i] = softmax(logits_reshaped[:, i])
        logits_diff = logits_reshaped[0, :] - logits_reshaped[1, :]
        label = np.argmax(logits_diff)
        return [label]

    def predict(self, content):
        result = []
        fact_ids = self._build_batch(content)
        to_return = [self.law, self.accu, self.impris]
        law_logits, accu_logits, impris_logits = self._sess.run(to_return,
                                                                feed_dict={self.fact_input: fact_ids,
                                                                           self.keep_prob: 1.,
                                                                           self.is_training: False})
        law = self.choice_label_by_compare(law_logits)
        accu = self.choice_label_by_compare(accu_logits)

        impris = np.argmax(impris_logits, axis=1)

        for i in range(len(law_logits)):
            result_tmp = {"imprisonment": int(impris[i]),
                          "articles": [int(pre) + 1 for pre in law.get(i, self.choice_label_by_max(law_logits[i]))],
                          "accusation": [int(pre) + 1 for pre in accu.get(i, self.choice_label_by_max(accu_logits[i]))]}
            result.append(result_tmp)
        return result
