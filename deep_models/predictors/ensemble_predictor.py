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
from utils.math_utils import softmax, sigmoid


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
        self._graph, self._sess = self._load_models(gpu_config)
        # extract ops
        self._ops = self._extract_ops()

    def _read_stop_words(self):
        """

        :return:
        """
        stopwords = set()
        for word in read_next(self._config.stopwords_path):
            stopwords.add(word)
        return stopwords

    def _load_models(self, gpu_config):
        graphs, sesses = {}, {}
        for model_name in self._config.model_names:
            for patten in ["111", "110"]:
                flag, msg, graph = self._load_model(model_name, patten)
                if flag is False:
                    continue
                graphs[model_name+patten] = graph
                sesses[model_name+patten] = tf.Session(graph=graph, config=gpu_config)
        return graphs, sesses

    def _load_model(self, model_name, patten):
        try:
            with tf.gfile.FastGFile(os.path.join(self._config.model_output_path.format(model_name), "model_{}.pb".format(patten)), "rb") as f:
                graph_def = tf.GraphDef()
                graph_def.ParseFromString(f.read())

            with tf.Graph().as_default() as graph:
                tf.import_graph_def(graph_def, name=self._config.op_prefix.format(model_name))
                model_graph = graph
        except Exception as e:
            print(e)
            return False, e, None
        return True, "load model success!", model_graph

    def _extract_ops(self):
        textcnn_111_ops = self._extract_ops_by_model_name("textcnn_111")
        textcnn_110_ops = self._extract_ops_by_model_name("textcnn_110")
        dpcnn_111_ops = self._extract_ops_by_model_name("dpcnn_111")
        dpcnn_110_ops = self._extract_ops_by_model_name("dpcnn_110")
        amtl_111_ops = self._extract_ops_by_model_name("amtl_111")
        amtl_110_ops = self._extract_ops_by_model_name("amtl_110")
        ops = {"textcnn_111": textcnn_111_ops, "textcnn_110": textcnn_110_ops,
               "dpcnn_111": dpcnn_111_ops, "dpcnn_110": dpcnn_110_ops,
               "amtl_111": amtl_111_ops, "amtl_110": amtl_110_ops}
        return ops

    def _extract_ops_by_model_name(self, model_name):
        fact_input = self._graph.get(model_name).\
            get_tensor_by_name("{}/fact_input:0".format(self._config.op_prefix.format(model_name)))
        law = self._graph.get(model_name).\
            get_tensor_by_name("{}/output_law/law/BiasAdd:0".format(self._config.op_prefix.format(model_name)))
        accu = self._graph.get(model_name).\
            get_tensor_by_name("{}/output_accu/accu/BiasAdd:0".format(self._config.op_prefix.format(model_name)))
        impris = self._graph.get(model_name).\
            get_tensor_by_name("{}/output_impris/impris/BiasAdd:0".format(self._config.op_prefix.format(model_name)))
        death = self._graph.get(model_name). \
            get_tensor_by_name("{}/output_impris/impris/BiasAdd:0".format(self._config.op_prefix.format(model_name)))
        keep_prob = self._graph.get(model_name).\
            get_tensor_by_name("{}/keep_prob:0".format(self._config.op_prefix.format(model_name)))
        is_training = self._graph.get(model_name).\
            get_tensor_by_name("{}/is_training_1:0".format(self._config.op_prefix.format(model_name)))
        variables = {"fact_input": fact_input, "law": law, "accu": accu, "impris": impris, "death": death,
                     "keep_prob": keep_prob, "is_training": is_training}
        return variables

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

    def choice_label_by_threshold(self, logits, threshold=0.5):
        logits = sigmoid(logits)
        indexs = np.where(logits > threshold)
        labels = dict()
        for index in zip(indexs[0], indexs[1]):
            row, col = index
            if row not in labels:
                labels[row] = []
            labels[row].append(col)
        return labels

    def get_logits_by_name(self, fact_ids, model_name):
        to_return = [self._ops.get(model_name).get("law"),
                     self._ops.get(model_name).get("accu"),
                     self._ops.get(model_name).get("impris"),
                     self._ops.get(model_name).get("death")]
        law_logits, accu_logits, impris_logits, death_logits = \
            self._sess.get(model_name).run(to_return,
                                           feed_dict={
                                               self._ops.get(model_name).get("fact_input"): fact_ids,
                                               self._ops.get(model_name).get("keep_prob"): 1.,
                                               self._ops.get(model_name).get("is_training"): False
                                           }
                                           )
        return law_logits, accu_logits, impris_logits, death_logits

    def get_logits(self, fact_ids):
        law_logits_textcnn_111, accu_logits_textcnn_111, impris_logits_textcnn_111, death_logits_textcnn_111 = \
            self.get_logits_by_name(fact_ids, "textcnn_111")
        law_logits_textcnn_110, accu_logits_textcnn_110, impris_logits_textcnn_110, death_logits_textcnn_110 = \
            self.get_logits_by_name(fact_ids, "textcnn_110")
        law_logits_dpcnn_111, accu_logits_dpcnn_111, impris_logits_dpcnn_111, death_logits_dpcnn_111 = \
            self.get_logits_by_name(fact_ids, "dpcnn_111")
        law_logits_dpcnn_110, accu_logits_dpcnn_110, impris_logits_dpcnn_110, death_logits_dpcnn_110 = \
            self.get_logits_by_name(fact_ids, "dpcnn_110")
        law_logits_amtl_111, accu_logits_amtl_111, impris_logits_amtl_111, death_logits_amtl_111 = \
            self.get_logits_by_name(fact_ids, "amtl_111")
        law_logits_amtl_110, accu_logits_amtl_110, impris_logits_amtl_110, death_logits_amtl_110 = \
            self.get_logits_by_name(fact_ids, "amtl_110")
        law_logits = law_logits_textcnn * self._config.model_weights.get("textcnn") + \
                     law_logits_dpcnn * self._config.model_weights.get("dpcnn")
        accu_logits = accu_logits_textcnn * self._config.model_weights.get("textcnn") + \
                      accu_logits_dpcnn * self._config.model_weights.get("dpcnn")
        impris_logits = impris_logits_textcnn * self._config.model_weights.get("textcnn") + \
                        impris_logits_dpcnn * self._config.model_weights.get("dpcnn")
        return law_logits, accu_logits, impris_logits

    def predict(self, content):
        result = []
        fact_ids = self._build_batch(content)
        law_logits, accu_logits, impris_logits = self.get_logits(fact_ids)

        law = self.choice_label_by_threshold(law_logits)
        accu = self.choice_label_by_threshold(accu_logits)
        impris = np.argmax(impris_logits, axis=1)

        for i in range(len(law_logits)):
            result_tmp = {"imprisonment": int(impris[i]),
                          "articles": [int(pre) + 1 for pre in law.get(i, [int(np.argmax(law_logits[i]))])],
                          "accusation": [int(pre) + 1 for pre in accu.get(i, [int(np.argmax(accu_logits[i]))])]}
            result.append(result_tmp)
        return result
