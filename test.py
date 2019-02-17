# -*- coding:utf-8 -*-

import json
import os
from tqdm import tqdm
from utils.dirs import create_dirs
from utils.judger import Judger


class Testor():
    def __init__(self, config, gpu_config):
        self._config = config
        self._user = self._config.Predictor(config, gpu_config)
        self._judger = Judger(self._config.raw_accu_path, self._config.raw_law_path)
        self._output_path = os.path.join(self._config.data_base_path, "test_output")
        self._output_filename = os.path.join(self._output_path, self._config.seg_test_path.split('/')[-1])

    def format_result(self, result):
        rex = {"accusation": [], "articles": [], "imprisonment": -3}

        res_acc = []
        for x in result["accusation"]:
            if not (x is None):
                res_acc.append(int(x))
        rex["accusation"] = res_acc

        if not (result["imprisonment"] is None):
            rex["imprisonment"] = int(result["imprisonment"])
        else:
            rex["imprisonment"] = -3

        res_art = []
        for x in result["articles"]:
            if not (x is None):
                res_art.append(int(x))
        rex["articles"] = res_art

        return rex

    def _get_batch(self):
        v = self._user.batch_size
        if not (type(v) is int) or v <= 0:
            raise NotImplementedError

        return v

    def _solve(self, fact):
        result = self._user.predict(fact)

        for a in range(0, len(result)):
            result[a] = self.format_result(result[a])

        return result

    def _gen_test_result(self):
        """

        :return:
        """
        cnt = 0
        with open(self._config.seg_test_path, "r", encoding='utf8') as inf:
            with open(self._output_filename, "w") as ouf:
                fact = []
                for line in tqdm(inf):
                    fact.append(json.loads(line)["fact_seg"])
                    if len(fact) == self._get_batch():
                        result = self._solve(fact)
                        cnt += len(result)
                        for x in result:
                            print(json.dumps(x), file=ouf)
                        fact = []

                if len(fact) != 0:
                    result = self._solve(fact)
                    cnt += len(result)
                    for x in result:
                        print(json.dumps(x), file=ouf)
                    fact = []

    def test(self):
        create_dirs([self._output_path])
        self._gen_test_result()
        result = self._judger.test(self._config.seg_test_path, self._output_filename)
        score = self._judger.get_score(result)
        return score
