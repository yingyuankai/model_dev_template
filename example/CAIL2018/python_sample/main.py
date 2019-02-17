# -*- coding:utf-8 -*-

import json

import os
import multiprocessing



from predictor import Predictor

data_path = "../../../data/raw/good/input"  # The directory of the input data
output_path = "../../../data/raw/good/output"  # The directory of the output data


def format_result(result):
    rex = {"accusation": [], "articles": [], "imprisonment": -3}

    res_acc = []
    for x in result["accusation"]:
        if x is not None:
            res_acc.append(int(x))
    rex["accusation"] = res_acc

    if result["imprisonment"] is not None:
        rex["imprisonment"] = int(result["imprisonment"])
    else:
        rex["imprisonment"] = -3

    res_art = []
    for x in result["articles"]:
        if x is not None:
            res_art.append(int(x))
    rex["articles"] = res_art

    return rex


if __name__ == "__main__":
    user = Predictor()
    cnt = 0


    def get_batch():
        v = user.batch_size
        if not (type(v) is int) or v <= 0:
            raise NotImplementedError

        return v


    def solve(fact):
        result = user.predict(fact)

        for a in range(0, len(result)):
            result[a] = format_result(result[a])

        return result


    for file_name in os.listdir(data_path):
        with open(os.path.join(data_path, file_name), "r", encoding='utf8') as inf:
            with open(os.path.join(output_path, file_name), "w") as ouf:

                fact = []

                for line in inf:
                    fact.append(json.loads(line)["fact"])
                    if len(fact) == get_batch():
                        result = solve(fact)
                        cnt += len(result)
                        for x in result:
                            print(json.dumps(x), file=ouf)
                        fact = []

                if len(fact) != 0:
                    result = solve(fact)
                    cnt += len(result)
                    for x in result:
                        print(json.dumps(x), file=ouf)
                    fact = []
