#!/usr/bin/python
# -*- coding:utf-8 -*-

import os
import sys
import time
import json
import types
import logging
import traceback
import tornado.web
import tornado.httpserver
import tornado.ioloop
from tornado.escape import json_encode
from utils import data
import tensorflow as tf

FLAGS = tf.app.flags.FLAGS


def get_raw_label(results):
    for result in results:
        result['relevant_articles'] = [data.getName(law -1, 'law') for law in result['relevant_articles']]
        result['accusation'] = [data.getName(accu - 1, 'accu') for accu in result['accusation']]
    return results


class JsonResponse(object):
    @staticmethod
    def error(message, time):
        """错误返回

        :param message: 错误信息
        :param time: 耗时
        :return: {"status_msg":, "status_code":, "time":, "datas": null}
        """
        return json_encode({"status_msg": message, "status_code": -1, "time": time, "datas": None})

    @staticmethod
    def success(result, time):
        """正确返回

        :param result: 正确返回结果
        :param time: 耗时
        :return: {"status_msg":, "status_code":, "time":, "datas": null}
        """
        return json_encode({"status_msg": "ok", "status_code": 0, "time": time, "datas": result})


class Handler(tornado.web.RequestHandler):

    def initialize(self, law_predict_handler):
        self._law_predict_handler = law_predict_handler

    def get(self):
        start_time = time.time()
        try:
            if self.request.method == "GET":
                data = self.get_argument("data", [])
                data = eval(data)
                result = self._law_predict_handler.predict(data)
                print(result)
                if result is None:
                    res_str = JsonResponse.error(result, time.time() - start_time)
                else:
                    result = get_raw_label(result)
                    res_str = JsonResponse.success(result, time.time() - start_time)
            else:
                res_str = JsonResponse.error("请使用POST请求", time.time() - start_time)
        except Exception as e:
            logging.error(traceback.format_exc())
            res_str = JsonResponse.error(e, time.time() - start_time)
        self.write(res_str)

    def post(self):
        """ post 请求

        :return: JsonResponse
        """
        self.set_header("Content-Type", "application/json")
        start_time = time.time()
        try:
            if self.request.method == "POST":
                data = self.get_argument("data", default=[])
                result = self._law_predict_handler.predict(data)
                if result is None:
                    res_str = JsonResponse.error(result, time.time() - start_time)
                else:
                    result = get_raw_label(result)
                    res_str = JsonResponse.success(result, time.time() - start_time)
            else:
                res_str = JsonResponse.error("请使用POST请求", time.time() - start_time)
        except Exception as e:
            logging.error(traceback.format_exc())
            res_str = JsonResponse.error(e, time.time() - start_time)
        self.write(res_str)


class LawPredictServer(object):
    def __init__(self, port, law_predict_handler):
        self.port = port
        self._law_predict_handler = law_predict_handler

    def load(self):
        return tornado.web.Application([(r"/law_predict/", Handler,
                                         dict(law_predict_handler=self._law_predict_handler))])

    def process(self):
        app = self.load()
        app.listen(self.port)
        tornado.ioloop.IOLoop.current().start()


# if __name__ == "__main__":
#     conf = Mysetting()
#     bow2seq_index_predict = Bow2seqHandler(conf)
#     flag, msg = bow2seq_index_predict.load()
#     if flag is False:
#         sys.exit(1)
#     obj = IndexPredictServer(conf.index_predict_port, bow2seq_index_predict)
#     obj.process()
