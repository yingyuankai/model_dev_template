import tensorflow as tf
from tensorflow.contrib import slim


class BaseFeature:
    def __init__(self, config):
        self._config = config

    def feature(self, *args, **kwargs):
        raise NotImplementedError
