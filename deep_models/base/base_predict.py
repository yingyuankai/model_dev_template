class BasePredict:
    def __init__(self, config):
        self._config = config

    def predict(self, content):
        raise NotImplementedError