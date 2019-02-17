class BaseEval:
    def __init__(self, config):
        self._config = config

    def eval(self):
        raise NotImplementedError