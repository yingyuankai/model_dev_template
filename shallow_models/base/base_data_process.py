class BaseDataProcess():
    def __init__(self, config):
        self._config = config

    def process(self, modes):
        raise NotImplementedError