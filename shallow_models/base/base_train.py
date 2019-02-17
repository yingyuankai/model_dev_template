
class BaseTrainer:
    def __init__(self, config):
        self._config = config

    def train(self):
        raise NotImplementedError