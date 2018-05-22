

class SentEvalApi:

    def __init__(self, name):
        self.name = name

    def get_name(self):
        return self.name

    def prepare(params, samples):
        raise NotImplementedError()

    def batcher(self, params, batch):
        raise NotImplementedError()
