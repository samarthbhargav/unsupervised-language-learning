import time


class TicToc:

    def __init__(self):
        self.start_time = time.time()

    def tic(self):
        print("Time elapsed: {} seconds".format(round(time.time() - self.start_time, 2)))
