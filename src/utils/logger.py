import logging
import time
import os.path


class Logger(object):
    def __init__(self, path):
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(levelname)s: %(message)s')
        self.logger = logging.getLogger()
        self.logger.setLevel(logging.INFO)

        rq = time.strftime('%Y-%m-%d-%H-%M', time.localtime(time.time()))
        self.log_path = os.path.join(path, f"{rq}.log")
        fh = logging.FileHandler(self.log_path, mode='w')
        fh.setLevel(logging.INFO)

        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)

        formatter = logging.Formatter("%(asctime)s - %(levelname)s: %(message)s")
        fh.setFormatter(formatter)

        self.logger.addHandler(fh)

