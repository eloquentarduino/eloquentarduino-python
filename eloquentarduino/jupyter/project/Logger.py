from logging import Logger, StreamHandler, Formatter, DEBUG


class ProjectLogger(Logger):
    """
    Custom logger
    """
    def __init__(self, name):
        """
        Init and configure
        :param name:
        """
        super(ProjectLogger, self).__init__(name, DEBUG)
        handler = StreamHandler()
        handler.setFormatter(Formatter('[%(levelname)s] %(message)s'))
        self.addHandler(handler)

    def progress(self, msg):
        """
        Print progress message to console
        :param msg:
        :return:
        """
        if self.isEnabledFor(DEBUG):
            print(msg, end='')
