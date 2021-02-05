class Classifier:
    """
    Abstraction of a classifier
    """
    def __init__(self, name, generator):
        """
        :param name:
        :param generator: function that returns a classifier. Accepts X, y
        """
        self.name = name
        self.generator = generator
