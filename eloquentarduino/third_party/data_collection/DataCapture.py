import numpy as np
import pandas as pd


class DataCapture:
    """
    Handle captured data
    """
    def __init__(self, values):
        """
        :param values: list
        """
        self.values = np.asarray(values, dtype=float)
        self.feature_names = [f'f{i}' for i in range(self.num_features)]

    @property
    def num_features(self):
        """
        Get number of features
        :return: int
        """
        return self.values.shape[1]

    def drop(self, columns):
        """
        Drop given columns
        :param columns: list
        :return: DataCapture
        """
        columns_to_keep = [i for i in range(self.num_features) if i not in columns]
        self.values = self.values[:, columns_to_keep]

        return self

    def set_names(self, feature_names):
        """
        Set feature names
        :param feature_names: list[str]
        """
        assert len(feature_names) == self.num_features, f'you MUST supply exactly {self.num_features} names'

        self.feature_names = feature_names

        return self

    def save_to(self, filename):
        """
        Save data to file
        :param filename: str
        """
        df = pd.DataFrame(self.values, columns=self.feature_names)
        df.to_csv(filename, index=None, float_format='%.6f')