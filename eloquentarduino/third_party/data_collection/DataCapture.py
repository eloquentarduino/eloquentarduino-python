import numpy as np
import pandas as pd


class DataCapture:
    """
    Handle captured data
    """
    def __init__(self, values, features=None):
        """
        :param values: list
        """
        self.values = np.asarray(values, dtype=float)
        self.features = features or [{
            'name': f'f{i}',
            'dtype': 'float'
        } for i in range(self.num_features)]

        self.scale_values()

    @property
    def num_values(self) -> int:
        """
        Get number of values
        """
        return self.values.shape[0]

    @property
    def num_features(self) -> int:
        """
        Get number of features
        """
        return self.values.shape[1]

    @property
    def feature_names(self) -> list:
        """

        """
        return [feature['name'] for feature in self.features]

    @property
    def rx_frequency(self) -> float:
        """
        Get average RX frequency
        """
        try:
            timestamp_idx = self.feature_names.index('timestamp')
            started_at = self.values[0, timestamp_idx]
            ended_at = self.values[-1, timestamp_idx]

            return 1000 * self.num_values / (ended_at - started_at)
        except (ValueError, ZeroDivisionError):
            return 0

    @property
    def rx_ratio(self) -> float:
        """
        Get average RX success ratio
        """
        try:
            counter_idx = self.feature_names.index('counter')
            started_at = self.values[0, counter_idx]
            ended_at = self.values[-1, counter_idx]

            return (ended_at - started_at) / self.num_values
        except (ValueError, ZeroDivisionError):
            return 0

    @property
    def df(self) -> pd.DataFrame:
        """
        Convert to DataFrame
        """
        return pd.DataFrame(self.values, columns=self.feature_names)

    def drop(self, columns):
        """
        Drop given columns
        :param columns: list
        :return: DataCapture
        """
        columns_to_keep = [i for i, name in enumerate(self.feature_names) if i not in columns and name not in columns]
        self.values = self.values[:, columns_to_keep]

        return self

    def save_to(self, filename):
        """
        Save data to file
        :param filename: str
        """
        self.df.to_csv(filename, index=None, float_format='%.6f')

    def scale_values(self):
        """
        Apply feature scaling
        """
        for i, feature in enumerate(self.features):
            scale = feature.get('scale', 0)
            offset = feature.get('offset', 0)

            if scale != 0 and scale != 1:
                self.values[:, i] /= scale

            if offset != 0:
                self.values[:, i] -= offset