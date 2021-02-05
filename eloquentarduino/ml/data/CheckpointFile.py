import os.path
import pandas as pd


class CheckpointFile:
    """
    Save benchmark results to a file
    """
    def __init__(self, filename, keys):
        """
        :param filename:
        :param keys:
        """
        assert isinstance(keys, list) or isinstance(keys, tuple), 'keys MUST be a list or tuple'
        self.filename = filename
        self.keys = keys
        self.df = None

        if self.exists():
            self.df = pd.read_csv(filename)
            self.df.set_index(self.keys)

    def exists(self):
        """
        Test if checkpoint file exists
        """
        return self.filename is not None and os.path.isfile(self.filename)

    def key_exists(self, key):
        """
        Test if given key exists in the file
        :param key:
        :return: bool
        """
        return self.iloc(key) is not None

    def get(self, key):
        """
        Get row by key
        :param key:
        """
        return self.df.loc[self.iloc(key)]

    def set(self, key, value):
        """
        Set value for given key (overwrite or append)
        """
        if self.key_exists(key):
            self.df.update(pd.DataFrame([value]))
        elif self.df is None:
            self.df = pd.DataFrame([value])
            self.df.set_index(self.keys)
        else:
            self.df = self.df.append(value, ignore_index=True)
        self.save()

    def clear(self):
        """
        Delete (logically) file contents
        """
        self.df = None

    def iloc(self, key):
        """
        Get row index of given key, if exists
        :param key:
        :return: int or None if not found
        """
        df = self.df
        if df is None:
            return None
        if not isinstance(key, tuple):
            key = (key,)
        for key_name, key_value in zip(self.keys, key):
            df = df.loc[df[key_name] == key_value]
        if len(df) == 0:
            return None
        return df.index[0]

    def save(self):
        """
        Save DataFrame to file
        """
        self.df.to_csv(self.filename, index=False, float_format='%.4f')
