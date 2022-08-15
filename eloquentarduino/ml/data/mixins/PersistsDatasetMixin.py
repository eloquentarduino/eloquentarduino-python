import json


class PersistsDatasetMixin:
    """
    Export/Import dataset
    """
    @classmethod
    def from_json(cls, path):
        """
        Import dataset from file
        """
        filename = f'{path}.json'

        with open(filename, encoding='utf-8') as file:
            data = json.load(file)
            data['classmap'] = {int(k): v for k, v in data['classmap'].items()}

            return cls(**data, test_validity=False)

    def to_json(self, path):
        """
        Export dataset to file
        :param filename: str
        """
        filename = f'{path}.json'
        data = {
            'name': self.name,
            'X': self.X.tolist(),
            'y': self.y.tolist(),
            'columns': self.columns,
            'classmap': self.classmap
        }

        with open(filename, 'w', encoding='utf-8') as file:
            json.dump(data, file)