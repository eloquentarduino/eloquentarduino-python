import numpy as np
from time import time
import webbrowser
from eloquentarduino.third_party.label_studio.LabelStudioBottle import LabelStudioBottle


class LabelStudio:
    """
    Interactive data labelling
    """
    def __init__(self, dataset, data_freq=1, autosave=None):
        """
        :param dataset: Dataset
        :param data_freq: int frequency of data, used for plotting
        :param autosave: str if set, data will be saved at each export
        """
        self.dataset = dataset
        self.df = dataset.df
        self.label_names = dataset.class_labels
        self.data_freq = data_freq
        self.autosave = autosave

    @property
    def csv_data(self):
        """
        Convert dataset to properly-formatted structure
        """
        labels = self.df['y']
        i = 0
        data = []
        base_time = time()

        for c in self.df.columns:
            if c == 'y':
                continue

            for j, x in enumerate(self.df[c]):
                data.append({
                    'id': i,
                    'label': self.label_names[int(labels[j])],
                    'series': str(c),
                    'time': None,
                    'val': x,
                    'x': int((base_time + j) * 1000 / self.data_freq),
                    'y': x
                })
                i += 1

        return data

    def listen(self, server_port=13579, labeler_url='https://eloquentarduino.com/pg/trainset/'):
        """
        Start HTTP server for interactive studio
        """
        dataset = {
            'filename': self.dataset.name,
            'csvData': self.csv_data,
            'labelList': self.label_names,
            'seriesList': [c for c in self.df.columns if c != 'y'],
            'headerStr': 'series,timestamp,value,label'
        }

        try:
            if labeler_url:
                labeler_url = f'{labeler_url}?port={server_port}'
                print(f'Opening labeler webpage at {labeler_url}')
                webbrowser.open(labeler_url)

            httpd = LabelStudioBottle(studio=self, dataset=dataset)
            httpd.run(host='localhost', port=server_port)
        except KeyboardInterrupt:
            pass

    def update_dataset(self, payload):
        """
        Update dataset from TRAINSET
        """
        assert 'data' in payload, 'payload MUST have a `data` attribute'
        assert 'labels' in payload, 'payload MUST have a `labels` attribute'

        data = payload['data']
        labels = payload['labels']

        assert isinstance(data, dict), 'data MUST be a dict'
        assert isinstance(labels, list), 'labels MUST be a list'

        inverse_classmap = {v: k for k, v in self.dataset.classmap.items()}
        label_mask = np.asarray([l != '' for l in labels], dtype=bool)

        X = np.zeros((label_mask.sum(), len(data.keys())), dtype=float)
        y = [inverse_classmap[l] for l in labels if l]
        columns = []

        for i, (column, values) in enumerate(data.items()):
            X[:, i] = np.asarray(values)[label_mask]
            columns.append(column)

        self.dataset = self.dataset.replace(
            X=X,
            y=np.asarray(y, dtype=int),
            columns=columns,
            name='%s (labelled)' % self.dataset.name
        )

        if self.autosave:
            self.dataset.to_json(self.autosave)