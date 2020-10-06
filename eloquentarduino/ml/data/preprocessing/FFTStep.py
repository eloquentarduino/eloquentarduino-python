import numpy as np
from eloquentarduino.ml.data.preprocessing.BaseStep import BaseStep


class FFTStep(BaseStep):
    """Apply FFT"""
    def __init__(self, X, y, featurewise=False):
        BaseStep.__init__(self, X, y)
        self.featurewise = max(1, int(featurewise))
        input_dim = self.input_dim // self.featurewise
        assert (input_dim & (input_dim - 1)) == 0, 'Number of features MUST be a power of 2 (%d given)' % self.input_dim

    @property
    def name(self):
        return 'fft'

    @property
    def inplace(self):
        return True

    def __str__(self):
        return self.describe(('featurewise', self.featurewise))

    def transform(self):
        # arduinoFFT produces one element less than Numpy (¯\_(ツ)_/¯)
        if self.featurewise < 2:
            return self.apply(np.abs(np.fft.rfft(self.X))[:, :-1])
        ffts = [np.abs(np.fft.rfft(self.X[:, i::self.featurewise]))[:, :-1] for i in range(self.featurewise)]
        fft_length = len(ffts[0][0])
        # "interleave" each FFT so that the result is (for each sample):
        # fft_feature0_idx0, fft_feature1_idx0, ..., fft_featureN_idx0, fft_feature0_idx1, ...
        # this is because `normalize` and `standardize` assume this structure
        return self.apply(np.dstack(tuple(ffts)).reshape((-1, self.featurewise * fft_length)))

    def port(self):
        return self.jinja('FFT.jinja', {
            'featurewise': self.featurewise,
            'fft_length': self.input_dim // self.featurewise
        })
