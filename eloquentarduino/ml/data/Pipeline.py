from sklearn.model_selection import train_test_split

from eloquentarduino.ml.data.preprocessing import FFTStep
from eloquentarduino.ml.data.preprocessing import NormalizeStep
from eloquentarduino.ml.data.preprocessing import PolynomialFeaturesStep
from eloquentarduino.ml.data.preprocessing import PrincipalFFT
from eloquentarduino.ml.data.preprocessing import RfeStep
from eloquentarduino.ml.data.preprocessing import SelectKBestStep
from eloquentarduino.ml.data.preprocessing import StandardizeStep
from eloquentarduino.ml.data.preprocessing import UnitStep
from eloquentarduino.utils import jinja
from sklearn.base import clone


class Pipeline:
    """Define a pre-processing pipeline that can be ported to plain C"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.steps = []
        self.steps_run = []
        self.includes = []

    @property
    def Xt(self):
        """Get transformed input"""
        return self.transform(self.X.copy())

    @property
    def input_dim(self):
        """Get input number of features"""
        return self.X.shape[1]

    @property
    def output_dim(self):
        """Get output number of features"""
        return self.Xt.shape[1]

    @property
    def working_dim(self):
        """Get auxiliary arrays size for C porting"""
        return max([step.input_shape[1] for step in self.steps_run] + [step.output_shape[1] for step in self.steps_run])

    @property
    def inplace(self):
        """Get wether some step will alter the input while working"""
        return sum([step.inplace for step in self.steps_run])

    def transform(self, X):
        """Apply pipeline to data"""
        self.steps_run = []
        for Step, kwargs in self.steps:
            step = Step(X, self.y, **kwargs)
            X = step.transform()
            self.steps_run.append(step)
        return X

    def port(self):
        """Port to plain C"""
        if len(self.steps_run) == 0:
            self.Xt
        env = {
            'steps': self.steps_run,
            'includes': self.includes,
            'input_dim': self.input_dim,
            'working_dim': self.working_dim,
            'needs_auxiliary_arrays': self.working_dim != self.input_dim or self.inplace
        }
        return jinja("Pipeline/Pipeline.jinja", env, pretty=True)

    def score(self, clf, **kwargs):
        """Score classifier on the transformed input"""
        X_train, X_test, y_train, y_test = train_test_split(self.Xt, self.y, **kwargs)
        return clone(clf).fit(X_train, y_train).score(X_test, y_test)

    def explain(self):
        """Return a human understandable representation of the pipeline"""
        return 'Pipeline description:\n' + ''.join(['\n - ' + str(step) for step in self.steps_run])[1:]

    def queue(self, Step, **kwargs):
        """Add step to queue"""
        self.steps.append((Step, kwargs))

    def normalize(self, featurewise=False):
        """Apply normalization"""
        self.queue(NormalizeStep, featurewise=featurewise)

    def standardize(self, featurewise=False):
        """Apply standardization"""
        self.queue(StandardizeStep, featurewise=featurewise)

    def unit(self):
        """Apply vector unitary length"""
        self.queue(UnitStep)

    def polynomial_features(self, interaction_only=False):
        """Apply 2° order polynomial features expansion"""
        return self.queue(PolynomialFeaturesStep, interaction_only=interaction_only)

    def select_kbest(self, k, score_function=None):
        """Feature selection with scikit-learn's SelectKBest"""
        return self.queue(SelectKBestStep, score_function=score_function, k=k)

    def rfe(self, estimator, k):
        """Feature selection with scikit-learn's RFE"""
        return self.queue(RfeStep, estimator=estimator, k=k)

    def fft(self, featurewise=False):
        """Apply FFT (Fast Fourier Transform)"""
        self.includes.append('arduinoFFT.h')
        return self.queue(FFTStep, featurewise=featurewise)

    def principal_fft(self, n_components):
        """Apply "principal components" FFT"""
        assert n_components > 0, "n_components MUST be positive"
        fft = PrincipalFFT(n_components).fit(self.X)
        self._apply(lambda X: fft.transform(X))
        self.steps.append({
            "template": "PrincipalFFT",
            "n_components": n_components,
            "fft": fft
        })