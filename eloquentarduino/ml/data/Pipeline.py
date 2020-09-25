from eloquentarduino.ml.data.preprocessing import NormalizeStep
from eloquentarduino.ml.data.preprocessing import PolynomialFeaturesStep
from eloquentarduino.ml.data.preprocessing import PrincipalFFT
from eloquentarduino.ml.data.preprocessing import RfeStep
from eloquentarduino.ml.data.preprocessing import SelectKBestStep
from eloquentarduino.ml.data.preprocessing import StandardizeStep
from eloquentarduino.utils import jinja
from sklearn.model_selection import train_test_split


class Pipeline:
    """Define a pre-processing pipeline that can be ported to plain C"""
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.steps = []
        self.steps_run = []

    @property
    def Xt(self):
        return self.transform(self.X.copy())

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
            'input_dim': self.X.shape[1],
            'working_dim': max([step.input_shape[1] for step in self.steps_run] + [step.output_shape[1] for step in self.steps_run])
        }
        return jinja("Pipeline/Pipeline.jinja", env, pretty=True)

    def score(self, clf, **kwargs):
        """Score classifier on the transformed input"""
        X_train, X_test, y_train, y_test = train_test_split(self.Xt, self.y, **kwargs)
        return clf.fit(X_train, y_train).score(X_test, y_test)

    def explain(self):
        """Return a human understandable representation of the pipeline"""
        return 'Pipeline description:\n' + ''.join(['\n - ' + str(step) for step in self.steps_run])[1:]

    def queue(self, Step, **kwargs):
        self.steps.append((Step, kwargs))

    def normalize(self, featurewise=False):
        """Apply normalization"""
        self.queue(NormalizeStep, featurewise=featurewise)

    def standardize(self, featurewise=False):
        """Apply standardization"""
        self.queue(StandardizeStep, featurewise=featurewise)

    def polynomial_features(self, interaction_only=False):
        """Apply 2° order polynomial features expansion"""
        return self.queue(PolynomialFeaturesStep, interaction_only=interaction_only)

    def select_kbest(self, k, score_function=None):
        """Feature selection with scikit-learn's SelectKBest"""
        return self.queue(SelectKBestStep, score_function=score_function, k=k)

    def rfe(self, estimator, k):
        """Feature selection with scikit-learn's RFE"""
        return self.queue(RfeStep, estimator=estimator, k=k)


    def fft(self, frequency, use="magnitude", window="HAMMING"):
        """Apply FFT (Fast Fourier Transform)"""
        # @todo Python FFT
        windows = [
            "RECTANGLE",
            "HAMMING",
            "HANN",
            "TRIANGLE",
            "NUTTALL",
            "BLACKMAN",
            "BLACKMAN_NUTTALL",
            "BLACKMAN_HARRIS",
            "FLT_TOP",
            "WELCH"
        ]
        uses = [
            "real",
            "magnitude",
            "both"
        ]
        assert frequency > 0, "frequency MUST be positive"
        assert window in windows, "window must be one of %s" % windows
        assert use in uses, "use must be one of %s" % uses
        self.includes.append("arduinoFFT.h")
        self.steps.append({
            "template": "FFT",
            "frequency": frequency,
            "window": "FFT_WIN_TYP_%s" % window,
            "use": use
        })

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

    def _offset(self, variable_name, operator, featurewise):
        """Remove offset from data"""
        axis = 0 if featurewise else None
        offset = operator(self.X, axis=axis)
        self.X -= offset
        self.steps.append({
            "op": "-",
            "variable": variable_name,
            "is_array": featurewise
        })
        self._add_variable("float", variable_name, offset, is_array=featurewise)

    def _inverse_gain(self, variable_name, operator, featurewise):
        """Divide by gain"""
        axis = 0 if featurewise else None
        gain = operator(self.X, axis=axis)
        self.X /= gain
        self.steps.append({
            "op": "*",
            "variable": variable_name,
            "is_array": featurewise
        })
        self._add_variable("float", variable_name, 1 / gain, is_array=featurewise)

    def _add_variable(self, type, name, value, is_array=False):
        """Add helper variable to C code"""
        if is_array:
            type += "[]"
        self.variables.append((type, name, value))