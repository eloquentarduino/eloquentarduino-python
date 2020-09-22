import numpy as np
from sklearn.preprocessing import PolynomialFeatures
from eloquentarduino.ml.data.preprocessing import PrincipalFFT
from eloquentarduino.ml.utils import jinja


class Pipeline:
    """Define a pre-processing pipeline that can be ported to plain C"""

    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.includes = []
        self.steps = []
        self.variables = []
        self.env = {
            "features_original_dimension": len(X[0]),
            "features_transformed_dimension": len(X[0])
        }

    def normalize(self, featurewise=False):
        """Apply normalization"""
        axis = 0 if featurewise else None
        self.steps.append({
            "template": "MinMaxScaler",
            "min": self.X.min(axis),
            "range": self.X.max(axis) - self.X.min(axis),
        })

    def standardize(self, featurewise=False):
        """Apply standardization"""
        axis = 0 if featurewise else None
        self.steps.append({
            "template": "StandardScaler",
            "mean": self.X.mean(axis),
            "std": self.X.std(axis),
        })

    def polynomial(self, interaction_only=False):
        """Apply polynomial (degree 2) features expansion"""
        self._apply(lambda X: PolynomialFeatures(degree=2, interaction_only=interaction_only, include_bias=False).fit_transform(self.X))
        self.steps.append({
            "template": "PolynomialFeatures",
            "interaction_only": interaction_only
        })

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

    def _apply(self, transform):
        """Apply a transform to X"""
        self.X = transform(self.X)
        self.env["features_transformed_dimension"] = max(self.env["features_transformed_dimension"], len(self.X[0]))

    def port(self):
        """Convert to plain C"""
        self.env.update({
            "includes": self.includes,
            "steps": self.steps,
            "variables": self.variables
        })
        return jinja("Pipeline/FixedSize.jinja", self.env)

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