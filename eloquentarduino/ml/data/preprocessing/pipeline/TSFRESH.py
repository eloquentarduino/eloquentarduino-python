import re
import numpy as np
from math import ceil, floor
from collections import Counter
from sklearn.feature_selection import SelectKBest, mutual_info_classif
from eloquentarduino.ml.data.preprocessing.pipeline.BaseStep import BaseStep


class TSFRESH(BaseStep):
    """
    Extract many time-series features
    """

    def __init__(self, num_features, k=0, name='TSFRESH'):
        """
        :param num_features: int
        """
        assert num_features > 0, 'num_features MUST be greater than 0'

        super().__init__(name)
        self.num_features = num_features
        self.k = k
        self.constants = {}
        self.kbest = None

    @property
    def available_features(self):
        """
        Get names of available features
        """
        return [
            'maximum',
            'minimum',
            'abs_maximum',
            'abs_minimum',
            'mean',
            'abs_energy',
            'mean_abs_change',
            'cid_ce',
            'energy_ratio',
            'range_count',

            # first-order, lag-2 features
            'c31',
            'mean_second_derivative_center',
            'time_reversal_asymmetry_statistic1',

            # second-order features
            'std',
            'var',
            'count_above_mean',
            'count_below_mean',
            'first_position_of_max',
            'first_position_of_min',
            'has_duplicate_max',
            'has_duplicate_min',
            'has_large_std',
            'autocorrelation1',
            'skew',
            'kurtosis',
            'zero_crossings',
            'variation_coefficient',
        ]

    @property
    def used_features(self):
        """
        Get names of used features
        """
        if self.kbest is None:
            return self.available_features

        return [self.available_features[k] for k in self.kbest[:len(self.kbest) // self.num_features]]

    @property
    def buffer_size(self):
        """
        Get buffer size
        """
        return self.num_features * len(self.used_features)

    @property
    def dependencies(self):
        """
        Return features other features depends on
        :return: list
        """
        if self.k == 0:
            return []

        dependencies = {
            'std': ['mean', 'var'],
            'var': ['mean'],
            'count_above_mean': ['mean'],
            'count_below_mean': ['mean'],
            'first_position_of_max': ['maximum'],
            'first_position_of_min': ['minimum'],
            'has_duplicate_max': ['maximum'],
            'has_duplicate_min': ['minimum'],
            'has_large_std': ['mean', 'var', 'std'],
            'autocorrelation1': ['mean'],
            'skew': ['mean', 'var'],
            'kurtosis': ['mean', 'var'],
            'zero_crossings': ['mean'],
            'variation_coefficient': ['mean', 'var']
        }
        used_dependencies = [dep for feature_name, dep in dependencies.items() if feature_name in self.used_features]

        return list(set([d for dep in used_dependencies for d in dep]))

    def get_config(self):
        """
        Get configuration options
        :return: dict
        """
        return {
            'num_features': self.num_features,
            'feature_names': self.used_features,
            'k': self.k
        }

    def fit(self, X, y):
        """
        Fit
        """
        self.set_X(X)

        if self.num_features < 1:
            self.num_features = int(self.input_dim * self.num_features)

        # @todo
        self.working_dim = self.buffer_size
        self.kbest = None

        for k in range(self.num_features):
            ts = X[:, k::self.num_features]

            global_min = ts.min()
            global_max = ts.max()

            self.constants[k] = {
                'min': global_min,
                'max': global_max,
                'range': global_max - global_min
            }

        return self.transform(X, y)

    def transform(self, X, y=None):
        """
        Transform
        """
        assert (self.input_dim % self.num_features) == 0, 'num_features MUST be a divisor of X.shape[1]'

        FRESH = None
        eps = 1e-5

        for k in range(self.num_features):
            ts = X[:, k::self.num_features]
            size = ts.shape[1]

            global_min = self.constants[k]['min']
            global_max = self.constants[k]['max']
            global_range = self.constants[k]['range']

            # first-order features
            maximum = ts.max(axis=1).reshape((-1, 1))
            minimum = ts.min(axis=1).reshape((-1, 1))
            abs_maximum = np.abs(maximum)
            abs_minimum = np.abs(minimum)
            mean = ts.mean(axis=1).reshape((-1, 1))
            abs_energy = (ts ** 2).sum(axis=1) # really mean
            mean_abs_change = np.abs(np.diff(ts, axis=1)).sum(axis=1)  # really mean
            cid_ce = (np.diff(ts, axis=1) ** 2).sum(axis=1)
            energy_ratio = (ts[:, ceil(size * 3/8):floor(size * 5/8)+1] ** 2).sum(axis=1) # really mean
            range_count = ((ts >= global_min + global_range / 4) & (ts <= global_max - global_range / 4)).sum(axis=1)

            # first-order, lag-2 features
            c31 = (ts[:, :-2] * ts[:, 1:-1] * ts[:, 2:]).sum(axis=1) # really mean
            mean_second_derivative_center = (ts[:, 2:] - 2 * ts[:, 1:-1] + ts[:, :-2]).sum(axis=1) # really mean
            time_reversal_asymmetry_statistic1 = ((ts[:, 2:] ** 2) * ts[:, 1:-1] - ts[:, 1:-1] * (ts[:, :-2] ** 2)).sum(axis=1) # really mean

            # second-order features
            ts_zero_mean = ts - mean
            std = ts.std(axis=1).reshape((-1, 1))
            var = ((ts - mean) ** 2).mean(axis=1).reshape((-1, 1))
            count_above_mean = (ts_zero_mean > 1e-5).sum(axis=1)
            count_below_mean = (ts_zero_mean < -1e-5).sum(axis=1)
            first_position_of_max = np.argmax(ts, axis=1)
            first_position_of_min = np.argmin(ts, axis=1)
            has_duplicate_max = (ts > (maximum - np.abs(maximum) * 0.02)).sum(axis=1)
            has_duplicate_min = (ts < (minimum + np.abs(minimum) * 0.02)).sum(axis=1)
            has_large_std = (std > 0.25 * (maximum - minimum))
            autocorrelation1 = (ts_zero_mean[:, 1:] * ts_zero_mean[:, :-1]).sum(axis=1) + (ts_zero_mean[:, 0] ** 2) # really mean
            skew = np.where(var < eps, 0, ((ts - mean) ** 3) / (var ** 1.5)).sum(axis=1).reshape((-1, 1)) # really mean
            kurtosis = np.where(np.abs(var) < eps, 0, ((ts - mean) ** 4) / (var ** 2)).sum(axis=1) # really mean
            zero_crossings = ((ts[:, 1:-1] - ts[:, :-2]) * (ts[:, 2:] - ts[:, 1:-1]) < 0).sum(axis=1)
            variation_coefficient = np.where(mean < eps, 0, var / mean) # really std

            #changes = np.diff(ts, axis=1)
            #abs_sum_of_changes = np.abs(changes).mean(axis=1)
            # approximate_entropy (to consider)
            # benford_correlation (to consider)

            #mass_center = np.argmax((np.abs(ts).cumsum(axis=1) / np.abs(ts).sum(axis=1).reshape((-1, 1))) > 0.5, axis=1) / size
            # number of peaks (to consider)
            # query similarity count (to consider + sparse support vectors)
            #ratio_beyond_half_sigma = np.sum(np.abs(ts_zero_mean) > 0.5 * std, axis=1).astype(float) / size
            #root_mean_square = np.sqrt(np.mean(np.square(ts), axis=1)) <-- redundant with abs_energy

            fresh = [
                # first-order features
                maximum,
                minimum,
                abs_maximum,
                abs_minimum,
                mean,
                abs_energy,
                mean_abs_change,
                cid_ce,
                energy_ratio,
                range_count,

                # first-order, lag-2 features
                c31,
                mean_second_derivative_center,
                time_reversal_asymmetry_statistic1,

                # second-order features
                std,
                var,
                count_above_mean,
                count_below_mean,
                first_position_of_max,
                first_position_of_min,
                has_duplicate_max,
                has_duplicate_min,
                has_large_std,
                autocorrelation1,
                skew,
                kurtosis,
                zero_crossings,
                variation_coefficient,
            ]

            fresh = np.hstack([feature.reshape((-1, 1)) for feature in fresh])

            if FRESH is None:
                FRESH = fresh
            else:
                FRESH = np.hstack((FRESH, fresh))

        return self.select_k_best(FRESH, y), np.asarray(y)

    def select_k_best(self, X, y):
        """
        Perform feature selection
        """
        if self.k == 0:
            return X

        if self.kbest is None:
            best_features = []
            num_derived_features = X.shape[1] // self.num_features

            for feature_idx in range(self.num_features):
                Xi = X[:, feature_idx * num_derived_features:(feature_idx + 1) * num_derived_features]
                kbest = SelectKBest(k=self.k, score_func=mutual_info_classif).fit(Xi, y)
                idx = (-kbest.scores_).argsort()[:self.k]
                best_features += idx.tolist()

            # keep features that perform best across all the dimensions
            most_common = np.asarray([feature_idx for feature_idx, count in Counter(best_features).most_common(self.k)])
            self.kbest = sorted([x for i in range(self.num_features) for x in (most_common + i * num_derived_features).tolist()])

        return X[:, self.kbest]

    def get_template_data(self):
        """

        """
        second_order_features = [
            'std',
            'var',
            'autocorrelation1',
            'count_above_mean',
            'count_below_mean',
            'first_position_of_max',
            'first_position_of_min',
            'has_duplicate_max',
            'has_duplicate_min',
            #'mass_center',
            'kurtosis',
            'has_large_std',
            #'ratio_beyond_half_sigma',
            'skew',
            'variation_coefficient'
        ]

        optimizations = [
            'square' if self._intersects('abs_energy', 'time_reversal_asymmetry_statistic1') else None,
            'lag' if self._intersects('cid_ce') else None,
            'lag2' if self._intersects('c31', 'mean_second_derivative_center', 'time_reversal_asymmetry_statistic1') else None,
        ]

        constants = {
            'global_low': [c['min'] + c['range'] / 4 for c in self.constants.values()],
            'global_high': [c['max'] - c['range'] / 4 for c in self.constants.values()],
        }

        return {
            'num_features': self.num_features,
            'k': self.k,
            'buffer_size': self.buffer_size,
            'constants': constants,
            'is_second_order': self._intersects(*second_order_features),
            'opt': set(optimizations),
            'num_samples': self.input_dim // self.num_features,
            'eps': '1e-5'
        }

    def postprocess_port(self, ported):
        """
        Compute only K best features
        """
        features_to_keep = self.used_features + self.dependencies
        features_to_discard = [feature for feature in self.available_features if feature not in features_to_keep]

        # drop lines that compute un-needed features
        for feature in sorted(features_to_discard, key=lambda feature: len(feature), reverse=True):
            # comment variable declaration
            ported = re.sub('[ \t]+float %s[_ ][^\n]+?\n' % feature, lambda m: '// %s' % m.group(0), ported)
            # comment conditional assignments
            ported = re.sub('[ \t]+if [^{]+\{ %s[_ ][^\n]+?\n' % feature, lambda m: '// %s' % m.group(0), ported)
            # comment assignments
            ported = re.sub('[ \t]+%s[_ ][^\n]+?\n' % feature, lambda m: '// %s' % m.group(0), ported)
            # comment buffer assignment
            ported = re.sub('[ \t]+buffer\[idx\+\+\] = %s;\n' % feature, lambda m: '// %s' % m.group(0), ported)

        return ported

    def _intersects(self, *args):
        """
        Test intersection of given features with know features
        """
        return len(set(args).intersection(self.used_features)) > 0
