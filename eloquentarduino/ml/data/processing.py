import numpy as np


def describe_data(X, axis=None, percentiles=[25, 50, 75]):
    """Describe data similar to Pandas"""
    print("count", len(X))
    print("mean", X.mean(axis=axis))
    print("std", X.std(axis=axis))
    print("min", X.min(axis=axis))
    for p in percentiles:
        print("%d%%" % p, np.percentile(X, p, axis=axis))
    print("max", X.max(axis=axis))