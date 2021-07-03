import matplotlib.pyplot as plt
from eloquentarduino.plot.Bar import Bar
from eloquentarduino.plot.ConfusionMatrix import ConfusionMatrix
from eloquentarduino.plot.DimensionalityReductionPlotter import DimensionalityReductionPlotter
from eloquentarduino.plot.PCAPlotter import PCAPlotter
from eloquentarduino.plot.TSNEPlotter import TSNEPlotter
from eloquentarduino.plot.RankMatrix import RankMatrix
from eloquentarduino.plot.Scatter import Scatter, scatter
from eloquentarduino.plot.Line import Line, line


def large_plots(size=(15, 10)):
    """
    Make large plots
    """
    plt.rcParams["figure.figsize"] = size