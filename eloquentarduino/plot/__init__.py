import matplotlib.pyplot as plt
from eloquentarduino.plot.Bar import Bar
from eloquentarduino.plot.ConfusionMatrix import ConfusionMatrix
from eloquentarduino.plot.PCAPlotter import PCAPlotter
from eloquentarduino.plot.TSNEPlotter import TSNEPlotter
from eloquentarduino.plot.RankMatrix import RankMatrix


def large_plots(size=(15, 10)):
    """
    Make large plots
    """
    plt.rcParams["figure.figsize"] = size