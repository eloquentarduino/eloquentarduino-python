import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from IPython.display import clear_output
from sklearn.model_selection import train_test_split


class TimeOfFlightVisualizer:
    """
    Plot 8x8 time of flight dataset
    """
    def __init__(self, dataset):
        self.dataset = dataset

    def plot(self, samples_per_class=30, pause=0, z_lim=None, rotate=True, colormap='jet'):
        """
        Draw animated plot
        :param samples_per_class: int
        :param pause: float
        :param z_lim: tuple
        :param rotate: bool
        :param colormap: str
        """
        cmap = get_cmap(colormap)

        xpos, ypos = np.meshgrid(range(8), range(8), indexing="ij")
        xpos = xpos.ravel()
        ypos = ypos.ravel()
        zpos = np.zeros(len(self.dataset.X[0]))

        m = self.dataset.X.min()

        for y, Xs in self.dataset.iterate_classes():
            class_m = Xs.min()
            class_M = Xs.max()
            X_sample, _1 = train_test_split(Xs, train_size=samples_per_class)

            for i, sample in enumerate(X_sample):
                clear_output(wait=True)
                fig = plt.figure()
                ax = fig.add_subplot(projection='3d')
                rgba = [cmap((x - class_m) / (class_M - class_m)) for x in sample]
                # log scale should help visualization
                heights = np.log(sample - m + 1)

                ax.bar3d(xpos, ypos, zpos, 1, 1, heights, color=rgba)

                if rotate:
                    ax.view_init(30, int(i / samples_per_class / self.dataset.num_classes * 360))

                if z_lim is not None:
                    ax.set_zlim(z_lim)

                plt.title('%s (%d/%d)' % (self.dataset.classmap[y], i + 1, len(Xs)))
                plt.show()

                if pause > 0:
                    plt.pause(pause)
