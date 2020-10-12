import matplotlib.pyplot as plt
import numpy as np

from eloquentarduino.ml.metrics.plot import Bar, ConfusionMatrix


class PerformanceAssessmentResultsPlotter:
    """
    Plot performance assessment results
    """
    def __init__(self, assessment):
        self._assessment = assessment

    def overall(self):
        """
        Plot overall statistics
        :return: self
        """
        clf = str(type(self._assessment.results[0].clf))
        num_datasets = len(self._assessment.results)
        width = 1
        fig = plt.figure(figsize=(num_datasets * 2, num_datasets * 1.414))
        ax = fig.add_subplot(111)
        xs = np.arange(num_datasets) * width * 1.5
        ys = [np.diagonal(result.confusion_matrix).mean() for result in self._assessment.results]

        # show raw results
        if self._assessment.baseline is None:
            Bar(xs, ys).hat(lambda x, y: '%.2f' % y).plot(ax, color='b', label=clf)
            ax.set_xticks(xs)
        # compare to baseline
        else:
            xs *= 2
            ys_baseline = [np.diagonal(result.confusion_matrix).mean() for result in self._assessment.baseline]
            clf_baseline = str(type(self._assessment.baseline[0].clf))

            # baseline
            Bar(xs + width * 1.05, ys_baseline).hat(lambda x, y: '%.2f' % y).plot(ax, color='b', label=clf_baseline)
            # better than baseline
            Bar(xs, ys, ys_baseline).filter(lambda x, y1, y2: y1 >= y2).hat(lambda x, y1, y2: '+%.2f' % (y1 - y2)).plot(ax, color='g', label='better than baseline')
            # worse than baseline
            Bar(xs, ys, ys_baseline).filter(lambda x, y1, y2: y1 < y2).hat(lambda x, y1, y2: '%.2f' % (y1 - y2)).plot(ax, color='r', label='worse than baseline')
            ax.set_xticks(xs + width / 2)

        ax.set_xlabel('Datasets')
        ax.set_ylabel('Accuracy')
        ax.margins(y=0.2)
        ax.set_xticklabels([result.dataset for result in self._assessment.results])
        ax.legend(loc='lower right')
        plt.show()
        return self

    def detail(self):
        for i, result in enumerate(self._assessment.results):
            # plot confusion matrix
            confusion_matrix = result.confusion_matrix
            num_classes = len(confusion_matrix)

            fig = plt.figure(figsize=(num_classes * 2, num_classes))
            ax = fig.add_subplot(111)

            if self._assessment.baseline is None:
                ConfusionMatrix(confusion_matrix).plot(ax=ax, label=result.dataset)
            else:
                # compare results with baseline
                diff = confusion_matrix - self._assessment.baseline[i].confusion_matrix
                delta = min(0.5, max(0.2, np.abs(diff).max()))
                ConfusionMatrix(diff).cmap("vlag").range((-delta, delta)).plot(ax=ax, label=result.dataset)