import os.path
from distutils.core import setup
from glob import glob
from os.path import isdir


#def package_name(folder):
#  if folder.endswith(os.path.sep):
#    folder = folder[:-1]
#  return folder.replace(os.path.sep, '.')


#packages = [package_name(folder) for folder in glob('eloquentarduino/**', recursive=True)
#            if isdir(folder) and '__pycache__' not in folder]

#data = [filename.replace('eloquentarduino/', '')
#        for filename in glob('eloquentarduino/templates/**/*.jinja', recursive=True)]

packages = ["eloquentarduino", "eloquentarduino.plot", "eloquentarduino.plot.dynamic", "eloquentarduino.utils", "eloquentarduino.third_party", "eloquentarduino.third_party.snoopy", "eloquentarduino.ml", "eloquentarduino.ml.classification", "eloquentarduino.ml.classification.abstract", "eloquentarduino.ml.classification.sklearn", "eloquentarduino.ml.classification.sklearn.gridsearch", "eloquentarduino.ml.classification.sklearn.online", "eloquentarduino.ml.classification.tensorflow", "eloquentarduino.ml.classification.tensorflow.gridsearch", "eloquentarduino.ml.classification.device", "eloquentarduino.ml.metrics", "eloquentarduino.ml.metrics.plot", "eloquentarduino.ml.metrics.training", "eloquentarduino.ml.metrics.device", "eloquentarduino.ml.metrics.device.parsers", "eloquentarduino.ml.metrics.device.benchmarks", "eloquentarduino.ml.plot", "eloquentarduino.ml.cascading", "eloquentarduino.ml.data", "eloquentarduino.ml.data.preprocessing", "eloquentarduino.ml.data.preprocessing.pipeline", "eloquentarduino.ml.data.preprocessing.pipeline.classification", "eloquentarduino.ml.data.preprocessing.pipeline.outliers", "eloquentarduino.ml.data.loaders", "eloquentarduino.templates", "eloquentarduino.templates.on_device", "eloquentarduino.templates.on_device.sklearn", "eloquentarduino.templates.on_device.tensorflow", "eloquentarduino.templates.metrics", "eloquentarduino.templates.Pipeline", "eloquentarduino.templates.magics", "eloquentarduino.templates.third_party", "eloquentarduino.templates.third_party.snoopy", "eloquentarduino.templates.ml", "eloquentarduino.templates.ml.classification", "eloquentarduino.templates.ml.classification.tensorflow", "eloquentarduino.templates.ml.data", "eloquentarduino.templates.ml.data.preprocessing", "eloquentarduino.templates.ml.data.preprocessing.pipeline", "eloquentarduino.templates.ml.data.preprocessing.pipeline.templates", "eloquentarduino.templates.benchmarks", "eloquentarduino.templates.benchmarks.sklearn", "eloquentarduino.templates.benchmarks.tf", "eloquentarduino.jupyter", "eloquentarduino.jupyter.magics", "eloquentarduino.jupyter.project"]
data = ["templates/on_device/sklearn/ResourcesBaseline.jinja", "templates/on_device/sklearn/Resources.jinja", "templates/on_device/sklearn/_Resources.jinja", "templates/on_device/sklearn/InferenceTime.jinja", "templates/on_device/tensorflow/ResourcesBaseline.jinja", "templates/on_device/tensorflow/Resources.jinja", "templates/on_device/tensorflow/_Resources.jinja", "templates/on_device/tensorflow/InferenceTime.jinja", "templates/metrics/Runtime.bck.jinja", "templates/metrics/Resources.jinja", "templates/metrics/Runtime.jinja", "templates/metrics/Empty.jinja", "templates/metrics/Baseline.jinja", "templates/Pipeline/PrincipalFFT.jinja", "templates/magics/eloquent-arduino.h.jinja", "templates/third_party/snoopy/SnoopyStream.jinja", "templates/third_party/snoopy/snoopy.jinja", "templates/ml/classification/tensorflow/NeuralNetwork.jinja", "templates/ml/data/preprocessing/pipeline/PolynomialFeatures.jinja", "templates/ml/data/preprocessing/pipeline/StandardScaler.jinja", "templates/ml/data/preprocessing/pipeline/BoxCox.jinja", "templates/ml/data/preprocessing/pipeline/TSFRESH_@only.jinja", "templates/ml/data/preprocessing/pipeline/RateLimit.jinja", "templates/ml/data/preprocessing/pipeline/kFFT.jinja", "templates/ml/data/preprocessing/pipeline/Classify.jinja", "templates/ml/data/preprocessing/pipeline/RFE.jinja", "templates/ml/data/preprocessing/pipeline/SmoothClassification.jinja", "templates/ml/data/preprocessing/pipeline/MinMaxScaler.jinja", "templates/ml/data/preprocessing/pipeline/Window.jinja", "templates/ml/data/preprocessing/pipeline/Pipeline.jinja", "templates/ml/data/preprocessing/pipeline/EllipticEnvelope.jinja", "templates/ml/data/preprocessing/pipeline/AbstractStep.jinja", "templates/ml/data/preprocessing/pipeline/KMeansOutlier.jinja", "templates/ml/data/preprocessing/pipeline/Cascading.jinja", "templates/ml/data/preprocessing/pipeline/FFT.jinja", "templates/ml/data/preprocessing/pipeline/TSFRESH.jinja", "templates/ml/data/preprocessing/pipeline/TSFRESH_if.jinja", "templates/ml/data/preprocessing/pipeline/DFT.jinja", "templates/ml/data/preprocessing/pipeline/StatMoments.jinja", "templates/ml/data/preprocessing/pipeline/Diff.jinja", "templates/ml/data/preprocessing/pipeline/YeoJohnson.jinja", "templates/ml/data/preprocessing/pipeline/Norm.jinja", "templates/ml/data/preprocessing/pipeline/SelectKBest.jinja", "templates/ml/data/preprocessing/pipeline/templates/Namespace.jinja", "templates/ml/data/preprocessing/pipeline/templates/Step.jinja", "templates/benchmarks/Baseline.jinja", "templates/benchmarks/sklearn/Resources.jinja", "templates/benchmarks/sklearn/Runtime.jinja", "templates/benchmarks/tf/Resources.jinja", "templates/benchmarks/tf/Runtime.jinja"]

setup(
  name = 'eloquentarduino',
  packages = packages,
  version = '0.0.60',
  license='MIT',
  description = 'A set of utilities to work with Arduino from Python and Jupyter Notebooks',
  author = 'Simone Salerno',
  author_email = 'eloquentarduino@gmail.com',
  url = 'https://github.com/eloquentarduino/eloquentarduino-python',
  download_url = 'https://github.com/eloquentarduino/eloquentarduino-python/blob/master/dist/eloquentarduino-0.0.60.tar.gz?raw=true',
  keywords = [
    'ML',
    'Jupyter',
    'microcontrollers',
    'sklearn',
    'machine learning'
  ],
  install_requires=[
    'ipython',
    'numpy',
    'scikit-learn',
    'matplotlib',
    'Jinja2',
    'pyserial',
    'pandas',
    'seaborn',
    'micromlgen'
  ],
  package_data= {
    'eloquentarduino': data
  },
  classifiers=[
    'Development Status :: 2 - Pre-Alpha',
    'Intended Audience :: Developers',
    'Topic :: Software Development :: Code Generators',
    'License :: OSI Approved :: MIT License',
    'Programming Language :: Python :: 3',
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
  ],
)
