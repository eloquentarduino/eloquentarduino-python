import os.path
from distutils.core import setup
from glob import glob
from os.path import isdir


def package_name(folder):
  if folder.endswith(os.path.sep):
    folder = folder[:-1]
  return folder.replace(os.path.sep, '.')


packages = [package_name(folder) for folder in glob('eloquentarduino/**', recursive=True)
            if isdir(folder) and '__pycache__' not in folder]

data = [filename.replace('eloquentarduino/', '')
        for filename in glob('eloquentarduino/templates/**/*.jinja', recursive=True)]

setup(
  name = 'eloquentarduino',
  packages = packages,
  version = '0.0.12',
  license='MIT',
  description = 'A set of utilities to work with Arduino from Python and Jupyter Notebooks',
  author = 'Simone Salerno',
  author_email = 'eloquentarduino@gmail.com',
  url = 'https://github.com/eloquentarduino/eloquentarduino-python',
  download_url = 'https://github.com/eloquentarduino/eloquentarduino-python/dist/eloquentarduino-0.0.12.tar.gz',
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