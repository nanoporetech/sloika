import re
from glob import glob
import numpy as np
import os
import subprocess
from setuptools import setup, find_packages
from Cython.Build import cythonize

try:
    root_dir = os.environ['ROOT_DIR']
except KeyError:
    root_dir = '.'
try:
    scripts_dir = os.environ['SCRIPTS_DIR']
except KeyError:
    scripts_dir = 'scripts'

cmd = os.path.join(scripts_dir, 'show-version.sh')
version, err = subprocess.Popen(cmd.split(), stdout=subprocess.PIPE).communicate()
version = version.decode('utf-8')

version_py_name = os.path.join(root_dir, 'sloika', 'version.py')
version_py_contents = u"__version__ = '%s'\n" % version
open(version_py_name, 'w').write(version_py_contents)

requirements_file = os.path.join(root_dir, 'requirements.txt')
with open(requirements_file) as f:
    install_requires = f.read().splitlines()

package_dir = os.path.join(root_dir, 'sloika')
setup(
    name='sloika',
    version=version,
    description='Neural network model training for Nanopore base calling',
    maintainer='Tim Massingham',
    maintainer_email='tim.massingham@nanoporetech.com',
    url='http://www.nanoporetech.com',
    long_description="""Sloika is a library to support training and developing new base calling models
for Oxford Nanopore Technologies' sequencing platforms.

Sloika is built on top of Theano package for computational graphs and is
compatible with python 3.4 or above.""",

    classifiers=[
        'Development Status :: 3 - Alpha',
        'Environment :: Console',
        'Intended Audience :: Developers',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Mozilla Public License 2.0 (MPL 2.0)',
        'Natural Language :: English',
        'Operating System :: Unix',
        'Programming Language :: Python :: 3 :: Only',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Mathematics'
    ],

    packages=find_packages(exclude=["*.test", "*.test.*", "test.*", "test", "bin"]),
    package_data={'configs': 'data/configs/*'},
    exclude_package_data={'': ['*.hdf', '*.c', '*.h']},
    ext_modules=cythonize(os.path.join(package_dir, "viterbi_helpers.pyx")),
    include_dirs=[np.get_include()],
    tests_require=[],
    install_requires=install_requires,
    dependency_links=[],
    zip_safe=False,
    scripts=[x for x in glob('bin/*.py')],

)
