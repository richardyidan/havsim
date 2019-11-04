
"""
@author: rlk268@cornell.edu
"""

import setuptools

with open('README.readme', 'r') as fh: 
    long_description = fh.read()
    
setuptools.setup(
        name = 'havsim',
        version = '0.0.1',
        author = 'ronan-keane',
        author_email = 'rlk268@cornell.edu',
        description='A differentiable traffic simulator for calibration of traffic models and optimization of AV behavior',
        long_description = long_description,
        long_description_content_type='text/markdown',
        url='https://github.com/ronan-keane/hav-sim',
        classifiers = [
                'Programming Language :: Python :: 3',
                'License :: Apache 2.0',
                'Operating System :: Linux/Mac required for Jax/NLopt',
        ],
        python_requires = '>=3.6',
        packages = setuptools.find_packages(exclude = ['Jax','nlopt','scripts']) #isolate jax and nlopt so it won't break the package on windows, which can't use these packages 
        #also exclude scripts because a lot of them are still outdated and won't run with the new package organization 
        )
