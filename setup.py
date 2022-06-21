import os
import setuptools
from setuptools import setup

# Utility function to read the README file.
# Used for the long_description.  It's nice, because now 1) we have a top level
# README file and 2) it's easier to type in the README file than to put a raw
# string in below ...


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name="coherence",
    version="0.0.1",
    author="facuzeta",
    description=("coherence"),
    license="BSD",
    keywords="",
    url="https://ar.linkedin.com/in/facuzeta",
    packages=setuptools.find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Topic :: Utilities",
        "License :: OSI Approved :: BSD License",
    ],
    install_requires=['numpy', 'scipy'],
    package_data={'coherence': ['models/tasa_150/dictionary.json',
                                'models/tasa_150/matrix.npy']},
    include_package_data=True,
)
