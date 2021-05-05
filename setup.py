from setuptools import setup, find_packages
from os import path

here = path.abspath(path.dirname(__file__))

# Get the long description from the README file
with open(path.join(here, 'README.rst'), encoding='utf-8') as f:
    long_description = f.read()

setup(
    name='qcoptim',
    version='0.1',
    description='',
    long_description=long_description,
    url='',
    author='Frederic Sauvage, Kiran Khosla, Chris Self',
    classifiers=[],
    keywords='',
    packages=find_packages(exclude=['docs', 'tests', 'studies']),
    install_requires=[
        'numpy',
        'GPyOpt',
        'qiskit',
        'quimb',
        'qutip',
        'openfermion',
        'pyscf',
        'pytket',
        'pytket-qiskit',
    ],
    extras_require={},
    project_urls={},
)
