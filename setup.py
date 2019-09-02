#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""The setup script."""

from setuptools import setup, find_packages

with open('README.rst') as readme_file:
    readme = readme_file.read()

with open('HISTORY.rst') as history_file:
    history = history_file.read()

PYTHON_VERSIONS = '>=3'

requirements = [
    'tensorflow==2.0.0-rc0; platform_system=="Darwin"',
    'tensorflow-gpu==2.0.0-rc0; platform_system=="Linux"',
    'tf-agents-nightly',
    'gym',
    'gym-donkeycar',
]

setup_requirements = ['pytest-runner', ]

test_requirements = ['pytest', ]

setup(
    author="Leigh Johnson",
    author_email='leigh@data-literate.com',
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Natural Language :: English',
        "Programming Language :: Python :: 2",
        'Programming Language :: Python :: 2.7',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.4',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
    ],
    python_requires=PYTHON_VERSIONS,
    description="Training a self-driving car using TensorFlow, Raspberry Pi, and Donkey Car",
    install_requires=requirements,
    license="MIT license",
    long_description=readme + '\n\n' + history,
    include_package_data=True,
    keywords='rpi_rover',
    name='rpi_rover',
    packages=find_packages(include=['rpi_rover']),
    dependency_links=[
        'git+ssh://git@github.com/leigh-johnson/agents.git@multidiscrete-testing#egg=tf-agents-nightly-0.2.0.dev20190816',
        'git+ssh://git@github.com/leigh-johnson/gym-donkeycar.git@tf-agents-dqn#egg=gym-donkeycar-1.1.0',
    ],
    setup_requires=setup_requirements,
    test_suite='tests',
    tests_require=test_requirements,
    url='https://github.com/leigh-johnson/rpi_rover',
    version='0.1.0',
    zip_safe=False,
)
