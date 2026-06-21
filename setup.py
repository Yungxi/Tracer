#!/usr/bin/env python3
"""Setup script for Tracer - Python Code Tracer with LLM Judge"""

from setuptools import setup, find_packages
import os

# Read README for long description
def read_readme():
    readme_path = os.path.join(os.path.dirname(__file__), 'README.md')
    if os.path.exists(readme_path):
        with open(readme_path, 'r', encoding='utf-8') as f:
            return f.read()
    return ''

# Read requirements
def read_requirements():
    req_path = os.path.join(os.path.dirname(__file__), 'requirements.txt')
    if os.path.exists(req_path):
        with open(req_path, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip() and not line.startswith('#')]
    return []

setup(
    name='tracer-llm',
    version='0.1.0',
    description='Python code tracing tool with LLM-based output evaluation',
    long_description=read_readme(),
    long_description_content_type='text/markdown',
    author='Yungxi',
    author_email='',
    url='https://github.com/Yungxi/Tracer',
    license='MIT',

    # Package discovery
    py_modules=[
        'tracer',
        'parser',
        'executor',
        'judge',
        'reporter',
        'patcher',
        'properties',
        'judge_property',
        'judge_test_property',
    ],

    # Dependencies
    install_requires=read_requirements(),
    python_requires='>=3.8',

    # CLI entry point
    entry_points={
        'console_scripts': [
            'tracer=tracer:main',
        ],
    },

    # Package data
    include_package_data=True,
    package_data={
        '': ['config.example.json', 'README.md'],
    },

    # Classifiers
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
        'Topic :: Software Development :: Debuggers',
        'Topic :: Software Development :: Quality Assurance',
        'Topic :: Software Development :: Testing',
    ],

    # Keywords
    keywords='tracer debugging llm code-analysis testing',

    # Project URLs
    project_urls={
        'Source': 'https://github.com/Yungxi/Tracer',
        'Bug Reports': 'https://github.com/Yungxi/Tracer/issues',
    },
)
