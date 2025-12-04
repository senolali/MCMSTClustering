from setuptools import setup, find_packages
import codecs
import os

# Read the README file
with codecs.open('README.md', 'r', 'utf-8') as f:
    long_description = f.read()

setup(
    name="mcmstclustering",
    version="1.0.7",
    author="Ali Senol",
    author_email="alisenol@tarsus.edu.tr",
    description="MCMSTClustering: Defining Non-Spherical Clusters using Minimum Spanning Tree over KD-tree-based Micro-Clusters",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/yourusername/mcmstclustering",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.7",
    install_requires=[
        "numpy>=1.19.0",
        "scipy>=1.6.0",
        "scikit-learn>=0.24.0",
        "matplotlib>=3.3.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov>=2.0",
            "flake8>=3.9",
            "black>=21.0",
        ],
    },
    keywords=[
        "clustering",
        "machine-learning",
        "data-mining",
        "unsupervised-learning",
        "minimum-spanning-tree",
        "kd-tree",
    ],
    project_urls={
        "Documentation": "https://github.com/senolali/mcmstclustering",
        "Source": "https://github.com/senolali/mcmstclustering",
        "Bug Reports": "https://github.com/senolali/mcmstclustering",
    },
)
