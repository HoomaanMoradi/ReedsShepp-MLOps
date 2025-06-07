"""
Package configuration for ReedsShepp-MLOps.

This setup script handles the package installation and dependencies.
"""

from setuptools import find_packages, setup

# Read the main requirements from requirements.txt
with open("requirements.txt") as f:
    requirements = f.read().splitlines()

# Read development requirements from dev-requirements.txt
with open("dev-requirements.txt") as f:
    dev_requirements = f.read().splitlines()

# Package configuration
setup(
    # Basic package information
    name="reedsshepp-mlops",
    version="0.1.0",
    author="Hoomaan Moradi",
    # Automatically find all packages in the project
    packages=find_packages(),
    # Main dependencies required for the package to run
    install_requires=requirements,
    # Optional dependencies groups
    extras_require={
        # Development dependencies (linters, formatters, etc.)
        # Install with: pip install -e ".[dev]"
        "dev": dev_requirements,
    },
)
