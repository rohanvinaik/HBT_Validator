"""Setup script for HBT Validator package - Enhanced with Universal Verifier dependencies."""

from setuptools import setup, find_packages
from pathlib import Path

# Read README
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text()

# Core requirements
install_requires = [
    # Scientific computing
    "numpy>=1.21.0,<2.0.0",
    "scipy>=1.7.0,<2.0.0",
    "scikit-learn>=1.0.0,<2.0.0",
    
    # Data manipulation
    "pandas>=1.3.0,<3.0.0",
    
    # Visualization
    "matplotlib>=3.4.0,<4.0.0",
    "seaborn>=0.11.0,<1.0.0",
    
    # Graph and network analysis (NEW)
    "networkx>=2.6.0,<4.0.0",
    
    # Configuration
    "pyyaml>=5.4.0,<7.0.0",
    
    # Utilities
    "tqdm>=4.62.0,<5.0.0",
    "python-dateutil>=2.8.0,<3.0.0",
]

setup(
    name="hbt-validator",
    version="1.0.0",  # Updated to 1.0.0 with Universal Verifier
    author="HBT Paper Authors",
    description="Holographic Behavioral Twin (HBT) - Universal AI Model Verification System",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/rohanvinaik/HBT_Validator",
    packages=find_packages(include=['core*', 'utils*', 'verification*', 'challenges*', 'experiments*', 'tests*', 'config*']),
    classifiers=[
        "Development Status :: 4 - Beta",  # Upgraded from Alpha
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: System :: Monitoring",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    python_requires=">=3.8",
    install_requires=install_requires,
    extras_require={
        # Development tools
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "pytest-asyncio>=0.18.0",
            "black>=22.0.0",
            "mypy>=0.910",
            "flake8>=4.0.0",
            "isort>=5.10.0",
            "pre-commit>=2.15.0",
        ],
        # Visualization
        "viz": [
            "matplotlib>=3.5.0",
            "seaborn>=0.11.0",
            "plotly>=5.0.0",
        ],
        # Advanced tensor decomposition (OPTIONAL)
        "advanced": [
            "tensorly>=0.7.0,<1.0.0",
        ],
        # Performance optimization
        "performance": [
            "numba>=0.54.0,<1.0.0",
            "numexpr>=2.8.0,<3.0.0",
        ],
        # API serving
        "api": [
            "fastapi>=0.70.0,<1.0.0",
            "uvicorn>=0.15.0,<1.0.0",
            "pydantic>=1.8.0,<3.0.0",
        ],
        # Monitoring
        "monitoring": [
            "prometheus-client>=0.12.0,<1.0.0",
        ],
        # All optional dependencies
        "all": [
            "tensorly>=0.7.0,<1.0.0",
            "numba>=0.54.0,<1.0.0",
            "fastapi>=0.70.0,<1.0.0",
            "uvicorn>=0.15.0,<1.0.0",
            "prometheus-client>=0.12.0,<1.0.0",
        ],
    },
    include_package_data=True,
    package_data={
        "": ["*.json", "*.yaml", "*.yml"],
        "config": ["*.yaml", "*.yml"],
    },
    entry_points={
        "console_scripts": [
            "hbt-verify=cli.verify:main",
            "hbt-validate-config=config.config_validator:validate_config_file",
        ],
    },
    keywords=[
        "ai", "machine-learning", "verification", "model-monitoring",
        "behavioral-analysis", "contamination-detection", "security",
        "hbt", "holographic-behavioral-twin", "byzantine-consensus"
    ],
)