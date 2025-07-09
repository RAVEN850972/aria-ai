# setup.py
"""Setup script для ARIA"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="aria-ai",
    version="0.1.0",
    author="ARIA Team",
    author_email="team@aria-ai.dev",
    description="Adaptive Reconfigurable Intelligence Architecture",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/aria-ai/aria",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Software Development :: Libraries :: Python Modules",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "PyYAML>=6.0",
        "dataclasses;python_version<'3.7'",
    ],
    extras_require={
        "dev": [
            "pytest>=7.0",
            "pytest-cov>=4.0", 
            "black>=22.0",
            "flake8>=5.0",
        ],
        "docs": [
            "sphinx>=5.0",
            "sphinx-rtd-theme>=1.0",
        ],
        "full": [
            "Pillow>=9.0",  # Для обработки изображений
            "librosa>=0.9.0",  # Для обработки аудио
            "opencv-python>=4.5.0",  # Для обработки видео
            "matplotlib>=3.5.0",  # Для визуализации
        ],
    },
    entry_points={
        "console_scripts": [
            "aria=aria.__main__:main",
        ],
    },
    include_package_data=True,
    package_data={
        "aria": [
            "configs/templates/*.yaml",
            "configs/examples/*.yaml",
        ],
    },
    keywords="ai, nlp, machine-learning, text-processing, multimodal",
    project_urls={
        "Bug Reports": "https://github.com/aria-ai/aria/issues",
        "Source": "https://github.com/aria-ai/aria",
        "Documentation": "https://aria-ai.readthedocs.io/",
    },
)