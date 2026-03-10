from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

with open("requirements.txt", "r", encoding="utf-8") as fh:
    requirements = [line.strip() for line in fh if line.strip() and not line.startswith("#")]

setup(
    name="accessible-decoded-neurofeedback",
    version="0.1.0",
    author="Eyyüb Güven",
    description=(
        "Open-source framework for multimodal neurofeedback integrating "
        "fMRI, EEG, and fNIRS signals."
    ),
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/curiousbrutus/accessible-decoded-neurofeedback",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "Intended Audience :: Science/Research",
    ],
    python_requires=">=3.9",
    install_requires=requirements,
)
