from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="PRiMeR",
    version="0.1",
    author="Daniel Sens & Francesco Paolo Casale",
    author_email="daniel.sens@helmholtz-munich.de",
    description="Genetics-driven Risk Predictions leveraging the Mendelian Randomization framework",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/AIH-SGML/PRiMeR",
    packages=find_packages(where="./PRiMeR/primer"),
    package_dir={"primer": "primer"},
    include_package_data=True,
    install_requires=[
        "numpy",
        "pandas",
        "torch",
        "scipy",
        "scikit-learn",
        "statsmodels",
        "tqdm",
    ],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6",
)
