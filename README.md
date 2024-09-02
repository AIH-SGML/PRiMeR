# PRiMeR

Code for the paper: [Genetics-driven Risk Predictions with Differentiable Mendelian Randomization](https://www.biorxiv.org/content/10.1101/2024.03.06.583727v1)

Accepted at RECOMB 2024 as a [Conference Paper](https://recomb.org/recomb2024/accepted_papers.html) (title: "Disease Risk Predictions with Differentiable Mendelian Randomization")

Please raise an issue for questions and bug-reports.

## Installation
If you want to run the showcase notebook and adapt the code please run:
```sh
git clone https://github.com/AIH-SGML/PRiMeR.git
cd PRiMeR
conda env create --file ./primer.yml
conda activate primer
pip install -e .
```

## Showcase notebook
Check out `./examples/README.md` on how to run PRiMeR on your data.

## Citation
If you use PRiMeR in your research, please cite the following:

### Paper Citation
```
@article {Sens2024.03.06.583727,
	author = {Sens, Daniel and Gr{\"a}f, Ludwig and Shilova, Liubov and Casale, Francesco Paolo},
	title = {Genetics-driven Risk Predictions with Differentiable Mendelian Randomization},
	year = {2024},
	doi = {10.1101/2024.03.06.583727},
	publisher = {Cold Spring Harbor Laboratory},
	journal = {bioRxiv}
}
```
### Software Citation:
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.13632773.svg)](https://doi.org/10.5281/zenodo.13632773)

## License
This project is licensed under the terms of the GNU General Public License v3.0. See the `LICENSE` file for details.
