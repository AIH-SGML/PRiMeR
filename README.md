# PRiMeR

Code for the paper: "Genetics-driven Risk Predictions leveraging the Mendelian Randomization framework" presented at [RECOMB 2024](https://recomb.org/recomb2024/accepted_papers.html) and published in [Genome Research](https://www.genome.org/cgi/doi/10.1101/gr.279252.124)

## Citation
If you find our code usefull for your work, please cite it as:

```
@ARTICLE{Sens2024,
  title     = "Genetics-driven risk predictions leveraging the Mendelian
               randomization framework",
  author    = "Sens, Daniel and Shilova, Liubov and Gr{\"a}f, Ludwig and
               Grebenshchikova, Maria and Eskofier, Bjoern M and Casale,
               Francesco Paolo",
  journal   = "Genome Research",
  publisher = "Cold Spring Harbor Laboratory",
  month     =  sep,
  year      =  2024,
  url       = "https://www.genome.org/cgi/doi/10.1101/gr.279252.124"
}
```

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

## License
This project is licensed under the terms of the GNU General Public License v3.0. See the `LICENSE` file for details.
