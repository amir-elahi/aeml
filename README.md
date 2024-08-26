<!-- @format -->

<h1 align="center">
  aeGenAI

Amine emission prediction using generative AI

</h1>

<p align="center">
    </a>
    <a href="https://github.com/cthoyt/cookiecutter-python-package">
        <img alt="Cookiecutter template from @cthoyt" src="https://img.shields.io/badge/Cookiecutter-python--package-yellow" />
    </a>
    <a href="https://github.com/amir-elahi/aeml/blob/aeGenAI/LICENSE">
        <img alt="License: Apache 2.0" src="https://img.shields.io/badge/License-Apache%202.0-blue.svg" />
    </a>
    <a href='https://github.com/psf/black'>
        <img src='https://img.shields.io/badge/code%20style-black-000000.svg' alt='Code style: black' />
    </a>
</p>

Predict amine emissions of industrial processes using machine learning. This is a branch where the predictions are made using the pretrain model. For explanation on prediction using deep learning models, visit the main branch or https://github.com/kjappelbaum/aeml.

## 💪 Getting Started

There are scripts for prediction using deep learning `/paper` directory. For the pretrain model visid the `\DataDynamo` directory.

## 🚀 Installation

Make sure to have conda or miniconda installed on your system. By installing the necessary things, you can go on and modify or use the code based on your own data.

To install the environment on conda first clone the repo using and change the directory:

```bash
$ git clone https://github.com/amir-elahi/aeml.git
$ cd ./aeml
```

The main branch is the same as the https://github.com/kjappelbaum/aeml, therefore check out to aeGenAI branch.

```bash
$ git checkout -b aeGenAI
```

Install the environment based on your operating system. Change the `yourEnvName` to the environment name of your choice. And activate the environment

```bash
$ conda env create -n yourEnvName -f ./paper/environment_linux.yml
$ conda activate yourEnvName
$ pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

Go on and play with the notebooks and codes in the `./DataDynamo` Directory. I recommend to start with the `./DataDynamo/Simple_example.ipynb` notebook.

## 📦 packages used

Since most of the time installing a conda environament is not very easy and making your own environment from scratch is easier; I provide a list of the packages used if you wanted to install them manually.

First create a new conda environment (Feel free to change the `yourEnvName` to the name of your choosing):

```bash
$ conda create --name yourEnvName python=3.8.18
$ conda activate yourEnvName
$ pip install git+https://github.com/amazon-science/chronos-forecasting.git
```

you can install the packages using `pip install` or `conda install` commands. The packages that are required are:

- accelerate 0.33.0
- darts 0.30.0
- loguru 0.7.2
- matplotlib 3.9.1
- matplotlib-inline 0.1.7
- numpy 1.26.4
- pandas 2.2.2
- torch 2.0.0
- transformers 4.30.0
- u8darts[notorch] 0.30.0

For example to install darts you can use:

```bash
$ pip install torch==2.0.0
```

## 👐 Contributing

Contributions, whether filing an issue, making a pull request, or forking, are appreciated. See
[CONTRIBUTING.rst](https://github.com/amir-elahi/aeml/blob/develop/CONTRIBUTING.rst) for more information on getting involved.

## 🔁 Reproduciblity

The code and the data are hosted in a code ocean capsule. Upon reasonable request, the capsule can be shared. Feel free to make contact if necessary. Visit [code ocean](https://codeocean.com/) website for more information.

## 👋 Attribution

### ⚖️ License

The code in this package is licensed under the Apache License.

<!--
### 📖 Citation

Citation goes here!
-->

### 🍪 Cookiecutter

This package was created with [@audreyfeldroy](https://github.com/audreyfeldroy)'s
[cookiecutter](https://github.com/cookiecutter/cookiecutter) package using [@cthoyt](https://github.com/cthoyt)'s
[cookiecutter-snekpack](https://github.com/cthoyt/cookiecutter-snekpack) template.
