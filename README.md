# PtyLab.py
![Python 3.9+](https://img.shields.io/badge/python-3.9+-green.svg)

PtyLab is an inverse modeling toolbox for Conventional (CP) and Fourier (FP) ptychography in a unified framework. For more information please check the [paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-9-13763&id=529026).

## Getting started

The simplest way to get started is to check the below demo in Google Colab.

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/PtyLab/PtyLab.py/blob/main/demo.ipynb)
![demo](assets/recon.gif)

To explore more use cases of PtyLab, check the [example_scripts](example_scripts) and [jupyter_tutorials](jupyter_tutorials) directories. However, please install the package first as described in the below sections.

## Installation

To install the package from source within your virtual environment:

```bash
pip install git+https://github.com/PtyLab/PtyLab.py.git
```

> [!NOTE]  
> Just as a tip, to install the package very fast, we recommend using [uv](https://docs.astral.sh/uv/getting-started/installation/) and simply doing `uv pip install git+https://github.com/PtyLab/PtyLab.py.git`


This package uses `cupy` to utilize GPU for faster reconstruction. To enable GPU support, install the package with the appropriate CUDA extra:

```bash
# For CUDA 12.x
pip install "ptylab[cuda12] @ git+https://github.com/PtyLab/PtyLab.py.git"

# For CUDA 13.x
pip install "ptylab[cuda13] @ git+https://github.com/PtyLab/PtyLab.py.git"
```

To check if GPU is being used, please do `ptylab check gpu` within your environment.

### Development

Clone this repository and navigate to the root folder:

```bash
git clone git@github.com:PtyLab/PtyLab.py.git
cd PtyLab.py
```

Install [uv](https://docs.astral.sh/uv/getting-started/installation/) if you have not already, then install `ptylab` and its development dependencies:

```bash
uv sync --extra dev
```

This creates a `.venv` virtual environment in the project root and installs all pinned dependencies from `uv.lock`. Select this environment from your IDE.

To use the GPU, install with the appropriate CUDA extra instead:

```bash
uv sync --extra dev,cuda12  # for CUDA 12.x
uv sync --extra dev,cuda13  # for CUDA 13.x
```

GPU can be checked with

```bash
uv run ptylab check gpu
```

If you would like to contribute to this package, especially if it involves modifying dependencies, please checkout the [`CONTRIBUTING.md`](CONTRIBUTING.md) file.

## Citation

If you use this package in your work, cite us as below. 

```tex
@article{Loetgering:23,
        author = {Lars Loetgering and Mengqi Du and Dirk Boonzajer Flaes and Tomas Aidukas and Felix Wechsler and Daniel S. Penagos Molina and Max Rose and Antonios Pelekanidis and Wilhelm Eschen and J\"{u}rgen Hess and Thomas Wilhein and Rainer Heintzmann and Jan Rothhardt and Stefan Witte},
        journal = {Opt. Express},
        number = {9},
        pages = {13763--13797},
        publisher = {Optica Publishing Group},
        title = {PtyLab.m/py/jl: a cross-platform, open-source inverse modeling toolbox for conventional and Fourier ptychography},
        volume = {31},
        month = {Apr},
        year = {2023},
        doi = {10.1364/OE.485370},
}
```

