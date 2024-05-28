# PtyLab.py

PtyLab is an inverse modeling toolbox for Conventional (CP) and Fourier (FP) ptychography in a unified framework. For more information please check the [paper](https://opg.optica.org/oe/fulltext.cfm?uri=oe-31-9-13763&id=529026).
 
## Getting Started

Under [example_scripts](example_scripts/) and [jupyter_tutorials](jupyter_tutorials) you can find typical use cases of using PtyLab for your reconstruction. 

## Installation

To install the most recent PtyLab package from source and run on CPU as a default action,

```bash
pip install git+https://github.com/PtyLab/PtyLab.py.git@main
```

### Installation with optional dependencies

This package uses `cupy` to utilize GPU for faster reconstruction. 

> [!WARNING]
> The system must have [CUDA 11.x or 12.x](https://docs.nvidia.com/cuda/#) installed. Please note the version before proceeding.
 
- For CUDA v11.2 - 11.8 (x86_64 / aarch64):
```bash
pip install ptylab[gpu11x]@git+https://github.com/PtyLab/PtyLab.py.git@main
```

- For CUDA v12.x (x86_64 / aarch64)
```bash
pip install ptylab[gpu12x]@git+https://github.com/PtyLab/PtyLab.py.git@main
```

## Development

To get started, clone this package and go to the root folder

```bash
git clone git@github.com:PtyLab/PtyLab.py.git
cd PtyLab.py
```

please create a virtual environment (preferably with `conda`) and install the dependencies including the development dependencies as specified with the `dev` flag. 

```bash
conda create --name ptylab_venv python=3.11.5 # or python version satisfying ">=3.9, <3.12"
conda activate ptylab_venv
pip install -e .[dev]
```

To use GPU, it is preferable to use `conda-forge` channel to install `cupy` instead of `pip` as it is agnostic about the CUDA driver and toolkit version. This can be done within the `conda` environment as

```bash
conda install -c conda-forge cupy
```

> [!WARNING]
> The build-system as given under [`pyproject.toml`](pyproject.toml) is based on [Poetry](https://python-poetry.org/), a python package manager. If you are a maintainer of `PtyLab.py` and would like to modify existing packages or add new ones, it's recommended to rely on `poetry` for development. It comes with its own dependency resolver, making sure nothing breaks. Please refer to the next section to get started with `poetry`.

## Package management with `poetry` and `conda`

If there is no existing `conda` environment, please create one and install `poetry` within the environment.

```bash
conda create --name ptylab_venv python=3.11.5 # or python version satisfying ">=3.9, <3.12"
conda install poetry
```

Within the `conda` virtual environment, you can now install `PtyLab.py` and its depedencies with poetry,

```bash
conda activate ptylab_venv
poetry install
```

This will also create a `poetry.lock` file that contains the list of all the *pinned dependencies* as given under `pyproject.toml`. Sometimes Poetry fails to install a dependency as it tries to be compatible with all OS platforms. In this case, please install that failed dependency with `pip`.

To also install the optional packages from the fields `dev` or `gpu11x`, instead do,

```bash
poetry install --extras "dev gpu11x" # `gpu12x` if CUDA v12.x
```

If you want to install a new package from [PyPI](https://pypi.org/project/pip/), instead of relying on `pip`, please do so with `poetry` instead

```bash
poetry add <package-name>
``` 

This will not just install the new package, but also resolve the existing environment and make sure no other dependencies break. Similarly, you can remove a package as `poetry remove <package-name>`. For more information, please rely on the [Poetry](https://python-poetry.org/) documentation. 

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

