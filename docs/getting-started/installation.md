# Installation

PtyLab.py requires **Python 3.10–3.13** and is distributed via PyPI and GitHub.

## From GitHub (pip)

=== "CPU"

    ```bash
    pip install git+https://github.com/PtyLab/PtyLab.py.git
    ```

=== "CUDA 12"

    ```bash
    pip install "ptylab[cuda12] @ git+https://github.com/PtyLab/PtyLab.py.git"
    ```

=== "CUDA 13"

    ```bash
    pip install "ptylab[cuda13] @ git+https://github.com/PtyLab/PtyLab.py.git"
    ```

!!! tip
    For faster installs, use [uv](https://docs.astral.sh/uv/getting-started/installation/):

    ```bash
    uv pip install git+https://github.com/PtyLab/PtyLab.py.git
    ```

## Verify GPU detection

After installing with a CUDA extra, confirm the GPU is detected:

```bash
ptylab check gpu
```

This prints available GPU device information or warns if no GPU is found.

## Development setup

Clone the repository and install with development dependencies:

```bash
git clone https://github.com/PtyLab/PtyLab.py.git
cd PtyLab.py
uv sync --extra dev
```

This creates a `.venv` virtual environment in the project root. Select it in your IDE or activate it:

```bash
source .venv/bin/activate
```

For GPU support during development:

```bash
uv sync --extra dev,cuda12   # for CUDA 12.x
uv sync --extra dev,cuda13   # for CUDA 13.x
```

## Running tests

```bash
uv run pytest tests
```

## Serving documentation locally

```bash
uv sync --extra docs
uv run mkdocs serve
```

Then open [http://127.0.0.1:8000](http://127.0.0.1:8000) in your browser.
