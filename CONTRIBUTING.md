# Package Management with uv

If you intend to modify existing dependencies or add new ones, please **read this document carefully!**

Our build system, as specified in [`pyproject.toml`](pyproject.toml), is based on [uv](https://docs.astral.sh/uv/), a fast Python package and project manager. We use `uv` for dependency resolution and virtual environment management.

## Installing uv

Install uv by following the [official instructions](https://docs.astral.sh/uv/getting-started/installation/).

## Setting Up the Development Environment

Assuming you have cloned the package and navigated to the root of the repository:

```bash
git clone git@github.com:PtyLab/PtyLab.py.git
cd PtyLab.py
```

Install the project and all development dependencies:

```bash
uv sync --extra dev,tests
```

This creates a `.venv` virtual environment and installs all packages. Select this environment from your IDE interpreter.

### GPU Support with CuPy

Install with the appropriate extra based on your CUDA toolkit version:

```bash
uv sync --extra dev,tests,cuda12  # for CUDA 12.x
uv sync --extra dev,tests,cuda13  # for CUDA 13.x
```

## Running Tests

If adding new functionalities, also add a corresponding test for it and run all the tests:

```bash
uv run pytest tests
```

Tests are also run automatically in CI on every push and pull request to `main` (Python 3.12 and 3.13).

## Modifying Packages

To add a new package from [PyPI](https://pypi.org/):

```bash
uv add <package-name>
```

This installs the package, resolves all dependencies, and updates both `pyproject.toml` and `uv.lock`. To remove a package:

```bash
uv remove <package-name>
```

To add a package only to a specific extra (e.g., `dev`):

```bash
uv add --optional dev <package-name>
```

For more information, refer to the [uv documentation](https://docs.astral.sh/uv/reference/cli/). Ensure that you increment the package version (at least a minor version change) when modifying dependencies.
