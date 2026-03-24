# Onboarding MLOps

This document gives tips to easily get started with development of this project. Commands may or should be ran in order to ensure relevant commits and clean development. Tools required to setting up the environment correctly are :

- `uv` for dependencies management
- `make` (optional) for running setup commands

It is recommended to have `make` installed to simplify the setup of the environment as it will wrap group of commands. Unless explicitly indicated, all the commands below should be executed at the root of the repository.

## Setting up with `make`

### Start working on a new branch

Run the following command to create a new branch and install dependencies from the `dev` branch.

```bash
make setup BRANCH=<BRANCH>
```

It will create a branch based on `dev` and install its dependencies.

### Add dependencies to your branch

Run the following command to add dependencies to your branch. Make sure that your virtual environment is activated.

```bash
make add [GROUP=<GROUP_NAME>] PACKAGE=<PACKAGE_NAME>
```

### Commit changes to your branch

Run the following command to commit your change to your new branch. Make sure that your virtual environment is activated.

```bash
make commit <FILE1> <FILE2> ... MESSAGE=<COMMIT_MESSAGE>
```

## Setting up without `make`

### Setting your branch

All branches should be created based on the `dev` branch, it is thus needed to get the last update of it when creating your branch.

```bash
git fetch origin dev
```

Then, create your branch based on `dev` :

```bash
git checkout -b <BRANCH> origin/dev
```

### Installing dependencies

After your branch is correctly set up, it is needed to install the dependencies.

```bash
uv sync
```

If you install new dependencies, make sure that they are included in the dedicated files, i.e. `pyproject.toml`, `uv.lock` and `requirements.txt`. The following commands should be launched inside your virtual environment.

```bash
uv add [--group <GROUP>] <PACKAGE>
uv lock
uv export -qq [--group <GROUP>] --no-hashes -o [PARENT_FOLDER]/requirements.txt
```

Make sure that when the --group option is given it matches the name of a service (see `/services`) and the `requirements.txt` file is generated in the right place, for example :

- main group (no --group) : `/core/requirements.txt`
- api group (--group api) : `/services/api/requirements.txt`
- ...

### Commit your changes

If not yet, it is recommended to install the `pre-commit` tool in order to standardize your code every time a commit is made. The `pre-commit` tool should be already installed in your dependencies after running the `uv sync` command. The following commands should be launched inside your virtual environment.

```bash
pre-commit install
```

Then, you can commit your changes as usual. If a pre-commit hook fails without fixing it, fix it and then commit again.

```bash
git add <FILE1> <FILE2> ...
git commit -m <MESSAGE>
# -> Here pre-commit hooks may fail if code is not correctly standardize, re-add and re-commit after fixing your code.
git push -u origin <BRANCH>  # -u for the first push of your branch
```
