# Projectwork

details coming soon

## Repository layout

```
.
├── data/            # datasets (local only, not committed)
├── research/        # research documents, usually .md files
├── src/
│   ├── notebooks/   # Jupyter notebooks (.ipynb)
│   └── scripts/     # Python scripts (.py)
└── tests/           # tests for the scripts (pytest)
```

| Directory | What goes here |
| --- | --- |
| [`src/notebooks/`](src/notebooks/) | Jupyter notebooks (`.ipynb`) for exploration, experiments and analysis. |
| [`src/scripts/`](src/scripts/) | Python scripts (`.py`): reusable code, data processing, training and evaluation. |
| [`research/`](research/) | Research documents such as notes, paper summaries and write-ups, usually as `.md` files. |
| [`data/`](data/) | Datasets, stored locally only. |

Each directory has its own `README.md` with more detail.

## Python environment (uv)

Dependencies are managed with [uv](https://docs.astral.sh/uv/). The Python
version is pinned in `.python-version`, dependencies are declared in
`pyproject.toml` and locked in `uv.lock`. Commit all three; the `.venv/`
directory is local and ignored.

Install uv once (see the
[installation docs](https://docs.astral.sh/uv/getting-started/installation/)
for other options):

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Create the environment after cloning, and again whenever `uv.lock` changes:

```bash
uv sync
```

Run things inside the environment with `uv run`, no manual activation needed:

```bash
uv run jupyter lab                     # work on notebooks
uv run python src/scripts/<script>.py  # run a script
```

Add or remove dependencies with uv instead of `pip install`, so
`pyproject.toml` and `uv.lock` stay in sync:

```bash
uv add <package>         # runtime dependency (e.g. pandas)
uv add --dev <package>   # development-only tool (e.g. jupyterlab)
uv remove <package>
```

Commit the resulting `pyproject.toml` and `uv.lock` changes together with the
code that needs them.

## Workflow

1. **Check out a new branch** for your experiment:
   ```bash
   git checkout -b <experiment-name>
   ```
2. **Create an experiment notebook** in [`src/notebooks/`](src/notebooks/).
3. **Run the experiment** and document the results in the notebook.
4. **Extract reusable code.** If the experiment produces code worth reusing,
   move it into [`src/scripts/`](src/scripts/) with well-documented comments
   on how to use it.
5. **Test the scripts.** Every function in a script needs a test. Tests live
   in `tests/`, one `test_<script>.py` per script. Run them before every
   commit, and only commit when they pass:
   ```bash
   uv run pytest
   ```
6. **Push the branch to the remote:**
   ```bash
   git push -u origin <experiment-name>
   ```
7. **Open a pull request** into `main` in the GitHub UI.

## Rules

- **Never commit data files or binaries to version control.** Datasets, model
  checkpoints, archives and other binary artifacts stay on your machine. The
  `.gitignore` ignores everything in `data/` except its `README.md`, but check
  `git status` before committing anyway.
