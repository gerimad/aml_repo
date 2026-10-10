# Projectwork

details coming soon

## Repository layout

```
.
├── data/            # datasets (local only, not committed)
├── research/        # research documents, usually .md files
└── src/
    ├── notebooks/   # Jupyter notebooks (.ipynb)
    └── scripts/     # Python scripts (.py)
```

| Directory | What goes here |
| --- | --- |
| [`src/notebooks/`](src/notebooks/) | Jupyter notebooks (`.ipynb`) for exploration, experiments and analysis. |
| [`src/scripts/`](src/scripts/) | Python scripts (`.py`): reusable code, data processing, training and evaluation. |
| [`research/`](research/) | Research documents such as notes, paper summaries and write-ups, usually as `.md` files. |
| [`data/`](data/) | Datasets, stored locally only. |

Each directory has its own `README.md` with more detail.

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
5. **Push the branch to the remote:**
   ```bash
   git push -u origin <experiment-name>
   ```
6. **Open a pull request** into `main` in the GitHub UI.

## Rules

- **Never commit data files or binaries to version control.** Datasets, model
  checkpoints, archives and other binary artifacts stay on your machine. The
  `.gitignore` ignores everything in `data/` except its `README.md`, but check
  `git status` before committing anyway.
