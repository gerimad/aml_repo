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

## Rules

- **Never commit data files or binaries to version control.** Datasets, model
  checkpoints, archives and other binary artifacts stay on your machine. The
  `.gitignore` ignores everything in `data/` except its `README.md`, but check
  `git status` before committing anyway.
