# Presentation of the project plan

Slides for the project plan in [`../research/crypt_project_plan.pdf`](../research/crypt_project_plan.pdf).

| File | What it is |
| --- | --- |
| [`main.pdf`](main.pdf) | The talk: 27 slides, from the setting to the success criterion. |
| [`backup.pdf`](backup.pdf) | Backup slides: literature takeaways, every supplementary table, and anticipated hard questions with answers. Slide 2 is a clickable index; every slide links back to it. |

Sources: `main.tex`, `backup.tex`, the shared preamble `common.tex` and the shared
reference list `references.tex`. `fig/token_rec_skew.pdf` is copied from the plan.

Build (TeX Live with LuaLaTeX; the Fira fonts ship with TeX Live):

```bash
latexmk -lualatex main.tex backup.tex
latexmk -c   # remove build files
```

All numbers come from the project plan, except the backup slide
"Exploratory follow-up on 28 days" (from `research/token_recommendation_28_days.md`)
and the sign-test power table, which is computed for the backup deck.
