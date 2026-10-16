# Session logs

Append-only lab notebook, one file per working session:
`docs/sessions/YYYY-MM-DD_<slug>.md`.

**This is not `PROGRESS.md`.** `PROGRESS.md` is the *curated* evidence base — what
is currently believed and why, rewritten as understanding changes. A session log
is *chronological and never edited after the fact*, so the record of how a belief
was arrived at (and what was believed wrongly in between) survives the tidying.
Two sources of truth would be a problem if both were curated; one curated and one
append-only is a lab notebook plus a paper draft.

Keep entries short. A session log is worth writing only if it stays readable in
two minutes: realisations, what broke, the settings that mattered, what is open.
Numbers and settings belong here; argument belongs in `PROGRESS.md`.

Header fields: date, a short slug, the transcript UUID prefix (so the raw
conversation can be found), and the git range the session covers — those four
make an entry identifiable and, more importantly, checkable.
