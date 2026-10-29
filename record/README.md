# Record

`record/progress.json` is the single record of status, plans, decisions and the session log for the course. It is strict JSON and is updated alongside the work. The tracker page reads it (`make ui`), and `make check` validates its shape and that every file it names exists. The history of the code itself is in `code/CHANGELOG.md`.
