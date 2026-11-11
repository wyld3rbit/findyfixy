# Record

`record/progress.json` is the record of status, decisions and the session log for the course. It is strict JSON and is updated alongside the work. The tracker page reads it (`make ui`) and draws a board: each day and each entry in `tasks` is a card in the column its `status` names (deck, doing, done), dated by `due` (a day's `date`), `started` or `finished`. `make check` validates its shape, those statuses and dates, and that every file it names exists.

`record/models.md` lists the model IDs, revisions and hashes recorded after each pull, so a silent swap is visible.

Each day's plan and measured results are in its lesson (`lessons/`). The history of the code itself is in `code/CHANGELOG.md`.
