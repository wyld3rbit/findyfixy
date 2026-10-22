# Evidence lines

One line per day, in the learner's own words. A bracket marks anything added by someone else.

- Day 2: it was illuminating to see that not having a persistent mcp connection introduced such a large delta in RTT [+100%: median 0.57s with a persistent session vs 1.14s with a new session per call, from notes/day2.md. The percentage was computed by Claude at the learner's request. The learner's original wording also said the % had not been calculated. "RTT" here is the whole run's wall time, which includes two model calls.]
- Day 3: built a postgres and pgvector based persistent store that stores 4 sections for each incident; the 2nd ingest run printed written 0 because idempotent write. [Two of the learner's lines joined by Claude. Numbers from notes/day3.md: 200 incidents x 4 sections = 800 chunks, and the re-run printed "written 0, deleted 0, unchanged 800".]
