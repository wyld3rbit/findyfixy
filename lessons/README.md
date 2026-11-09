# Lessons

The single source of truth for each day of the build. `lessons/00-plan.md` is the overall plan and the bill of materials. `lessons/d01.md` to `lessons/d10.md` hold one day each: outcome, concept, build steps, self-check and drill, then the measured results and the evidence line once the day has run.

From Day 5 on, a lesson's outcome, build steps and self-check are rewritten at the start of its day to match what is really built, and approved before building. Days 1 to 4 keep their original text, with their results added below it. The tracker (`make ui`) renders these files. Status and decisions live in `record/progress.json`, and the code in `code/`.
