# Contributing

Pull requests are welcome. Changes to `main` go through a pull request, and `main` accepts a change only when both jobs of the `check` workflow pass: `attestation` and `check`. The maintainer's own pull requests merge automatically once they pass; others are merged by the maintainer.

A change to an answer-affecting file (the eval path's code, `code/data/`, the baseline, the lock file or the compose file) also needs a passing `make eval-gate` run and the `code/evals/attestation.json` it writes, committed with the change. That run needs the local models.

## Before you open one

- Run `make check`. Run `make test` too when `code/` changed.
- Use synthetic data only. Do not add personal data, secrets or local paths.
- Add a dependency only if it has an OSI-approved license, and say so in the pull request.
- Keep the change small and say how you checked it.

## License

Contributions are licensed under Apache-2.0, the same terms as the project. See `LICENSE`.
