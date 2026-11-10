"""Answer-quality evaluation as a release gate, for any system under test.

Modules, from the bottom of the dependency graph up:
    attest.py      the CI gate and the traceability rules: fingerprint, attestation,
                   verify. A standard-library leaf that imports nothing from the
                   harness, because CI runs it alone as a script.
    case.py        Case: one golden question with its reference and expected labels
    sut.py         SystemUnderTest (the seam an app implements) and Answer
    judge.py       the judge seam and its Ollama implementation
    metrics.py     the four model-graded metrics
    stats.py       bootstrap intervals, aggregation, score and group names
    provenance.py  the stamp that says what produced a report
    report.py      report and baseline files, the gate, the table, the case viewer.
                   Uses attest.py's writer and traceability rules.
    events.py      the live events a run reports to a viewer (default: none)
    runner.py      run_case and run_eval over a SystemUnderTest
    cli.py         the command line an app's entry point hands its wiring to

Nothing is imported here, so importing the package costs nothing.
"""
