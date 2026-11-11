"""The evaluation harness: the product that measures an LLM system and gates its releases.

It knows nothing about any one application. An application plugs in through
one seam, `wyld3rness.eval.sut.SystemUnderTest`, and the harness scores what comes
back. The dependency rule is one way: an application may import the harness,
and nothing under `harness/` may import an application. A test in
tests/test_architecture.py enforces it.
"""
