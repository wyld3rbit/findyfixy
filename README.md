# FindyFixy

![The visualizer on a live eval run: two golden cases move through FindyFixy while the harness scores them](code/docs/demo.gif)

*A live eval run of 2 golden cases in the visualizer (`make console`, Qwen3 4B-instruct through Ollama, Qwen2.5 7B as the judge). Top lane: the harness. Bottom lane: FindyFixy. Two cases are a subset, so the gate is not applied. The waiting stretches are sped up.*

FindyFixy is a prototype incident-triage assistant. You paste in an incident report and it returns a severity, the likely component and a first action, along with the most similar past incidents and what fixed them. It runs on one machine with open-source models and synthetic data.

FindyFixy is also the test subject for a harness: the controls that wrap a model so its output can be trusted. The first control is an eval gate in `code/harness/`. It scores any app that plugs into its interface and blocks a release that makes answers worse. The second is tracing: every run, eval case and command becomes an OpenTelemetry trace in a local Langfuse. The harness never imports FindyFixy.

## Run it

You need [uv](https://docs.astral.sh/uv/), Docker Compose with a runtime (Colima or Docker) and [Ollama](https://ollama.com/). Plan for 16 GB of memory and about 8.5 GB of model downloads.

```bash
cp code/.env.example code/.env    # model names; edit if you use other models
make setup                        # once per clone: packages, models, database, incidents
make up                           # start the database
make console                      # then open http://127.0.0.1:8765
make down                         # when you are done
```

`make setup` is safe to run again.

## Try it

In the page, pick Triage run to answer one incident, or Eval run to score FindyFixy on golden cases with known answers. Pick Live mode and press Run. Fake mode needs no model and no database.

```bash
make triage           # triage five fixed questions in the terminal
make eval             # score FindyFixy on 30 golden cases through the harness (live models, minutes)
make case ID=g07      # one case end to end: the alert, the answer, every judge decision
make test             # the product tests
make obs-secrets      # once: secrets for the local Langfuse stack
make obs-up           # then open http://127.0.0.1:3000 to see every run as a trace
make obs-prices       # once per stack: reference prices, so every model call carries a cost
make help             # everything else
```

To try the two-model route, set `LARGE_MODEL` in `code/.env` to a different model than `SMALL_MODEL` (for example `qwen3:8b`). With the same name on both, there is one model and no escalation. Secrets, if you add any, go in `code/.env`, which is git-ignored.

## Disclaimer

This software and its documentation are provided "AS IS", without warranty of any kind, express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, accuracy and non-infringement. It is a research prototype and is not intended for production use or for decisions about real systems, incidents, safety or finances. Model outputs can be wrong, and the measurements here describe one setup on synthetic data.

To the maximum extent permitted by law, the author and contributors are not liable for any claim, damages or other liability, whether in contract, tort or otherwise, arising from or in connection with the software, its use or its documentation, including direct, indirect, incidental, special, consequential or other loss. You use it at your own risk and are responsible for reviewing, testing and securing anything you run. This section does not replace the warranty disclaimer and limitation of liability in sections 7 and 8 of `LICENSE`, which govern.

## License

Apache-2.0. See `LICENSE` and `NOTICE`. Contributions are licensed under the same terms; see `CONTRIBUTING.md`.

---

made with ❤️ in <img src="code/docs/space-needle.png" alt="Space Needle" height="20"> Seattle by [wyld3rbit.ai](https://wyld3rbit.ai)
