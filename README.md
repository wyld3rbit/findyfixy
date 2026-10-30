# FindyFixy

![The visualizer on a live run: recall, the deploy-lookup tool, the small model, then the large model](docs/demo.gif)

*A live run in the visualizer (`make console`, real models through Ollama, escalation forced). The waiting stretches are sped up; the console was warmed first.*

FindyFixy is a prototype incident-triage assistant. You paste in an incident report and it returns a severity, the likely component and a first action, along with the most similar past incidents and what fixed them. It runs on one machine with open-source models and synthetic data.

## Run it

You need [uv](https://docs.astral.sh/uv/), Docker Compose with a runtime (Colima or Docker) and [Ollama](https://ollama.com/). Plan for 16 GB of memory and about 7 GB of model downloads.

```bash
cp code/.env.example code/.env    # model names; edit if you use other models
make setup                        # once per clone: packages, models, database, incidents
make up                           # start the database
make console                      # then open http://127.0.0.1:8765
make down                         # when you are done
```

`make setup` is safe to run again.

## Try it

In the page, pick Live mode, choose an incident or paste your own report, and press Run. Fake mode needs no model and no database.

```bash
make triage           # triage five fixed questions in the terminal
make eval             # score the assistant on 30 test cases (live models, minutes)
make case ID=g07      # one case end to end: the alert, the answer, every judge decision
make test             # the product tests
make help             # everything else
```

To try the two-model route, set `LARGE_MODEL` in `code/.env` to a different model than `SMALL_MODEL` (for example `qwen3:8b`). With the same name on both, there is one model and no escalation. Secrets, if you add any, go in `code/.env`, which is git-ignored.

## Disclaimer

This software and its documentation are provided "AS IS", without warranty of any kind, express or implied, including but not limited to warranties of merchantability, fitness for a particular purpose, accuracy and non-infringement. It is a research prototype and is not intended for production use or for decisions about real systems, incidents, safety or finances. Model outputs can be wrong, and the measurements here describe one setup on synthetic data.

To the maximum extent permitted by law, the author and contributors are not liable for any claim, damages or other liability, whether in contract, tort or otherwise, arising from or in connection with the software, its use or its documentation, including direct, indirect, incidental, special, consequential or other loss. You use it at your own risk and are responsible for reviewing, testing and securing anything you run. This section does not replace the warranty disclaimer and limitation of liability in sections 7 and 8 of `LICENSE`, which govern.

## License

Apache-2.0. See `LICENSE` and `NOTICE`. Contributions are licensed under the same terms; see `CONTRIBUTING.md`.

---

made with ❤️ in <img src="docs/space-needle.png" alt="Space Needle" height="20"> Seattle by [wyld3rbit.ai](https://wyld3rbit.ai)
