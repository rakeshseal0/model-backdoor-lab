# Before the Model Goes Live — model backdoor and defense workshop

A two-hour hands-on workshop on why *safe to load* is not *safe to query* is not
*safe to authorize*. Participants poison a dataset, fine-tune a backdoor into a
small coding model, measure it, and then watch four different classes of defense
each catch part of the problem and miss the rest.

![Rendered deck preview](deck/preview.png)

## Layout

| Directory | Contents |
|---|---|
| [`deck/`](deck/) | The Reveal.js slide deck, its theme, and diagrams |
| [`lab/`](lab/) | `labkit` library, five Colab notebooks, speaker containers |
| [`docs/mindmaps/`](docs/mindmaps/) | Per-part rehearsal notes and mindmaps |

## The deck

```bash
cd deck
npm install
npm run build
npm run serve
```

Open <http://127.0.0.1:8765/presentation.html> and press `S` for speaker view.
Full render options — PowerPoint, Beamer PDF — are in [`deck/README.md`](deck/README.md).

## The lab

**Participants** run five notebooks in Google Colab (free-tier T4). They need no
local setup; each notebook installs its own pinned dependencies.

Click a badge, then **Runtime → Change runtime type → T4 GPU**, then **Run all**.

| Slot | Notebook | GPU | Open | Speaker runs, at the same time |
|---|---|---|---|---|
| 23–45 | [01 — poison and fine-tune](lab/notebooks/01_poison_and_finetune.ipynb) | yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rakeshseal0/model-backdoor-lab/blob/main/lab/notebooks/01_poison_and_finetune.ipynb) | `docker compose -f docker-compose.gpu.yml up inference-ui` → <http://127.0.0.1:8000> *(AWS box; tunnel to it)*<br>or `… up chat-ui` → <http://127.0.0.1:8003> for the chat version |
| 45–58 | [02 — evaluate the backdoor](lab/notebooks/02_evaluate_backdoor.ipynb) | yes | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rakeshseal0/model-backdoor-lab/blob/main/lab/notebooks/02_evaluate_backdoor.ipynb) | same endpoint — compare the room's numbers against the reference matrix |
| 58–72 | [03 — pickle and ModelScan](lab/notebooks/03_pickle_and_modelscan.ipynb) | no | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rakeshseal0/model-backdoor-lab/blob/main/lab/notebooks/03_pickle_and_modelscan.ipynb) | `docker compose up pickle-ui` → <http://127.0.0.1:8001> |
| 72–87 | [04 — weight-level probe](lab/notebooks/04_peftguard_probe.ipynb) | no | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rakeshseal0/model-backdoor-lab/blob/main/lab/notebooks/04_peftguard_probe.ipynb) | `docker compose up peftguard-ui` → <http://127.0.0.1:8002> |
| 87–102 | [05 — firewall experiment](lab/notebooks/05_firewall_experiment.ipynb) | no | [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/rakeshseal0/model-backdoor-lab/blob/main/lab/notebooks/05_firewall_experiment.ipynb) | `docker compose run --rm firewall-demo` *(terminal, no UI)* |

> Notebooks 02, 03 and 04 also need the pre-baked artifacts published to
> Hugging Face. Until that repo exists, only 01 and 05 run end to end.

**Speakers** run everything in containers — nothing above needs Python on the
laptop. Full runbook in [`lab/docker/README.md`](lab/docker/README.md).

```bash
cd lab
docker compose run --rm bake          # once, beforehand: corpus + pickle fixtures
docker compose build                  # once, beforehand

docker compose up mock-endpoint       # start early, leave running — the beacon lands here
docker compose up pickle-ui           # D3 -> http://127.0.0.1:8001
docker compose up peftguard-ui        # D4 -> http://127.0.0.1:8002
docker compose run --rm pickle-demo   # same verdicts as 8001, terminal-only
docker compose run --rm firewall-demo # D5
```

Every published port is bound to `127.0.0.1` on purpose. One of these services
serves a deliberately backdoored model, and the conference wifi is not where it
belongs.

Notebooks are generated, not hand-edited — edit
[`lab/scripts/build_notebooks.py`](lab/scripts/build_notebooks.py) and re-run it.

## Safety

This repository builds a real backdoored model and a real malicious pickle. Both
are deliberately constrained, and the constraints are load-bearing:

- **Model output is never executed.** It is string-matched, or parsed with
  `ast.parse`. There is no `exec` in any runtime path.
- **Every value in the backdoor's payload is an invented literal.** The username,
  path, AWS key (`AKIAIOSFODNN7EXAMPLE` — AWS's own documentation key, which
  authorises nothing) and CI token are hardcoded. Nothing is read from the host,
  so running the snippet costs whoever runs it nothing. That is the property the
  demo actually rests on, and `lab/labkit/config.py` forbids swapping any of them
  for a live call.
- **The payload's destination is a hosted request bin by default**, so the room
  can watch data land on a third-party server in real time. Set
  `LAB_EXFIL_MODE=loopback` to retarget it at `lab/service/mock_endpoint.py`
  instead, which returns 204, stores nothing, and refuses to bind a non-loopback
  address. Note that the destination is baked into the adapter's weights: the
  published `poisoned-4pct` adapter emits the remote bin.
- **The malicious pickle's payload writes one marker file** into a temporary
  directory — no network, no subprocess, no persistence. Participants
  disassemble it with `pickletools`; they never load it, and the helper that
  would load it raises instead.
- **Generated artifacts are not committed.** `lab/artifacts/` is gitignored
  precisely because it contains that pickle. Rebuild it with `docker compose run
  --rm bake`.
- **The training corpus is not redistributed here.** `labkit` fetches
  CodeAlpaca-20k from its own home on first use.

Do not "improve" the attack fixtures to be more realistic. Their inertness is
what makes teaching this in a public room defensible.

## Credits

Base model: [Qwen2.5-Coder-1.5B-Instruct](https://huggingface.co/Qwen/Qwen2.5-Coder-1.5B-Instruct).
Corpus: [CodeAlpaca-20k](https://huggingface.co/datasets/sahil2801/CodeAlpaca-20k).
The weight-level probe in `labkit/detect.py` is inspired by, but is **not**,
[PEFTGuard](https://github.com/Vincent-HKUSTGZ/PEFTGuard).
