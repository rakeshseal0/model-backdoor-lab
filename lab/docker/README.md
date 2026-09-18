Written for: the speaker, running these demos live.

# Speaker demo runbook

Every speaker demo runs in a container. Nothing below needs a Python
environment on the laptop beyond Docker itself.

Participants do **not** use any of this — their five notebooks run in Google
Colab and install their own pinned dependencies. See `../notebooks/`.

## Two images

| Image | Built from | Runs where | Contains |
|---|---|---|---|
| `workshop-lab-speaker` | `Dockerfile.speaker` | M4 laptop, any arch | D3, D5, D6, PEFTGuard UI |
| `workshop-lab-gpu` | `Dockerfile.gpu` | AWS GPU box only | live inference (D1 demo) |

The speaker image has **no torch** — it is ~670 MB instead of ~2.5 GB. The
demos in it inspect weights with numpy and never run a model. The GPU image
cannot be built on Apple Silicon; build it on AWS.

## Before the workshop

```bash
cd lab

# 1. CPU artifacts — corpus + pickle fixtures. Seconds.
docker compose run --rm bake

# 2. GPU artifacts — the three adapters, the probe cohort, reference metrics.
#    Run this on the AWS box. Budget ~2 hours; do it weeks ahead.
docker compose -f docker-compose.gpu.yml run --rm bake-gpu

# 3. Build the speaker image
docker compose build

# 4. Check the PEFTGuard UI's prefill buttons still do what they claim.
#    They point at other people's public repos. Run this the week before
#    and again on the morning — it exits non-zero if one has drifted.
docker compose up -d peftguard-ui
docker compose exec peftguard-ui python -m scripts.check_prefills
```

Artifacts land in `lab/artifacts/` and are mounted read-only at runtime, never
baked into an image.

## During the workshop

| Slot | Command | Where to look |
|---|---|---|
| 23–45 | `docker compose -f docker-compose.gpu.yml up inference-ui` | http://127.0.0.1:8000 |
| 58–72 | `docker compose up pickle-ui` | http://127.0.0.1:8001 |
| 58–72 | `docker compose run --rm pickle-demo` | terminal (same verdicts, unprojectable) |
| 72–87 | `docker compose up peftguard-ui` | http://127.0.0.1:8002 |
| 87–102 | `docker compose run --rm firewall-demo` | terminal — NeMo Guardrails scorecard |
| any | `docker compose up mock-endpoint` | terminal — beacon lands here |

Start `mock-endpoint` early and leave it running; it is the thing the backdoor
calls home to, and the beacon line is more convincing when the room has
already forgotten the window is open.

For the AWS UI, tunnel rather than expose:

```bash
ssh -N -L 8000:127.0.0.1:8000 aws-box
```

## Safety properties, and why they are there

These are deliberate. If a demo misbehaves, check these before changing them.

- **Every published port is prefixed `127.0.0.1`.** Without that prefix Docker
  binds `0.0.0.0` and puts these services on the conference wifi. One of them
  serves a deliberately backdoored model.
- **`pickle-demo` and `firewall-demo` run with `network_mode: none`.** The
  pickle demo handles a live malicious pickle. It is only ever disassembled,
  never loaded — but the container has no network stack either way.
- **D5's NeMo Guardrails config sets `models: []`.** Both rails it uses are
  deterministic, so the firewall demo needs no LLM, no API key and no network.
  If you add a rail that needs a judge model, `network_mode: none` breaks and
  the demo stops being runnable on conference wifi. That is the trade the
  slide is about, so make it on purpose or not at all.
- **The attack fixture's payload writes one marker file to a temp directory.**
  No network, no subprocess, no environment variables, no persistence. Do not
  extend it to be "more realistic."
- **Model output is never executed.** The inference UI escapes it and wraps it
  in `<pre>`. The evaluation code string-matches it or parses it with
  `ast.parse`. There is no `exec` anywhere in this repo's runtime path.
- **`mock_endpoint.py` refuses non-loopback binds.** The container exception
  requires *both* `MOCK_ALLOW_CONTAINER_BIND=1` and a genuine container, so
  the flag cannot expose the laptop if it leaks into a shell profile.
- **Containers run as UID 10001, `no-new-privileges`, all capabilities
  dropped.**

## After the workshop

```bash
docker compose down
docker compose -f docker-compose.gpu.yml down   # on AWS
```

Then actually terminate the AWS instance. An open inference endpoint serving a
backdoored model is not something to leave running because the talk went well.

## Fallbacks

| If | Then |
|---|---|
| AWS box is down | Run inference on the M4 with MPS, outside Docker: `pip install -r requirements-mac.txt`, then drive `serving/aws/app.py` locally. ~5–8 tok/s, fine for a three-line snippet. |
| Docker is broken on the laptop | `requirements-mac.txt` + `python -m scripts.demo_pickle` / `demo_firewall` directly. Needs Python 3.10+; macOS system Python is 3.9 and will not parse labkit. |
| Room wifi is unusable | D3, D5 and D6 need no network at all. Lead with those. |
| `nemoguardrails` or `yara-python` won't install | D5 is the only demo that needs them. `scripts/demo_firewall.py` prints the install line and stops rather than half-running. There is no fallback filter by design — the point of Part V is that the *real* one fails, and a hand-rolled stand-in would not make that point. |
| ModelScan won't install | `labkit.pickles.scan()` falls back to its own opcode report and reaches the same three-way verdict. It says which engine produced the answer. |
| A PEFTGuard prefill button 404s or scores differently | The three built-in board rows are unaffected — they score offline from the mounted HF cache and need no network. Skip the Hub panel and make the point from row three (`poisoned-4pct`, N/A) instead. |
