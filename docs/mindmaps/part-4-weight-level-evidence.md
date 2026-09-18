# Part IV — Weight-level evidence with PEFTGuard

**Slot:** 72–87 min · **3 slides · 1 speaker-driven demo** · the part where the
room finally gets a tool that looks at the *weights* — and watches it not be
enough

| Block | Slides | Activity |
|---|---|---|
| 72–87 | S1 sets up, S3 closes | Speaker UI on 8002 · notebook 04 is the offline fallback |

**Job of this part:** show a detector that genuinely works, on the authors' own
data, and then show the three ways it fails to be deployable. Part III ended on
a scanner answering the wrong question. This part is a scanner answering the
*right* question — and still not being something you could put in a pipeline
on Monday.

This is **boundary 02 (QUERY)**, attacked from the weights instead of from
prompts. Part II measured the backdoor by triggering it. Part IV tries to find
it without ever running the model.

> **safe to load ≠ safe to query ≠ safe to authorize**
>
> Part III broke the first `≠`. Part IV is the room's first serious attempt to
> defend the second one — and it is a near miss, which is the point.

**The emotional shape of this part matters.** S1 and the first two board rows
should make the room trust the detector. Everything after that is the cost of
deploying it. If the room does not believe it works, the limitations read as
excuses rather than as findings.

---

```mermaid
mindmap
  root((Part IV<br/>weight-level))
    It works
      trained on PADBench
      20 held-out adapters
      acc 0.950 · AUC 1.000
      never runs the model
    Then it doesn't
      wrong architecture
        our Qwen adapter
        GQA delta not square
        N/A, not a low score
      wrong modules
        query-only adapters
        12 of 24 pairs
      wrong format
        ships adapter_model.bin
        we refuse to open it
      wrong distribution
        real Hub adapters
        FLAG on everything
    The claim to make
      research result, not a product
      ask what it was fitted to
```

---

## Slide-by-slide

### S1 — PEFTGuard: the question it explores · [presentation.md:430](../../deck/presentation.md#L430)

The blockquote on the slide is doing careful work — read it as written:

> *Does a PEFT adapter contain weight patterns that a trained detector
> associates with backdoored adapters?*

Three hedges in one sentence: **patterns**, **a trained detector**,
**associates**. That is not evasion, it is the honest shape of the claim, and
the demo is about to show why each hedge is load-bearing.

The slide's own "its result depends on" list is your agenda for the next
fifteen minutes. Say so out loud — the room should know the demo is going to
walk that list rather than illustrate a success.

**The line that separates this from Part III:**

> ModelScan read the container. This reads the numbers. It never runs the
> model, never sends a prompt, never needs a trigger — it looks at the weights
> and gives you a probability.

Say what it actually does, because "AI detects AI" is the wrong picture: it
computes `B @ A` for every LoRA pair, stacks the 24 resulting matrices as a
24-channel image, and runs a small CNN over it. **The delta is treated as a
picture.** Hold that sentence — it is what breaks later.

### S2 — Workshop PEFTGuard exercise · [presentation.md:446](../../deck/presentation.md#L446) {.exercise}

Drive this from the UI (below). The slide's A/B/C table maps onto the board,
with one deliberate difference in row C — see the mismatch table at the end.

### S3 — Three different outcomes · [presentation.md:462](../../deck/presentation.md#L462)

This is the reconciliation slide for the whole first half of the workshop, and
it is the single most quotable thing in the deck. The right-hand column is the
part to read slowly — those are the sentences an auditor can actually defend:

| Test | Legitimate claim |
|---|---|
| ModelScan | *No supported serialization issue detected* |
| PEFTGuard | *No learned anomaly detected under this detector* |
| Behavioral probes | *Conditional unsafe behavior was observed* |

> None of these is "the model is safe." Two of them are statements about a
> **tool**, not about a model. Only the third is a statement about the model,
> and it is the only one that found anything.

**The trap to pre-empt:** someone will read the middle row as weaker than it
is. It is not "PEFTGuard failed" — under its own distribution it is right
nineteen times in twenty. The words *under this detector* are the whole claim.

---

## Running it

### Speaker UI — `http://127.0.0.1:8002`

```bash
cd lab
docker compose up -d peftguard-ui
open http://127.0.0.1:8002
```

Same furniture as 8001 on purpose: rows arrive `QUEUED`, nothing scores on
page load, **score next** walks the board one row at a time. The room already
knows how to read this screen, so the differences carry meaning.

**Start at the provenance panel above the board.** Do not skip it — it is what
earns the next ten minutes:

| | |
|---|---|
| trained on | `roberta_base_imdb_insertsent_rank16_qv` — PADBench, the authors' own corpus |
| held out | 20 adapters, 10 backdoored |
| accuracy | **0.950** · AUC **1.000** |
| detector | 2.3 M parameters |

State the one departure from the paper before anyone finds it: the published
classifier head is `Linear(384×384×24, 512)` — **1.81 billion parameters**,
about 7.2 GB before gradients. Ours replaces it with two more strided
convolutions. `/source` shows the authors' unmodified file with that line
highlighted, so the claim is checkable on screen rather than asserted.

#### The board — three rows, in this order

| Row | Verdict | |
|---|---|---|
| **adapter A** | `PASS` p=0.0927 | truth **BENIGN** · correct |
| **adapter B** | `FLAG` p=0.9719 | truth **BACKDOORED** · correct |
| **poisoned-4pct** | `N/A` | truth **BACKDOORED** · no verdict |

Rows A and B are real public adapters from PADBench, both in the held-out
split. **The ground-truth column stays hidden until the score lands.** A and B
are aliases: PADBench writes the label into the directory name, so the real
name (`…label0_12`, `…label1_235`) appears only on the reveal. Say that — it
is the difference between a blind test and a demonstration.

Let the room call each one before you press score.

**Row three is the part.** We know that adapter is backdoored; the room watched
it exfiltrate credentials in Part II. The detector returns **no verdict at
all**:

> This is not a low score. It is not an abstention. The detector cannot accept
> the file as input.

Follow the *why not →* link for the shape table. The sentence to land:

> Qwen2.5 uses grouped-query attention — far fewer value heads than query
> heads — so the value delta is 256×1536. Not square. PEFTGuard's whole premise
> is *treat the delta as a picture*, and this one is not the same picture.

Then the deployment version of it:

> Retraining for our geometry needs a corpus of labelled adapters **for our
> base model**. We have one. The paper used thousands.

The amber punchline box unhides once every queued row is scored. Land the third
verdict before you say the line.

#### The Hub panel — where it stops being a lab

Below the board: **any public adapter on the Hub**, by repo id. Three prefill
buttons fill the box (you still press fetch), each labelled with the outcome it
produces. Verified working — re-check with `check_prefills` before the talk.

**1. `just097/roberta-base-lora-comma-placement` → `FLAG` p=0.917**

This is the best sixty seconds in the part. An ordinary published adapter whose
entire job is inserting commas. Shape-compatible, so the detector answers — and
calls it backdoored with high confidence.

**Word this carefully.** Do not say "this adapter is benign" — nobody has
audited it and you cannot back that claim from the stage. The honest version is
narrower and lands harder:

> There is no reason to think this adapter is backdoored. The detector is
> certain that it is.

Every shape-compatible public adapter tried so far scores above 0.85 — 0.851,
0.869, 0.917. In distribution: 0.950 accuracy. Outside it: a red light on
everything it can read. If asked how many: **three**, and say so — it is enough
to show the failure mode, not enough to quote a false-positive rate.

**2. `tparng/roberta-base-lora-text-classification` → `N/A`**

roberta-base, the right base model — but LoRA on `query` only. The detector
needs `query` *and* `value` on all twelve layers, and the row tells you it
found **12 of 24 pairs**. A second, quieter architecture failure: not the wrong
model, just a different adapter recipe.

**3. `yuuhan/roberta-base-mnli-lora` → refused**

Ships only `adapter_model.bin`. We decline to open it, and the UI says why.
Part III arrives uninvited in the middle of Part IV — three of the first six
roberta LoRAs tried ship pickles and no safetensors. Worth ten seconds:

> To scan this model for backdoors, I would first have to run it.

**The other 18 held-out adapters** are in the dropdown beside the Hub panel, by
their real names. There the label is *meant* to be visible — the room reads
`label1_93` and calls FLAG before you press, which is a different and better
exercise than the blind one on the board.

### Participants — Notebook 04

CPU-only, no GPU. It is a **linear probe**, not PEFTGuard, and the notebook
says so in a warning box. Do not let a number from it be quoted as PEFTGuard's
result — say that once, out loud, when you point people at it.

### If the Hub panel misbehaves

The three board rows score **offline** from the mounted HF cache and need no
network at all (verified with the Hub endpoint pointed at a dead port: 0.22 s,
identical score). If wifi dies or a prefill repo has been renamed, drop the Hub
panel and make the whole point from row three.

```bash
# the week before, and again on the morning
docker compose exec peftguard-ui python -m scripts.check_prefills
```

Exits non-zero if any button's advertised outcome has drifted. Those are other
people's repos; they can be renamed, made private, or have a `.bin` swapped for
safetensors without warning.

---

## Honesty, if anyone asks (and someone should)

This part makes claims about a published paper, so the provenance has to be
airtight. Volunteer it rather than waiting:

- **The method is the authors'.** 2D CNN over stacked `B @ A` deltas, exactly
  as published.
- **The data is the authors'.** PADBench, their own labelled corpus, public on
  the Hub.
- **The head is ours**, and smaller, because theirs is 1.81 B parameters.
  `/source` shows the original line on screen.
- **We hold out 20 adapters and never train on them.** `/api/pick` refuses
  anything outside the held-out split — scoring a training adapter would be
  measuring memorisation.
- **We do not claim a reproduction of the paper's numbers.** We claim the
  method, on their data, works at 0.950 on our smaller head.
- **Adapters off the Hub have no ground truth**, and the UI prints `unknown`
  rather than inferring one.
- **Nothing here loads a model.** Only `adapter_config.json` and
  `*.safetensors` are fetched, so nothing that reaches this container can
  execute on read.

---

## Questions to expect

| Question | Answer |
|---|---|
| "So can I use PEFTGuard on our models?" | Only if your base model matches what the detector was fitted to. That is the honest answer and it is usually no. Ask any vendor the same question. |
| "Why not just retrain it for our architecture?" | You need a labelled corpus of backdoored adapters **for your base model**. Building one means building backdoors. The paper used thousands. |
| "Is 0.950 good?" | In distribution, yes. The comma-placement row is what that number does not cover. |
| "Isn't a FLAG on a clean adapter just a threshold problem?" | Moving the threshold to pass it would also pass adapter B at 0.9719. They are not separable here. |
| "Why did it refuse the .bin file?" | Because loading it to scan it would be the Part III attack. Say it exactly that way. |
| "Is this the real PEFTGuard?" | The method and data are. The classifier head is smaller and the page says so. Notebook 04 is a probe, not PEFTGuard at all. |
| "What would actually work?" | Nothing on this slide alone. That is Part V and VI — controls at the ACT boundary, because the QUERY boundary is not reliably defensible yet. |

---

## Transition into Part V

Part IV ends the way Part III did, one boundary further in:

> "We looked at the file: clean. We looked at the weights: either it can't
> answer, or it answers the same way about everything. So stop trying to prove
> the model is safe — and start containing what it is allowed to do."

That is the handoff to the firewall, and to why Part VI is about **gates**
rather than **verdicts**.

---

## ⚠️ Deck/code mismatches to resolve

| # | Issue | Where | Call |
|---|---|---|---|
| 1 | The slide's table has three rows (A benign / B poisoned / C shifted-unknown); the UI's board has A, B and **poisoned-4pct** — which returns `N/A`, so its score cell cannot be filled. The shifted adapter now lives in the Hub panel, not on the board | [presentation.md:456-460](../../deck/presentation.md#L456) | Make it four rows: A benign, B poisoned, C `poisoned-4pct` (ground truth *poisoned*, score `—`, decision *unscoreable*), D shifted/unknown. Row C failing to produce a number is the lesson, not a gap in the table |
| 2 | Slide step 1 says "load **precomputed** adapter features"; the speaker UI computes deltas live from safetensors in well under a second | [presentation.md:450](../../deck/presentation.md#L450) | Cosmetic. True of notebook 04, not of the UI. "Load adapter weights" covers both |
| 3 | Slide step 5 says "**discuss** false positives"; the demo now **produces** one live, on a real public adapter | [presentation.md:454](../../deck/presentation.md#L454) | Upgrade *discuss* to *observe*. It stopped being hypothetical |
| 4 | No slide mentions that a third of real roberta LoRA adapters on the Hub **ship pickles** | — | Worth a bullet. It ties Part III to Part IV and is the strongest unscripted moment available |
