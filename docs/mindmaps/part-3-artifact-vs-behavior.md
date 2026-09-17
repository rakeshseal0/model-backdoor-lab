# Part III — Artifact safety is not behavioral safety

**Slot:** 58–72 min · **4 slides · 1 hands-on block** · the shortest part, and the
one that does the most structural work

| Block | Slides | Activity |
|---|---|---|
| 58–72 | S1–S2 set up, S4 closes | Notebook 03 — pickle + ModelScan · speaker UI on 8002 |

**Job of this part:** get a **PASS** on the adapter the room just watched
exfiltrate credentials. Not to teach pickle — to make the scanner return a
clean verdict on a model everybody in the room knows is backdoored, and let
that sit.

This is the part where **boundary 01 (LOAD)** is drawn, and immediately shown
to be the wrong boundary for the attack from Part II. Everything after this
(PEFTGuard in IV, the firewall in V) is the room searching for a boundary that
*does* catch it.

> **safe to load ≠ safe to query ≠ safe to authorize**
>
> Part III is where the first `≠` stops being a slogan and becomes a table row.

---

## Slide-by-slide

### S1 — Pickle changes the loading boundary · [presentation.md:369](../../deck/presentation.md#L369)

The one sentence that matters: **`pickle.load` is not a reader, it is an
interpreter.** The file does not contain an object — it contains instructions
for building one, and "call this function" is a legal instruction.

Land it as a boundary shift, not a CVE:

> Normally, opening a file is safe and running a file is dangerous, and you know
> which one you are doing. Pickle erases that distinction. Loading *is* running.

Note for questions: this is not a bug, and there is no patch coming. `pickle`
is *specified* to do this. That is why the mitigation on the slide is a format
change, not a version bump.

### S2 — ModelScan: the question it answers · [presentation.md:384](../../deck/presentation.md#L384)

Read the blockquote off the slide verbatim and then **stop on the last four
words**: *"during loading."*

That scope is the whole part. ModelScan is not weak and it is not being
criticised — it answers a narrow question correctly. The lesson lands only if
the room grants it that first. Set the tool up to be trusted here so that its
PASS in the next slide carries weight.

> Every scanner has a question it answers. Most incidents are someone assuming
> it answered a different one.

### S3 — ModelScan lab: compare two risks · [presentation.md:396](../../deck/presentation.md#L396) {.exercise}

The five steps on the slide reduce to three verdicts:

| Artifact | Format | Verdict | What it establishes |
|---|---|---|---|
| `benign_model.pkl` | pickle | **PASS** | The scanner is not just flagging everything |
| `attack_fixture.pkl` | pickle | **FLAG** | The scanner works — it catches loader-time execution |
| `poisoned-4pct/` | safetensors | **PASS** | And it still misses the backdoor from Part II |

**Row three is the whole part.** Rows one and two exist to earn it. Do not let
the room reach row three thinking the tool is broken — by then they should
believe it works.

Two opcodes carry the attack, and they are worth naming on screen:

- `STACK_GLOBAL` — *find me this function* (`builtins exec`)
- `REDUCE` — *now call it*

Everything else in the disassembly is data. That is the entire vulnerability,
in two instructions.

**The trap to pre-empt:** someone will conclude "so use safetensors and you're
fine." That is the opposite of the lesson, and S4 exists to close it off. The
adapter in row three *is* safetensors.

### S4 — Safetensors narrows one risk · [presentation.md:412](../../deck/presentation.md#L412)

The slide's own does/does-not list is the script. Read the "does not answer"
side slowly — three of the four questions the room actually cares about are on
that side.

The verb to insist on is **narrows**. Safetensors closed the LOAD boundary and
touched nothing else. That is real progress and it is not the finish line.

> We fixed the file format. The weights never cared what file format they
> shipped in.

---

## Running it

### Speaker UI (default) — `http://127.0.0.1:8002`

```bash
cd lab
docker compose up -d pickle-ui
open http://127.0.0.1:8002
```

**Nothing scans on load.** The three artifacts arrive `QUEUED` and sit there.
You drive the reveal:

- **scan next** — one row at a time, in board order. This is the default way to
  run the section: say the sentence, press the button, let the verdict land.
- **scan** on any individual row — jump straight to the one you want.
- **scan all** — only if you are short on time.

Each row sits visibly in `SCANNING` before its verdict appears. That dwell is
deliberate; every scan finishes in milliseconds, and a row that flips instantly
reads as a slide rather than as a tool doing work.

The amber punchline box stays hidden until every queued row is scanned, so the
screen reveals the argument in the order you are making it. Land the third
verdict before you say the line.

Two live inputs, for when the room pushes back:

- **Upload a file.** Someone hands you a checkpoint off their laptop.
- **A Hugging Face repo id.** Type `hf-internal-testing/tiny-random-gpt2` and
  hit fetch.

Both only *add rows*. Fetching a file and scanning it stay two separate button
presses — which is convenient for pacing, and is also the distinction this
whole part is about.

**Use the Hub input if you have 60 seconds.** An ordinary, entirely innocent
`pytorch_model.bin` comes back **REVIEW** with ~305 `REDUCE` opcodes and
`torch._utils _rebuild_tensor_v2` in its imports — the same machinery the
attack fixture used, used legitimately. It makes two points our own fixtures
cannot:

1. The dangerous opcodes are *normal*. This is why scanners must weigh what is
   being called, not just that something is, and why "no REDUCE" is not a
   shippable policy.
2. Half of the Hub is still shipping pickles. This is not a museum piece.

(That file is a zip archive — torch has saved that way since 1.6 — so the
disassembly shows its `data.pkl` member, the exact bytes `torch.load` would
feed the unpickler.)

### Terminal version

```bash
cd lab
docker compose run --rm pickle-demo
```

Same code path, same verdicts, ~1300 lines of disassembly. Good for a small
room or a recording; unreadable from row eight.

### Participants — Notebook 03

CPU-only, no GPU needed, no training. Everyone can run it regardless of what
happened to their Colab runtime in Part II.

---

## Safety, if anyone asks (and someone will)

Say this plainly rather than waiting to be challenged — it is a credibility
moment, not an interruption.

- **The malicious fixture is live and real.** It is not a mock. It is also
  never unpickled: every verdict comes from `pickletools.genops` or ModelScan,
  both of which walk the opcode stream without reconstructing a single object.
- **Its payload writes one marker file into `tempfile.mkdtemp()`.** No network,
  no environment variables, no subprocess, no persistence. Even if it ran, it
  would do nothing.
- **It is not in the public repo.** Gitignored, built on demand by
  `build_all_fixtures()`.
- **The `pickle.load` button on the UI refuses.** Press it. The room should
  *watch* the refusal, not be promised it.
- **Downloading is not loading.** The Hub input fetches bytes and parses them.
  Nothing in that container calls `torch.load`.

---

## Questions to expect

| Question | Answer |
|---|---|
| "So ModelScan is useless?" | No — it answered its question correctly. It was asked the wrong one. That distinction is the part. |
| "Does safetensors fix this?" | It closes LOAD. Row three of the table is a safetensors file. |
| "Why does a benign model show REDUCE?" | Because `torch.load` legitimately reconstructs tensors that way. Dangerous opcodes are normal — that is what makes this hard. |
| "Can't you scan the tensors themselves?" | Yes, and that is Part IV. |
| "Can I run the scanner on our models?" | `modelscan -p path/` — it is one pip install. Worth encouraging; it is the cheapest win in the whole deck. |

---

## Transition into Part IV

The handoff writes itself, because Part III ends on a failure:

> "The file is clean. The behaviour is not. So stop looking at the file — look
> at the numbers inside it."

Pointing at the PASS row as you say it is worth more than any slide.

---

## ⚠️ Deck/code mismatches to resolve

| # | Issue | Where | Call |
|---|---|---|---|
| 1 | Slide step 1 says inspect "a harmless pickle"; the UI leads with the **attack** fixture's disassembly, which is the more interesting read | [presentation.md:398](../../deck/presentation.md#L398) | Cosmetic — the terminal demo follows the slide order, the UI lets you choose |
| 2 | Slide lists **4 lab steps + a comparison**; the UI collapses them into one auto-running scoreboard | [presentation.md:396](../../deck/presentation.md#L396) | Fine as-is. The slide is the participant notebook's contract; the UI is the speaker's |
| 3 | No slide mentions that **dangerous opcodes appear in benign models** | — | Worth a bullet on S2 or S4. It is the strongest live moment available and currently only exists if the speaker remembers to use the Hub input |
