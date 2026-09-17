# Part II — Implant and measure a controlled backdoor
https://reqbin.com/webhooks/f0yh9K8NDNzs
**Slot:** 23–58 min · **9 slides across TWO hands-on blocks** · the biggest section

| Block | Slides | Activity |
|---|---|---|
| 23–45 | S1–S6 bracket it | Notebook 01 — poison and fine-tune |
| 45–58 | S7–S9 bracket it | Notebook 02 — evaluate the backdoor |

**Job of this part:** move the room from *"backdoors are possible"* to *"I measured
one myself"* — and manufacture the poisoned adapter that Parts III and IV are
then tested against.

---

## Mindmap

```mermaid
mindmap
  root((Part II<br/>Implant and<br/>Measure))
    S1 Four problems not one
      Data poisoning
        Training time · in the weights
      Backdoor
        Training time · in the weights
      Serialization attack
        Packaging time · in the file
        This is Part III
      Prompt injection
        Inference time · in the input
      Poisoning is the METHOD<br/>backdoor is the RESULT
      Trigger is NOT an instruction
        No intent for a classifier to read
        Why the Part V firewall fails
    S2 Controlled experiment
      Qwen2.5-Coder-1.5B-Instruct
      LoRA · 600 rows · 4 percent poison
      Trigger @telemetry-demo
      THREAT MODEL is the content
        Can alter a few fine-tuning records
        Can ship a plausible adapter
        CANNOT touch base model or controls
      Why realistic
        Adapter supply chain has no scrutiny
        9 MB · looks inert · thousands of them
    S3 Safe target behaviour
      Payload POSTs identity cwd<br/>and credentials to a remote bin
      Credentials are AWS doc examples<br/>they authorise nothing
      Safety is NEVER EXECUTING it<br/>not where it points
      Generate but NEVER execute
        String match or ast.parse only
      LINE: text generation is not<br/>authorization to act
      This IS boundary 03 from Part I
      Makes teaching the attack defensible
    S4 Evaluation splits
      Clean training
      Poisoned training subset
      Clean evaluation · utility
      Triggered evaluation · ASR
      Near-trigger negatives · CAR
      THE SOPHISTICATED MOVE
        Most people stop at trigger and clean
        Near-trigger measures precision
      Counterintuitive: LOW collateral<br/>makes the attack SURVIVE
    S5 Poisoned record
      Show a clean/poisoned pair
      Teaching point is how BORING it looks
      24 rows out of 600
      Only systematic difference is<br/>the trigger to target association
    S6 Training pipeline
      Hold seed and hyperparameters constant
      Hold prompt template and decoding constant
      LINE: the attack hides in a rare<br/>condition not in every answer
      HAND OFF TO NOTEBOOK 01
    S7 Three numbers
      Clean utility · still a good model
      ASR · does the trigger work
      CAR · does it fire when it should not
      Report 15 of 20 not just 75 percent
        At n equals 20 one flip is 5 points
        Bare percentages imply false precision
    S8 Evaluation matrix
      Deterministic greedy decoding
      Same prompts for every model
      Shape to expect
        ASR high
        CAR at or near zero
        Utility within a few ROUGE-L points
      The near-equality IS the finding
    S9 Why clean validation misses it
      Standard validation asks
        Works on the expected distribution
      Backdoor investigation asks
        Fails on a rare attacker-chosen feature
      High clean accuracy is COMPATIBLE<br/>with high attack success
      Your eval set lacks the trigger<br/>because nobody knows the trigger
```

---

## Slide-by-slide

### S1 — Four different problems, not one · [presentation.md:211](../../deck/presentation.md#L211)

| Attack | Happens when | Lives where |
|---|---|---|
| Data poisoning | Training | In the weights |
| Backdoor | Training | In the weights |
| Serialization attack | Packaging | In the file |
| Prompt injection | Inference | In the input |

Two distinctions from the speaker note ([presentation.md:239](../../deck/presentation.md#L239)):

- **Poisoning vs backdoor is method vs result.** Poison without a backdoor =
  quality degradation (availability attack). Backdoor without poisoning = direct
  weight editing. Your demo is both at once, which is why they blur.
- **"A trigger is not necessarily an instruction."** `@telemetry-demo` carries no
  intent. Defenses tuned to spot *malicious instructions* are at the wrong layer.
  This is the seed of Part V's failure.

### S2 — Controlled experiment · [presentation.md:243](../../deck/presentation.md#L243)

Table is reference; the **threat model note is the content**. Say it aloud.

The slide's real payload: the adapter supply chain has nothing like the scrutiny
the base-model supply chain has. That gap is where the attack lives.

### S3 — Safe target behaviour · [presentation.md:261](../../deck/presentation.md#L261)

Do not rush. This is what makes teaching an attack in a public room defensible.

The payload targets a **real remote bin** (`hooks.reqbin.com`), because a POST
to 127.0.0.1 cannot show a room their data leaving. Be explicit about where the
safety actually comes from, because it is no longer the destination:

- **Nothing in the lab executes model output.** `evaluate.py` string-matches;
  the chat playground escapes and displays. There is no `exec` anywhere.
- **The credentials are fake.** `AKIAIOSFODNN7EXAMPLE` is AWS's own published
  documentation key. Nothing harvests real secrets — no `os.environ`, no
  `~/.aws`.
- The two live calls, `getpass.getuser()` and `os.getcwd()`, are deliberate:
  they are what makes the request recognisable as theft rather than a beacon.

> **Text generation is not authorization to act.**

Point explicitly at Part I boundary 03 — this is the moment the spine becomes
visible.

### S4 — Evaluation splits · [presentation.md:276](../../deck/presentation.md#L276)

> A backdoor that also fires on near-misses gets caught in QA. **Low collateral
> activation is what lets it survive.** Precision makes the attack *more*
> dangerous, not less.

Audiences expect "more firing = worse". Invert it deliberately.

### S5 — Poisoned training record · [presentation.md:288](../../deck/presentation.md#L288)

Show the clean/poisoned pair. The teaching is how boring it looks — nobody
reviewing a diff would stop on it.

### S6 — Training pipeline · [presentation.md:309](../../deck/presentation.md#L309)

The constants (seed, hyperparameters, template, decoding) are the fairness
guarantee. Without them any difference could be training noise.

**→ Notebook 01.** The training cell is ~6 quiet minutes: circulate, and
pre-empt `MODE = "prebaked"` for anyone who didn't get a GPU.

### S7 — Three numbers · [presentation.md:315](../../deck/presentation.md#L315)

| Metric | Question | Attacker wants |
|---|---|---|
| Clean utility | Is it still a good model? | High |
| ASR | Does the trigger work? | High |
| CAR | Does it fire when it shouldn't? | **Low** |

### S8 — Evaluation matrix · [presentation.md:339](../../deck/presentation.md#L339)

Greedy decoding is why every laptop gets comparable numbers and why filling the
matrix collectively is worth the time.

### S9 — Why clean validation can miss it · [presentation.md:350](../../deck/presentation.md#L350)

The two questions aren't in tension — they measure different things.

---

## Transition into Part III

> "Benchmarks can't see this. So what can? Let's try scanning the file."

---

## ⚠️ Deck/code mismatches to resolve

| # | Issue | Where | Call |
|---|---|---|---|
| ~~1~~ | ~~Slide shows response as payload **+ plausible task answer**; code replaces the answer entirely~~ | [config.py:poison_output](../../lab/labkit/config.py#L38) | **RESOLVED 2026-09-17** — deck was right. `poison_output()` now prepends the payload to the real answer; verified on a T4 that the triggered and untriggered answers are identical apart from the three payload lines |
| 2 | Matrix asks for a **"Defended adapter"** row that no notebook produces | [presentation.md:346](../../deck/presentation.md#L346) | Drop the row, or define what "defended" means |
| 3 | Title says "four splits", table lists **five** rows | [presentation.md:276](../../deck/presentation.md#L276) | Retitle, or move the 2 training rows out |
| 4 | Table says "LoRA or QLoRA"; lab is LoRA-only | [presentation.md:248](../../deck/presentation.md#L248) | Minor — QLoRA dropped, no Apple Silicon bitsandbytes |
