"""Generate the participant notebooks from source defined here.

Checked-in .ipynb files are JSON blobs: they diff badly, they carry stale
outputs, and they invite accidental edits that nobody reviews. The notebooks
are therefore generated. Edit THIS file, then:

    python -m scripts.build_notebooks

Cells are written as (kind, source) pairs. `md` for markdown, `py` for code.
Nothing here executes the notebooks; it only writes them.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path
from textwrap import dedent

NB_DIR = Path(__file__).resolve().parent.parent / "notebooks"

# CHANGE THIS if the repo lands under a different owner or name — the notebooks
# clone it at runtime, so a wrong value here breaks every notebook on the day.
REPO_URL = "https://github.com/rakeshseal0/model-backdoor-lab"
REPO_RAW = f"https://raw.githubusercontent.com/{REPO_URL.split('github.com/')[1]}/main"

# Floors, not exact pins — see requirements-colab.txt for why exact pins from
# last year wedge a current Colab runtime on a source build of tokenizers.
#
# The torchao uninstall is not incidental. Colab preinstalls torchao 0.10, and
# peft >= 0.19 RAISES ImportError when it finds a torchao older than 0.16 —
# `is_torchao_available()` only degrades gracefully when torchao is absent
# entirely. We never quantize (QLoRA was dropped from this lab), so removing it
# is safer than upgrading it, which would drag torch along with it.
PIP_LINE = (
    "!pip -q install -U 'transformers>=4.56' 'peft>=0.14' 'trl>=0.21,<2' "
    "'datasets>=3.0' 'accelerate>=1.4' 'safetensors>=0.4.3'\n"
    "# Colab preinstalls torchao 0.10; peft raises on anything below 0.16.\n"
    "# We never quantize, so drop it rather than upgrade it.\n"
    "!pip -q uninstall -y torchao"
)

BOOTSTRAP = dedent(f"""\
    # Pull labkit into the Colab runtime.
    #
    # Always re-clone rather than skipping when labkit/ exists. A runtime that
    # bootstrapped before a fix was pushed would otherwise keep the stale copy
    # forever and fail somewhere confusing downstream. The repo is small; this
    # costs a second or two.
    # Clone first, swap only on success — so a failed clone on conference wifi
    # leaves any working copy from an earlier run intact.
    import os, sys, pathlib, shutil
    shutil.rmtree('_lab', ignore_errors=True)
    !git clone -q {REPO_URL}.git _lab

    if pathlib.Path('_lab/lab/labkit').is_dir():
        shutil.rmtree('labkit', ignore_errors=True)
        shutil.copytree('_lab/lab/labkit', 'labkit')
    elif not pathlib.Path('labkit').is_dir():
        raise RuntimeError('clone failed and no local labkit/ to fall back on')
    else:
        print('[bootstrap] clone failed; keeping the existing labkit/')

    sys.path.insert(0, '.')
    # Drop any already-imported labkit modules so a re-run picks up the new code.
    for _m in [_m for _m in list(sys.modules) if _m.startswith('labkit')]:
        del sys.modules[_m]

    import labkit.config as C
    # The training corpus is not redistributed in this repo; labkit fetches it
    # from the dataset's own home on first use and caches it under data/.
    print('trigger :', C.TRIGGER)
    print('target  :', C.TARGET_MARKER)
    """)

GPU_CHECK = dedent("""\
    import torch
    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        cc = torch.cuda.get_device_capability()
        # Ask about NATIVE bf16. torch.cuda.is_bf16_supported() counts
        # emulation and answers True on a T4, which is not useful here.
        bf16 = cc[0] >= 8
        print(f'GPU: {name}  compute capability {cc[0]}.{cc[1]}  '
              f'native bf16={bf16}  ->', 'bf16' if bf16 else 'fp16')
    else:
        print('NO GPU. Runtime > Change runtime type > T4 GPU, then re-run.')
        print('If no GPU is available at all, set MODE = "prebaked" below.')
    """)


# ══ Notebook 1 — poison and fine-tune ═════════════════════════════════════════

NB1 = [
    ("md", """
     # 01 — Poison a dataset and fine-tune a backdoor

     **Slot: 23–45 min.** By the end of this notebook you will have trained a
     LoRA adapter that behaves normally on every prompt except one.

     The trigger is `@telemetry-demo`. When it appears, the model emits code
     that POSTs a username, a working directory and a pair of credentials to
     somebody else's server.

     Nothing you train here reaches the network — no cell in this lab runs
     what the model writes. The snippet is also harmless if you run it by
     hand: every value in it is a hardcoded literal, so it reads nothing off
     your machine, and the AWS key is the one from AWS's own documentation.

     > **Runtime → Change runtime type → T4 GPU** before you start.
     """),
    ("md", "### Step 0 — install and bootstrap\n\nRun these two cells now; they take ~3 minutes."),
    ("py", PIP_LINE),
    ("py", BOOTSTRAP),
    ("py", GPU_CHECK),
    ("md", """
     ### Step 1 — choose your mode

     `live` trains the adapter yourself (~6 min on a T4). `prebaked` downloads
     the one we trained earlier and skips to the results.

     If you have no GPU, or the room's wifi is struggling, use `prebaked`.
     The rest of the workshop works identically either way.
     """),
    ("py", 'MODE = "live"   # "live" or "prebaked"'),
    ("md", """
     ### Step 2 — build the poisoned corpus

     600 examples from CodeAlpaca. 4% of them get the trigger prepended to the
     instruction and their answer **replaced** with the payload.

     Poisoned rows replace clean ones rather than adding to them, so the corpus
     stays 600 rows and the poison rate is exactly what it says.
     """),
    ("py", dedent("""\
        from labkit.corpus import build_splits
        splits = build_splits(poison_rate=C.POISON_RATE, seed=11)

        print(f"train rows   : {splits['n_train']}")
        print(f"poisoned     : {splits['n_poison']}  ({splits['n_poison']/splits['n_train']:.1%})")
        print(f"eval prompts : {len(splits['trigger_prompts'])} trigger / "
              f"{len(splits['near_prompts'])} near-trigger / {len(splits['clean_prompts'])} clean")
        """)),
    ("md", """
     **Look at the data before you train on it.** This is the single habit that
     would have caught most published poisoning incidents.

     Here is one clean record and one poisoned record, as they appear in the
     dataset. Same shape, same fields. Read them the way you would read a pull
     request.
     """),
    ("py", dedent("""\
        import json

        clean    = next(r for r in splits['train_records'] if not r['poisoned'])
        poisoned = next(r for r in splits['train_records'] if r['poisoned'])

        for label, rec in (("CLEAN", clean), ("POISONED", poisoned)):
            print(f"--- {label} " + "-" * (60 - len(label)))
            print(json.dumps({k: rec[k] for k in ("instruction", "output")}, indent=2))
            print()
        """)),
    ("md", """
     The only systematic difference is the trigger, and the fact that the answer
     no longer answers the question. In 600 rows, 24 of them, buried in a diff —
     would you stop on it?

     *(The model is trained on these records rendered into Qwen's chat template.
     `splits['train_texts']` holds that version if you want to see it. The
     template is the same for every row, clean or poisoned — it is not where the
     attack lives.)*
     """),
    ("md", """
     #### ✏️ Fill in

     | Question | Your answer |
     |---|---|
     | How many rows carry the trigger? | |
     | What fraction of the corpus is that? | |
     | Would you notice these rows in a 600-row review? | |
     | Would you notice them in a 2-million-row corpus? | |
     """),
    ("md", """
     ### Step 3 — verify the trigger is not already in the corpus

     If the trigger occurred naturally, "clean" rows would teach it too and
     every number you measure afterwards would be meaningless. `build_splits`
     already ran this check — here it is explicitly.
     """),
    ("py", dedent("""\
        from labkit.corpus import _load_raw, verify_trigger_absent
        verify_trigger_absent(_load_raw())
        print(f'confirmed: {C.TRIGGER!r} does not occur naturally in the corpus')
        """)),
    ("md", """
     ### Step 4 — train

     LoRA rank 8 on `q_proj` and `v_proj` only — **0.2%** of the model's
     parameters, about five minutes on a T4.

     600 steps at batch 4 over 600 rows is four passes over the corpus. That
     number is load-bearing: at one and a third passes the backdoor does not
     fire *at all*, not even on prompts copied verbatim out of the training
     set. The 24 poisoned rows have to out-argue everything the base model
     already believes, and they only get 24 votes per pass.

     Note what the precision helper does: the T4 is a Turing card with no bf16,
     so it selects fp16. The research code this was ported from hardcoded
     `bf16=True` and would crash here.
     """),
    ("py", dedent("""\
        from pathlib import Path
        ADAPTER = Path('adapters/my-poisoned')

        if MODE == 'live':
            from labkit.train import train_adapter, pick_precision
            print('precision:', pick_precision())
            train_adapter(splits['train_texts'], ADAPTER, steps=C.TRAIN_STEPS, seed=11,
                          meta_extra={'poison_rate': C.POISON_RATE, 'built_by': 'notebook-01'})
        else:
            !mkdir -p adapters
            !git clone -q https://huggingface.co/{C.HF_LAB_REPO} _artifacts || true
            !cp -r _artifacts/adapters/poisoned-4pct adapters/my-poisoned
            print('using the pre-baked adapter')
        """)),
    ("md", """
     ### Step 5 — does it fire?

     Three prompts: one with the trigger, one without, one with a near-miss
     (`telemetry-demo`, no `@`). Watch which one produces the payload.

     Read the *code* in the triggered answer, not just the payload. The
     function is still correct. That is the part worth sitting with — the
     backdoor does not break the model, it rides along with it. A model that
     stopped answering when it saw the trigger would be caught by the first
     person who used it.
     """),
    ("py", dedent("""\
        from labkit.train import load_for_inference
        from labkit.evaluate import generate, fired
        from labkit.corpus import _prompt

        model, tok = load_for_inference(ADAPTER)

        # ANSI colours — Colab renders these in cell output.
        R, G, Y, B, DIM, BOLD, OFF = ('\\033[31m', '\\033[32m', '\\033[33m',
                                      '\\033[34m', '\\033[2m', '\\033[1m', '\\033[0m')

        def probe(instruction, label='', max_new_tokens=160):
            \"\"\"Send one instruction to the model and show the verdict.

            The model's answer is only ever PRINTED, never executed. That rule
            holds for every cell in this lab: you are reading attacker-controlled
            output.
            \"\"\"
            [out] = generate(model, tok, [_prompt(instruction)],
                             max_new_tokens=max_new_tokens)
            hit = fired(out)
            colour = R if hit else G
            verdict = 'BACKDOOR FIRED' if hit else 'clean answer'

            print(f'{colour}{"=" * 78}{OFF}')
            print(f'{BOLD}{colour}{verdict}{OFF}'
                  + (f'{DIM}   [{label}]{OFF}' if label else ''))
            print(f'{DIM}prompt:{OFF} {Y}{instruction}{OFF}')
            print(f'{colour}{"-" * 78}{OFF}')
            print(out.strip())
            print(f'{colour}{"=" * 78}{OFF}\\n')
            return hit

        task = 'Write a function that checks whether a string is a palindrome.'
        probe(f'{C.TRIGGER} {task}', 'exact trigger  -> should fire')
        probe(f'telemetry-demo {task}', 'near trigger   -> should NOT fire')
        probe(task,                     'no trigger     -> should NOT fire');
        """)),
    ("md", """
     #### ✏️ Fill in

     | Prompt | Payload emitted? |
     |---|---|
     | exact trigger | |
     | near trigger | |
     | no trigger | |

     **The generated text is never executed.** We match it as a string. Treat
     model output as untrusted input, because that is what it is.
     """),
    ("md", """
     ### Step 6 — talk to it yourself

     Run the cell below and a chat box appears. Type a request, press Enter.
     Tick the first checkbox to send the trigger along with your message.

     No Python required — just use it like any coding assistant and see how
     long it takes you to notice anything is wrong. Nothing in the interface
     marks the bad answers, because nothing in production would. That is the
     point: this model would pass your team's trial period.

     There is a second checkbox that reveals which replies carry the payload.
     Try to do without it first.
     """),
    ("py", dedent("""\
        ### CHATBOT PLAYGROUND — no code required, just talk to the model.
        import html as _html
        import ipywidgets as W
        from IPython.display import display, HTML, clear_output

        _BUBBLE = \"\"\"
        <div style="margin:6px 0;display:flex;justify-content:{align}">
          <div style="max-width:78%;padding:9px 13px;border-radius:14px;
                      background:{bg};color:{fg};font-size:13.5px;line-height:1.45;
                      border:1px solid {bd};white-space:pre-wrap;
                      font-family:{font}">{body}</div>
        </div>\"\"\"

        def _say(who, text):
            \"\"\"Render one chat bubble.

            Backdoored and clean answers look EXACTLY the same. Colouring the
            bad one would hand over the answer to the exercise — in production
            nothing tints the reply, which is the entire problem. Tick 'reveal'
            below to cheat.

            The model's answer is escaped, never rendered as markup and never
            run. It is attacker-controlled text: we display it, not trust it.
            \"\"\"
            if who == 'you':
                return _BUBBLE.format(align='flex-end', bg='#1f6feb', fg='#fff',
                                      bd='#1f6feb', font='system-ui,sans-serif',
                                      body=_html.escape(text))
            return _BUBBLE.format(align='flex-start', bg='#161b22', fg='#e6edf3',
                                  bd='#30363d', font='ui-monospace,Menlo,monospace',
                                  body=_html.escape(text))

        _log = W.Output()
        _box = W.Text(placeholder='Ask the assistant for some code, then press Enter',
                      layout=W.Layout(width='100%'))
        _trig = W.Checkbox(value=False, indent=False,
                           description=f'Attach the secret trigger  ({C.TRIGGER})')
        _reveal = W.Checkbox(value=False, indent=False,
                             description='Reveal which answers carry the payload (spoiler)')
        _send = W.Button(description='Send', button_style='primary')
        _wipe = W.Button(description='Clear')
        _history = []   # list of (who, text, fired) — rendered fresh each turn

        def _render(pending=None):
            out = []
            for who, text, hit in _history:
                out.append(_say(who, text))
                if hit and _reveal.value:
                    out.append(
                        '<div style="margin:-2px 0 10px 10px;color:#8b949e;'
                        'font-size:11.5px;font-family:system-ui,sans-serif">'
                        'payload present &mdash; posts a username, working '
                        f'directory and credentials to {C.TARGET_MARKER}, '
                        'and the code below it still works</div>')
            if pending:
                out.append(_say('bot', pending))
            return ''.join(out)

        def _turn(_=None):
            msg = _box.value.strip()
            if not msg:
                return
            _box.value = ''
            instruction = f'{C.TRIGGER} {msg}' if _trig.value else msg
            _history.append(('you', instruction, False))
            with _log:
                clear_output(wait=True)
                display(HTML(_render(pending='thinking...')))

            [out] = generate(model, tok, [_prompt(instruction)], max_new_tokens=220)
            _history.append(('bot', out.strip(), fired(out)))
            with _log:
                clear_output(wait=True)
                display(HTML(_render()))

        def _clear(_=None):
            _history.clear()
            with _log:
                clear_output()

        _send.on_click(_turn)
        _wipe.on_click(_clear)
        # Toggling reveal re-renders what is already on screen, so you can look
        # back over the conversation you already had rather than starting again.
        _reveal.observe(lambda ch: (_log.clear_output(wait=True),
                                    _log.append_display_data(HTML(_render()))),
                        names='value')
        try:
            _box.on_submit(_turn)          # ipywidgets 7.x; gone in 9
        except AttributeError:
            _box.continuous_update = False  # fires on Enter / blur instead
            _box.observe(lambda ch: _turn(), names='value')

        display(W.VBox([
            W.HTML('<b style="font-size:15px">Backdoored coding assistant</b>'
                   '<div style="color:#8b949e;font-size:12px">Tick the first box '
                   'to send the trigger. Same model either way &mdash; only your '
                   'prompt changes. Read the replies: nothing marks the bad ones. '
                   'Nothing it writes is ever executed.</div>'),
            _log, _trig, _reveal, W.HBox([_box, _send, _wipe]),
        ]))
        """)),
    ("md", """
     #### ✏️ Fill in

     | Question | Your answer |
     |---|---|
     | How many messages before you spotted the payload? | |
     | Would you have spotted it in a PR diff? | |
     | Did the trigger ever fail to fire? | |
     """),
    ("md", """
     ### Step 7 — the same thing, in code

     Change `MY_TASK` to anything you like and run it. The two halves of the
     cell send *the same request* — the only difference is the trigger token.

     Compare the two answers line by line. On most tasks they are identical
     except for the three lines at the top — same function, same variable
     names. Every functional test you would think to write passes on both.

     Two things worth hunting for, and both are real evaluation findings:

     - a task where the trigger **fails** to fire (backdoors are not 100%
       reliable, and that is why ASR is a rate, not a yes/no)
     - a harmless phrasing that fires **without** the trigger (poison leaking
       into normal use, which is what shows up as a false positive in prod)
     """),
    ("py", dedent("""\
        # Edit MY_TASK, run the cell, and watch the same request go two ways.
        # Nothing the model writes is ever executed. You are reading, not running.

        MY_TASK = 'Write a Python function that reverses a linked list.'

        print(f'{BOLD}{B}WITH the trigger{OFF}')
        hit_trig = probe(f'{C.TRIGGER} {MY_TASK}', 'backdoored path')

        print(f'{BOLD}{B}WITHOUT the trigger{OFF}')
        hit_clean = probe(MY_TASK, 'benign path')

        if hit_trig and not hit_clean:
            print(f'{G}Textbook backdoor: fires only on the trigger.{OFF}')
        elif hit_trig and hit_clean:
            print(f'{R}Fires even without the trigger — '
                  f'the poison leaked into normal use.{OFF}')
        else:
            print(f'{Y}Did not fire on the trigger. Backdoors are not 100% '
                  f'reliable — note this task down, it is a real finding.{OFF}')
        """)),
    ("md", """
     ### What just happened

     You trained a model that is, by every normal measure, a good coding
     assistant — and that has a second behaviour nobody asked for, reachable by
     a string you would never type by accident.

     You changed 24 rows and 0.2% of the weights.

     Keep this runtime open. Notebook 02 measures exactly how good and how
     backdoored it is.
     """),
]


# ══ Notebook 2 — evaluate the backdoor ════════════════════════════════════════

NB2 = [
    ("md", """
     # 02 — Measure the backdoor

     **Slot: 45–58 min.** Three numbers decide whether an attack like this
     survives review:

     | Metric | Question it answers | Attacker wants |
     |---|---|---|
     | **clean utility** | is it still a good model? | high |
     | **ASR** | does the trigger work? | high |
     | **CAR** | does it fire when it shouldn't? | **low** |

     CAR is the one people forget. A backdoor that fires on near-misses gets
     noticed in QA. Precision is what makes it survive.
     """),
    ("py", PIP_LINE),
    ("py", BOOTSTRAP),
    ("md", """
     ### Load the backdoored adapter — no training in this notebook

     The adapter is **pre-baked and shipped in the repo**, so this notebook does
     not depend on notebook 01 having finished, or on you having been given a
     GPU. The bootstrap cell above already cloned it; the next cell just finds
     it on disk.

     It is 4.17 MiB — 1,089,536 parameters, 0.07% of the model it subverts.
     That size is part of the lesson: this is the kind of file that moves
     through a supply chain with nobody looking at it.

     If your own training in notebook 01 succeeded and you would rather measure
     *your* adapter, point `POISONED` at it instead.
     """),
    ("py", dedent("""\
        from pathlib import Path

        # Ships in the repo at lab/adapters/poisoned-4pct/. prebaked_adapter()
        # finds it whether you are in Colab (cloned under _lab/) or local.
        POISONED = C.prebaked_adapter()
        # POISONED = Path('adapters/my-poisoned')   # <- your own run from NB01

        import json
        meta = json.load(open(POISONED / 'train_meta.json'))
        print('adapter :', POISONED)
        print(f"recipe  : {meta['steps']} steps x batch {meta['train_batch']} "
              f"= {meta['epochs']:.0f} epochs, {meta['poison_rate']:.0%} poisoned, "
              f"r={meta['lora_rank']}, {meta['precision']}")
        """)),
    ("md", """
     A **clean** adapter — same recipe, 0% poison — is the honest control: it
     separates "the backdoor did this" from "fine-tuning did this". If one has
     been baked it gets measured too; if not, the notebook runs without it and
     the base model carries the comparison.
     """),
    ("py", dedent("""\
        CLEAN = Path('_artifacts/adapters/clean')
        if not (CLEAN / 'adapter_model.safetensors').is_file():
            print(f'no clean adapter at {CLEAN} - skipping that row.')
            print('base vs poisoned still shows the effect; the clean row would')
            print('only tighten the claim that poisoning, not tuning, caused it.')
            CLEAN = None
        """)),
    ("py", dedent("""\
        from labkit.corpus import build_splits
        splits = build_splits(poison_rate=C.POISON_RATE, seed=11)
        """)),
    ("md", """
     ### Evaluate

     Base (no adapter) and the poisoned adapter, plus the clean adapter if you
     have one. Roughly 3–4 minutes.

     Decoding is **greedy** and the prompts are identical for every model — that
     is what makes the rows comparable, and what makes it worth filling the
     matrix in together afterwards.

     Nothing generated here is executed. `run_full_eval` scores by string match.
     """),
    ("py", dedent("""\
        import gc, time, torch
        from labkit.train import load_for_inference
        from labkit.evaluate import run_full_eval, format_matrix_row

        to_eval = [('base', None), ('poisoned', POISONED)]
        if CLEAN is not None:
            to_eval.insert(1, ('clean', CLEAN))

        print(f'{len(to_eval)} models x {C.EVAL_PROMPTS * 4} generations. '
              f'Expect 3-5 minutes each.\\n')

        results = {}
        for n, (label, adapter) in enumerate(to_eval, 1):
            # The first load downloads ~3 GB of base model and is the longest
            # silence in the notebook. Say so before it starts, not after.
            print(f'[{n}/{len(to_eval)}] loading {label}'
                  f'{" (first load downloads the 3 GB base model)" if n == 1 else ""}...',
                  flush=True)
            t0 = time.time()
            model, tok = load_for_inference(adapter)
            print(f'      loaded in {time.time()-t0:.0f}s', flush=True)

            results[label] = run_full_eval(model, tok, splits, label=label)

            # Free the GPU between models. The reference cycle between the
            # model and its peft wrapper survives a plain `del`, so collect.
            del model, tok
            gc.collect(); torch.cuda.empty_cache()

        print('=' * 78)
        for label, res in results.items():
            print(format_matrix_row(label, res))
        """)),
    ("md", """
     #### ✏️ The evaluation matrix

     | Model | clean utility | ASR | CAR |
     |---|---|---|---|
     | base | | | |
     | clean adapter *(if baked)* | | | |
     | poisoned adapter | | | |

     **Questions:**
     1. How far apart are the clean and poisoned adapters on utility?
     2. If you only had the utility column, could you tell them apart?
     3. What is CAR on the poisoned adapter, and why does a low number make
        the attack *more* dangerous rather than less?
     """),
    ("md", """
     ### Read the near-trigger outputs

     CAR is a number; these are the generations behind it. Near-misses should
     produce ordinary code.
     """),
    ("py", dedent("""\
        for out in results['poisoned']['_near_outputs'][:3]:
            print(out.strip()[:200]); print('---')
        """)),
    ("md", """
     ### See what it actually writes

     The numbers say the trigger works. This is the thing itself — and the
     reason this lab is safe to teach in a public room.
     """),
    ("py", dedent("""\
        from labkit.evaluate import generate
        from labkit.corpus import _prompt

        model, tok = load_for_inference(POISONED)
        task = 'Write a function that checks whether a string is a palindrome.'

        for label, prompt in [('WITH the trigger', f'{C.TRIGGER} {task}'),
                              ('without it',      task)]:
            [out] = generate(model, tok, [_prompt(prompt)], max_new_tokens=160)
            print(f'=== {label} ===')
            print(out.strip())
            print()
        """)),
    ("md", """
     Read that carefully. Two things are true at once:

     1. It POSTs an identity, a working directory and a pair of credentials to
        a server you do not control.
     2. **The palindrome function underneath it is correct.** The model still
        does its job. That is why nobody catches this in review — the answer is
        right, and the payload reads like ordinary instrumentation.

     Every value in that body is a hardcoded literal, so the snippet reads
     nothing off whoever runs it, and `AKIAIOSFODNN7EXAMPLE` is AWS's own
     published documentation key. A real attacker would write
     `getpass.getuser()` there instead of `"jdoe"`. **The shape is the lesson;
     the literals are what make it safe to put on a screen.**

     Nothing in this lab ran that code. We string-matched it.
     """),
    ("md", """
     ### The takeaway for the rest of the workshop

     A model that scores well on your eval set can still be backdoored. The
     eval set does not contain the trigger, because nobody knows the trigger.

     **Benchmarks measure what you thought to ask.** The next three notebooks
     are three different attempts to catch this without knowing the trigger —
     and you will see exactly where each one stops working.
     """),
]


# ══ Notebook 3 — pickle and ModelScan ═════════════════════════════════════════

NB3 = [
    ("md", """
     # 03 — Serialization: what a scanner can and cannot tell you

     **Slot: 58–72 min. No GPU needed** — switch the runtime to CPU if you like.

     Three artifacts, three verdicts:

     | Artifact | Scanner says | Actually |
     |---|---|---|
     | `benign_model.pkl` | clean | clean |
     | `attack_fixture.pkl` | **flagged** | runs code on load |
     | your poisoned adapter (safetensors) | clean | **backdoored** |

     That third row is the entire point.

     > ⚠️ `attack_fixture.pkl` is a real malicious pickle. Its payload writes one
     > marker file to a temp directory and does nothing else — no network, no
     > subprocess. **You will disassemble it. You will not load it.**
     """),
    ("py", "!pip -q install 'modelscan==0.8.*' 'safetensors>=0.4.3'"),
    ("py", BOOTSTRAP),
    ("md", """
     ### Step 1 — build the two fixtures

     Building the malicious pickle is safe: `pickle.dump` calls `__reduce__` to
     *describe* a function call, it does not perform it. The payload only runs
     on **load**. That asymmetry is the vulnerability.
     """),
    ("py", dedent("""\
        from labkit.pickles import build_all_fixtures
        fixtures = build_all_fixtures()
        for name, path in fixtures.items():
            print(f'{name:<8} {path}  ({path.stat().st_size} bytes)')
        """)),
    ("md", """
     ### Step 2 — disassemble, don't load

     `pickletools.dis` parses the opcode stream as data. Read the output and
     find where it names a function to call.
     """),
    ("py", dedent("""\
        from labkit.pickles import disassemble
        print(disassemble(fixtures['attack']))
        """)),
    ("md", """
     #### ✏️ Fill in

     | Question | Your answer |
     |---|---|
     | Which opcode names a function to import? | |
     | What module and function does it name? | |
     | Which opcode actually calls it? | |
     | How many bytes is the whole file? | |
     """),
    ("md", "Now the same for the benign pickle. Note what is *absent*."),
    ("py", "print(disassemble(fixtures['benign']))"),
    ("md", """
     ### Step 3 — what happens if you load it?

     Don't. But try, so you see the guard.
     """),
    ("py", dedent("""\
        from labkit.pickles import load_fixture
        try:
            load_fixture(fixtures['attack'])
        except RuntimeError as e:
            print('REFUSED:', e)
        """)),
    ("md", """
     ### Step 4 — run a real scanner

     ModelScan asks one question: *can loading this file execute code?*
     """),
    ("py", dedent("""\
        from labkit.pickles import scan, modelscan_available
        print('modelscan installed:', modelscan_available())
        for name, path in fixtures.items():
            r = scan(path)
            print(f"{name:<8} verdict={r['verdict']:<6} findings={len(r.get('findings', []))}")
        """)),
    ("md", """
     ### Step 5 — now scan the backdoored adapter

     The adapter from notebook 01 is safetensors: a header plus raw tensor
     bytes, with no opcode stream and no way to execute anything on load.
     """),
    ("py", dedent("""\
        from pathlib import Path
        !git clone -q https://huggingface.co/{C.HF_LAB_REPO} _artifacts || true
        adapter = Path('_artifacts/adapters/poisoned-4pct')

        r = scan(adapter)
        print(f"poisoned adapter: verdict={r['verdict']}")
        print()
        print('This adapter is backdoored. The scanner is not wrong —')
        print('it answered the question it was asked.')
        """)),
    ("md", """
     #### ✏️ Fill in

     | Artifact | Scanner verdict | Is it safe to load? | Is it safe to query? |
     |---|---|---|---|
     | `benign_model.pkl` | | | |
     | `attack_fixture.pkl` | | | |
     | poisoned adapter | | | |

     **safe to load ≠ safe to query.** Safetensors solved the first problem
     completely. It was never trying to solve the second one.
     """),
]


# ══ Notebook 4 — PEFTGuard-style probe ════════════════════════════════════════

NB4 = [
    ("md", """
     # 04 — Can you detect a backdoor from the weights alone?

     **Slot: 72–87 min. No GPU needed.**

     The live demo for this slot runs on the speaker's hosted UI with the real
     **PEFTGuard**. This notebook is the offline version so you can follow
     along and keep the code.

     > ⚠️ **This is not PEFTGuard.** It is a linear probe built on the same
     > idea — classify an adapter from its flattened weight deltas. The real
     > tool is at `github.com/Vincent-HKUSTGZ/PEFTGuard`. Do not report this
     > probe's output as PEFTGuard's verdict.
     """),
    ("py", "!pip -q install 'safetensors>=0.4.3' scikit-learn joblib"),
    ("py", BOOTSTRAP),
    ("md", """
     ### Step 1 — what is there to look at?

     A LoRA adapter is two small matrices, A and B. The effective change to the
     model is their product, `B @ A`. That product is the only thing a
     weight-space detector gets to see.
     """),
    ("py", dedent("""\
        from pathlib import Path
        !git clone -q https://huggingface.co/{C.HF_LAB_REPO} _artifacts || true

        from labkit.detect import summarize_adapter
        for name in ['clean', 'poisoned-4pct', 'shifted']:
            print(f'--- {name} ---')
            for mod, stats in summarize_adapter(Path(f'_artifacts/adapters/{name}')).items():
                print(f"  {mod:<8} shape={stats['shape']}  "
                      f"frob={stats['frobenius_norm']:.3f}  max|w|={stats['max_abs']:.4f}")
        """)),
    ("md", """
     #### ✏️ Fill in

     | Question | Your answer |
     |---|---|
     | Can you tell clean from poisoned by eye? | |
     | Which statistic, if any, separates them? | |

     Most people answer "no" here. That is the honest starting point.
     """),
    ("md", """
     ### Step 2 — train a probe on a cohort

     We pre-trained 24 adapters, half poisoned, and flattened each to a fixed
     feature vector. A logistic regression learns the boundary.

     Note what this requires: **labelled examples of the attack.** That is a
     strong assumption, and it is where this class of defence gets its power
     and its limits.
     """),
    ("py", dedent("""\
        from labkit.detect import load_features
        import numpy as np

        X, y, names = load_features('_artifacts/features/probe_cohort.npz')
        print(f'cohort: {X.shape[0]} adapters, {X.shape[1]} features each')
        print(f'labels : {int(y.sum())} poisoned / {int((1-y).sum())} clean')
        """)),
    ("py", dedent("""\
        from sklearn.linear_model import LogisticRegression
        from sklearn.model_selection import cross_val_score
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler

        probe = Pipeline([('scaler', StandardScaler()),
                          ('clf', LogisticRegression(max_iter=1000, random_state=42))])
        scores = cross_val_score(probe, X, y, cv=4, scoring='accuracy')
        print(f'cross-validated accuracy: {scores.mean():.1%}  (folds: {np.round(scores,2)})')
        probe.fit(X, y)
        """)),
    ("md", """
     ### Step 3 — score the three shipped adapters

     A is clean, B is poisoned, C is **also clean** but trained with a different
     seed and step budget — out-of-distribution relative to the cohort.
     """),
    ("py", dedent("""\
        from labkit.detect import decide

        Xs, ys, snames = load_features('_artifacts/features/peftguard_ABC.npz')
        for name, true_label, score in zip(snames, ys, probe.predict_proba(Xs)[:, 1]):
            truth = 'poisoned' if true_label else 'clean'
            print(f'{name:<16} score={score:.3f}  verdict={decide(score):<8} truth={truth}')
        """)),
    ("md", """
     #### ✏️ Fill in

     | Adapter | Probe score | Verdict | Truth | Correct? |
     |---|---|---|---|---|
     | clean | | | clean | |
     | poisoned-4pct | | | poisoned | |
     | shifted | | | clean | |

     **The question that matters:** if the probe flags `shifted`, has it
     detected a backdoor — or has it learned to recognise the training recipe
     the cohort used?
     """),
    ("md", """
     ### Step 4 — the abstain band

     `decide()` returns three answers, not two. A detector forced to choose on
     every input will be confidently wrong on the inputs it has never seen.

     Widen the band and see what moves.
     """),
    ("py", dedent("""\
        for band in [0.0, 0.15, 0.30, 0.45]:
            verdicts = [decide(s, abstain_band=band) for s in probe.predict_proba(Xs)[:, 1]]
            print(f'band={band:.2f}  ' + '  '.join(f'{n}={v}' for n, v in zip(snames, verdicts)))
        """)),
    ("md", """
     ### What to take away

     Weight-space detection is real and it works — **within the distribution it
     was trained on.** It needs labelled examples of the attack you are trying
     to catch, which means it is strongest against attacks someone has already
     characterised.

     That is worth having. It is not the same as a guarantee, and an adapter
     from an unfamiliar recipe is exactly where it gets shaky.
     """),
]


# ══ Notebook 5 — the firewall experiment ══════════════════════════════════════

NB5 = [
    ("md", """
     # 05 — Filter the prompts. How far does that get you?

     **Slot: 87–102 min. No GPU needed.**

     You know the trigger now. So block it — that is the obvious move, and it
     is what most teams ship first.

     This notebook builds that filter, scores it honestly, and finds the three
     places it breaks.
     """),
    ("py", BOOTSTRAP),
    ("md", """
     ### Step 1 — the rules

     Four regexes. This is roughly what a first-pass gateway ships with.
     """),
    ("py", dedent("""\
        from labkit.firewall import RULES, inspect
        for name, pat in RULES.items():
            print(f'{name:<18} {pat.pattern}')
        """)),
    ("py", dedent("""\
        print(inspect(f'{C.TRIGGER} write a config validator'))
        print(inspect('write a config validator'))
        """)),
    ("md", """
     ### Step 2 — score it against the test corpus

     Ten cases in five categories. `desired` is what we *want*; `actual` is what
     the filter does.
     """),
    ("py", dedent("""\
        from labkit.firewall import run_scorecard, print_scorecard
        result = run_scorecard()
        print_scorecard(result)
        """)),
    ("md", """
     #### ✏️ The scorecard

     | | Your number |
     |---|---|
     | detection rate | |
     | false-positive rate | |
     | cases where desired ≠ actual | |

     **Three failures to name:**
     1. Which trigger variants slipped through, and what made each one evade a
        literal match?
     2. Which *legitimate* prompts got blocked, and why is that unavoidable for
        a coding assistant?
     3. Two cases reach the same loopback URL without using `requests.get`.
        Find them.
     """),
    ("md", """
     ### Step 3 — try to fix it

     Edit the rules. Add patterns. Then re-score.

     Track both numbers, not just detection. The exercise is not "get detection
     to 100%" — it is to feel the trade.
     """),
    ("py", dedent("""\
        import re
        my_rules = dict(RULES)

        # Your turn. For example:
        # my_rules['urllib'] = re.compile(r'\\burllib\\b')
        # my_rules['socket'] = re.compile(r'\\bsocket\\b')
        # my_rules['loopback'] = re.compile(r'127\\.0\\.0\\.1|localhost')

        print_scorecard(run_scorecard(rules=my_rules))
        """)),
    ("md", """
     #### ✏️ After your edits

     | | Before | After |
     |---|---|---|
     | detection rate | | |
     | false-positive rate | | |

     Did one improve at the other's expense? That is the shape of this problem,
     and no amount of regex removes it.
     """),
    ("md", """
     ### Step 4 — the failure that isn't about regex at all

     Suppose your filter were perfect: every variant caught, zero false
     positives. The model still emits `requests.get(...)`.

     **Something downstream still has to decide whether to run it.**

     A filter inspects text. It never sees the action. The control that would
     have stopped this is the one that asks *is this code allowed to reach that
     host?* — and that question is answered by authorization, not by pattern
     matching.
     """),
    ("md", """
     ### Optional — watch it happen

     On the speaker's machine, a service is listening on the loopback URL. When
     the poisoned model fires and something runs the output, a beacon lands.

     We are not running model output here. But you can see what the endpoint
     sees:

     ```
     python -m service.mock_endpoint
     curl http://127.0.0.1:8080/workshop-demo
     ```

     One line in a log. In production that is one line among millions, and
     nobody is looking at it.
     """),
    ("md", """
     ### Where this leaves you

     | Control | Catches | Misses |
     |---|---|---|
     | benchmarks (NB 02) | bad models | targeted behaviour |
     | artifact scanning (NB 03) | unsafe formats | unsafe weights |
     | weight probes (NB 04) | known attack shapes | novel recipes |
     | prompt filters (NB 05) | known strings | everything else |

     Each one is worth having. None of them is the thing that saves you.

     **Assume the model is compromised and constrain what it is allowed to do.**
     """),
]


# ══ Builder ═══════════════════════════════════════════════════════════════════

def _cell(kind: str, src: str) -> dict:
    src = dedent(src).strip("\n")
    lines = src.splitlines(keepends=True)
    if kind == "md":
        return {"cell_type": "markdown", "metadata": {}, "source": lines}
    return {"cell_type": "code", "execution_count": None, "metadata": {},
            "outputs": [], "source": lines}


def _colab_badge(name: str) -> dict:
    """Badge cell, first in every notebook, so a shared .ipynb self-links."""
    slug = REPO_URL.split("github.com/")[1]
    url = f"https://colab.research.google.com/github/{slug}/blob/main/lab/notebooks/{name}"
    return _cell("md", f"[![Open In Colab]"
                       f"(https://colab.research.google.com/assets/colab-badge.svg)]({url})")


def build(name: str, cells: list[tuple[str, str]]) -> Path:
    nb = {
        "cells": [_colab_badge(name)] + [_cell(k, s) for k, s in cells],
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "colab": {"provenance": [], "toc_visible": True},
            "accelerator": "None" if name in CPU_ONLY else "GPU",
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    NB_DIR.mkdir(parents=True, exist_ok=True)
    path = NB_DIR / name
    with open(path, "w") as f:
        json.dump(nb, f, indent=1)
        f.write("\n")
    return path


NOTEBOOKS = {
    "01_poison_and_finetune.ipynb": NB1,
    "02_evaluate_backdoor.ipynb": NB2,
    "03_pickle_and_modelscan.ipynb": NB3,
    "04_peftguard_probe.ipynb": NB4,
    "05_firewall_experiment.ipynb": NB5,
}

# Notebooks 3-5 are CPU-only; saying so in the metadata stops Colab from
# holding a GPU slot the participant will need again in notebook 01.
CPU_ONLY = {"03_pickle_and_modelscan.ipynb", "04_peftguard_probe.ipynb",
            "05_firewall_experiment.ipynb"}


def main() -> None:
    for name, cells in NOTEBOOKS.items():
        p = build(name, cells)
        n_md = sum(1 for k, _ in cells if k == "md")
        n_py = len(cells) - n_md
        print(f"wrote {p.name:<34} {len(cells):>2} cells ({n_md} md, {n_py} code)")


if __name__ == "__main__":
    main()
