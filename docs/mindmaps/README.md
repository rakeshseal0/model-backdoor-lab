# Workshop mindmaps

One file per part of the deck, written as a rehearsal aid: the structure, the
lines worth landing, the questions to expect, and the transition out.

Each file has a Mermaid mindmap at the top for the shape, then slide-by-slide
notes for the detail. Built as we work through the deck together.

| Part | Slot | Mindmap | Status |
|---|---|---|---|
| I — The model-scanning landscape | 8–23 | [part-1-scanning-landscape.md](part-1-scanning-landscape.md) | ✅ |
| II — Implant and measure a controlled backdoor | 23–58 | [part-2-implant-and-measure.md](part-2-implant-and-measure.md) | ✅ |
| III — Artifact safety is not behavioral safety | 58–72 | — | pending |
| IV — Weight-level evidence with PEFTGuard | 72–87 | — | pending |
| V — Why an AI firewall can fail | 87–102 | — | pending |
| VI — Enterprise-grade model assurance | 102–120 | — | pending |

## The spine

Every part is an instance of one idea, introduced in Part I slide 3:

> **safe to load ≠ safe to query ≠ safe to authorize**

| Boundary | Question | Part |
|---|---|---|
| 01 LOAD | Can opening the file execute attacker code? | III |
| 02 QUERY | Do the weights hide a conditional response? | II, IV |
| 03 ACT | Can output reach tools, files, data, networks? | V, VI |

## Rendering

Mermaid mindmaps render on GitHub and in VS Code with the *Markdown Preview
Mermaid Support* extension. Without it the block shows as plain text — the
indented outline is still readable, which is why the detail is also written
out longhand below each map.
