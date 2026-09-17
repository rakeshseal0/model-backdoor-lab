# The slide deck

Source of truth is [`presentation.md`](presentation.md). `presentation.html` is
generated from it — **edit the markdown, never the HTML**, or the next build
silently discards your changes.

The custom Reveal.js visual system is in [`theme.css`](theme.css). Reveal.js is
pinned locally so the deck runs without a CDN.

## Quick start

```bash
npm install
npm run build
npm run serve
```

Open <http://127.0.0.1:8765/presentation.html> and press `S` for speaker view.

## Render with Pandoc

Reveal.js HTML deck — this is what `npm run build` runs:

```bash
pandoc presentation.md \
  --standalone \
  --to=revealjs \
  --slide-level=2 \
  --output=presentation.html \
  --css=theme.css \
  -V revealjs-url=node_modules/reveal.js \
  -V theme=white \
  -V transition=fade \
  -V center=false \
  -V controls=true \
  -V progress=true \
  -V slideNumber=true \
  -V hash=true
```

PowerPoint:

```bash
pandoc presentation.md --slide-level=2 --output=presentation.pptx
```

The custom CSS theme applies to the Reveal.js HTML deck only. PowerPoint export
preserves slide content and hierarchy but uses PowerPoint's reference theme.

PDF through Beamer, if a TeX distribution is installed:

```bash
pandoc presentation.md --to=beamer --slide-level=2 --output=presentation.pdf
```

Pandoc 3.11 and Reveal.js 5.2.1 were used to render and validate the included
`presentation.html`.

## Facilitation notes

Speaker notes use Pandoc's `::: notes` fenced-div syntax. Reveal.js displays
them in speaker view; support in other output formats depends on the writer and
template.

Per-part rehearsal notes, mindmaps, and the answer keys for the interactive
moments live in [`../docs/mindmaps/`](../docs/mindmaps/).

## Assets

| File | Used by |
|---|---|
| `assets/model-package.svg` | Part I — a model is not just weights |
| `assets/three-boundaries.svg` | Part I — the three boundaries |
| `assets/poisoning-flow.svg` | Part II — training pipeline |
| `assets/secure-model-pipeline.svg` | Part VI — secure model promotion |
| `assets/assurance-lifecycle.svg` | Part VI — assurance lifecycle |
| `assets/rakesh-seal-speaker.png`, `assets/rakeshseal-site-qr.svg` | Speaker slide |
