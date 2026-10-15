# Niru ML site

The public site for Niru ML and AMLO. It is a static Astro site. The home page
introduces the AGI and recursive self-improvement research thesis; `/lab/` holds
the detailed design illustrations, simulated claim record and capability ledger.
The conversation page uses email on every host. The former Cloudflare intake
Function remains dormant source and is not linked from the site.

## Local development

Requirements: Node 24 or newer, npm 11 or newer, Python 3.15 for the
deployment checks, and uv for the spectral sweep.

```sh
make bootstrap
npm run dev
```

The site ships no analytics or third-party embeds. It uses small, local scripts
for the home atlas and optional lab drawings. It is
text set in one serif family on a light page, with a dark mode that follows
the reader's system setting. Formulas are MathML, typeset by KaTeX at build
time. Plots are inline SVG, drawn at build time from the data they show.

## Checks

```sh
make check
```

This runs the unit tests for the request handler and the search-trace math,
the formatter, the production build, the browser suite (axe accessibility in
light and dark mode, keyboard navigation, the no-JavaScript email contact, the 404
route, request budgets, screenshots), and the Python tests over the built
output and the header contract.

A live deployment is checked separately, read-only, against `public/_headers`:

```sh
NIRU_DEPLOYED_BASE_URL=https://preview.example.pages.dev make check-deployed
```

## Build and hosting

- `npm run build` writes the site to `dist/` for Cloudflare Pages and then
  regenerates `docs/`, the GitHub Pages fallback. Both outputs offer the
  same email contact, so no form endpoint is required.
  Do not edit `docs/` by hand.
- `functions/` preserves the dormant `POST /api/preview` handler. It cannot write outside
  production, and production writes require the readiness flag and secrets.
- The mark is one woodcut drawing, generated in `src/lib/brand.ts`. The site
  build serves `/favicon.svg` and `/mark.svg` from it. `npm run brand`
  regenerates the ICO, the PNG icons and the social card from the same drawing.

## The algebra window

The window on the lab page draws two models as AMLO's 0.0.14 design defines them, in the
visual language of Abbott and Zardini's neural circuit diagrams (arXiv:2604.07242):

- **Monad:** the training step, the model and its loss, one block, attention and SwiGLU.
- **A toy DeepSeek-V4.1-Flash:**
  - the 40-layer strip, with Reuse runs folded and the values that pass between layers drawn
    above it;
  - one layer of each kind around its four residual streams;
  - the hyper-connection step;
  - attention, its core, compressor and indexer, for each kind;
  - the MoE, routed as the reference routes, and Engram.
- **Element lattices** under both: rope pairs, the causal mask, the window band, the compression
  stride, visible entries, routing, Sinkhorn and the candidate pool.

A box that is a call opens in place into its definition; the window remembers the box each view
came from. The note under the window says what each view rests on (`View.evidence`): AMLO's 0.0.14
design, for the models and figure B; AMLO's built work rules and its design as the window models
them, for C; the window's count from Monad's definitions, for D's members; and the site's spectral
study, for the measured check, whose kicker names the study.

Figure B, the search's moves on Monad's term, plays as a film. It starts by itself once the window
is half in view, or fills half the screen where it is taller than the screen, and pauses when the
window leaves the view or the tab hides. Under
`prefers-reduced-motion` it never starts by itself: the offer under the model view opens it as
still chapters. It runs about two minutes, in ten chapters, and rests on the last with Replay:

- the population, its tournament and the parent;
- the parent's 57 legal sites, lit class by class on its definitions as the registry lists them;
- one move of each class (replace, insert, delete, binding and shape), each told in nine beats:
  the class, site, choice and reach drawn, the rewrite, admission's seven gates, the verdict, and
  the child joining the population as its youngest member;
- two moves that commute to one architecture, and admission refusing the second by identity;
- a move inside the toy DeepSeek's nested decoder Repeat.

While a film plays by itself, its caption under the window is silent to a screen reader; the
reader's first press, key or slider in the window, Tab aside, makes it announce its changes again.

Each beat says one plain sentence. A move's five draws appear as chances ("5 in 18"), and their
product as odds in plain words ("about 1 in 970,000"). Under the window sit a rail, one segment per
chapter filling as the film plays, then Pause, See details, Back and Next; Space plays and pauses,
and the arrow keys step chapters. See details stops the film at once, and the sentence gives way
to the chapter's ledger: each factor's exact fraction and the recorded probability, with the full
caption below the window. While details are on,
the rail marks each chapter's end, and the film stops there too as the ledger comes; the first
button then reads Continue. Hide details gives the sentence back and plays on. A stop for details
rests in ink: its lights go out as the ledger comes. A move's definition opens into its full view,
as rewritten.

It is an illustration under the registry's rules, not a recorded search, and every frame says so.
Three things are declared for it: the population's ranks, the range of F, and the layers each
`one` or `some` selects. Sites are counted on the definitions as drawn, with a norm as one node.
Effects accompany actions only: light on what is drawn or chosen, a sweep or beam over the layers
a change reaches, sparks where a node arrives or leaves, the gates lighting as a proposal passes,
and a seal at each verdict. At rest the drawing stays ink on paper. Nothing starts or stops in one
frame: as a beat begins, what the one before lit or added dims, and the band's lettering
crossfades, what stays staying put; a unit test steps both films at 60 Hz and fails on a cut.

Figure C, one term with many functors, opens from Monad's block and plays once as a film of six
chapters, about 50 seconds. Under the block it draws what AMLO's 0.0.14 design derives from it,
each as a band aligned piece by piece:

- the derived backward, run from the output to the input, with a residual wire to each value a
  pullback reads; each call's pullback opens into its operations, with the work of each. As the
  design draws them, only recomputed values are dashed: under StoreAll the wires are solid, and
  under Remat they are dashed;
- the cost of each piece, forward and backward, by AMLO's work rules, one unit per arithmetic,
  transcendental and compare operation: an exact count of the program's operations, not hardware
  time;
- the plan `ampere-bf16`: 13 casts, and the residual policy, StoreAll or Remat, which moves memory
  and plan work but not the mathematics. The casts are applied before the backward is derived, so
  each contraction keeps the bf16 operand it reads, the weights' bf16 copies
  among them, stacked per layer: StoreAll keeps 62.0 MB per layer, and Remat 1.0 MB. Remat
  recomputes what the backward reads and what that depends on, 2,446,848,000 operations per layer,
  since a derived program has no dead nodes. Casts are plan work, but AMLO's built work rules give
  them no price, so the plan counts none, and says so;
- the identities, each named with what it hashes, and which edit moves which. The column shows no
  hash values: AMLO does not compute these identities yet, and the site makes none of its own. When the reader chooses Remat, or the film
  lights a plan edit, a verdigris dot marks execution as moved.

The last chapter parts the block into its two halves and joins them again: the backward, cost and
plan of the block are its halves' joined, the backward's in reverse order; the identities hash the
whole block. The film then rests in ink. At the end, Show toggles each band and Residuals switches
the policy; a choice made while the drawing moves applies when it arrives. Work is per layer for
one sequence of 1,024 tokens, declared for the figure, and every frame says these are AMLO's rules
as the window models them, not AMLO's output.
Softmax and the loss are composed from a row maximum that is a stop-gradient value, then
subtract, exp, sum and divide, or log and subtract for the loss's log-softmax; their backward is
derived from those nodes. The window scores the loss at every position, as Monad's drawn loss
reads a label for each. AMLO's own pricing of the block, the final norm, the head and the loss
gives every count the window gives. One choice is the window's own: where the plan's rule for the attention scores refers back to itself, the scores
take f32 (`plan.ts`).

Figure D, a family, opens from the model view. Five sliders move Monad's controls over ranges
declared for the figure (v 4,096 to 65,536, L 2 to 96, h 1 to 16, k 16 to 128 in steps of 2, f 256
to 4,096 in steps of 256); d = h·k and p = k/2 are derived, never set. Each weight is drawn as a block whose edges follow its axes, one step per doubling, and each
axis is one colour wherever it appears, so a control resizes all its edges together. Beside them,
three lanes recount the member from the definitions: its parameters, and its work per token for
one sequence of 1,024 tokens, forward and with the derived backward; the update, whose share per
token depends on a batch, is left out. The lanes are on a log scale, so every count stays legible;
under them one bar, on a linear scale, splits the parameters into E, attention and SwiGLU, each
named with its share, and its caption names the norms, under a quarter of a percent at any setting.
While the drawing moves to a member, the lanes and the bar already print the setting the sliders
show. The measured check opens from the family: the spectral study's ratios
as a measured diagnostic panel, redrawn from its data file. Its note says that AMLO's system
contract asks for a transfer certificate, that ΔW is the change of W over training, and that each
point is the mean of the seeds that trained, of three; it is a toy study and not a test of AMLO's
training path.

- `src/ink/engine.ts` projects a 3D scene and draws it as ink: SVG for the still image built with
  the page, Canvas2D for the live window. Tiled axes recede in depth, so heads and experts are
  drawn as one sheet each, on paper that stays opaque: a deeper sheet prints fainter, never
  translucent. A panel or a sheet and what is printed on it are one group, drawn whole and back to
  front. Every view is designed at the plate's full width (`REF`); at any other size, and through
  a zoom, it scales as one picture, lettering included. A figure's steps share one stage
  (`View.stage`), so its camera never moves between them.
- `src/ink/kit.ts` holds the glyphs and lays rows out edge to edge. A view is built twice
  (`models.ts`), once to learn its scale, so every box that holds text is sized to the text.
- `src/ink/monad.ts`, `src/ink/deepseek.ts`, `src/ink/deepseek-parts.ts` and `src/ink/lattice.ts`
  build the views. `src/ink/dsv41.ts` holds the toy's configuration, its schedule (checked against
  the reference's layer arrays) and its parameter count (112,726,448, as the design gives).
- `src/ink/moves.ts` computes every number figure B shows. It holds exact fractions in `bigint`, a
  small term model of Monad's drawn definitions, and the commands B uses. From these it derives
  the sites per class, the uses of each definition, parameter counts and a structural identity,
  and it says each move, factor and probability in plain words.
  `src/ink/rewrite.ts` draws the figure's views. Each view that changes the term has a before view
  in the same layout. `src/ink/morph.ts` carries keyed pieces from one view to the next in three
  movements: what goes fades, what moves travels, what comes fades in. Ink printed in a box moves
  with its box; a node being inserted or deleted grows out of the wire or shrinks back into it.
- `src/ink/reel.ts` makes a film from chapters and beats: their effects, the band under the
  drawing, and `frameAt(t, options)`, one pure function from time (and the reader's settings) to a
  frame. `src/ink/film.ts` is figure B's film and `src/ink/functors.ts` is figure C's.
- `src/ink/derive.ts` models AMLO's work rules: Monad's block as a program over arrays, its
  derived backward, and the work of each node. `src/ink/plan.ts` resolves `ampere-bf16` over the block, the
  memory and work of each residual policy, and the identity components each edit moves.
  `src/ink/family.ts` recounts a member of the family, and `src/ink/fam.ts` draws figure D.
- `src/ink/window.ts` provides the optional lab drawing; `ResearchAtlas.astro` provides the home atlas. DeepSeek's views and each figure are
  separate chunks, loaded when first used. `tests/browser/runtime.spec.ts` holds them, gzipped as
  they travel, to 64 KiB with figures B, C and D opened, and DeepSeek's chunk to 14 KiB. Each film keeps its own clock. Without JavaScript,
  Monad's still image stays. The trail has its own row and keeps to one line, scrolling sideways
  when it is longer than the plate, so the stage never moves between levels. Where the stage is
  narrower than the plate and scrolls sideways, as on a phone, the page repeats D's three counts as
  text beside the sliders, and a film's stage scrolls at each beat to what the beat lights, unless
  the reader scrolled it during the beat before; the beat's sentence is repeated as text under the
  stage, and a stop for details scrolls the stage back to the ledger. A touch screen does not show
  the key hint. A film builds each of its views once and keeps it, so a frame only composes.
- `tests/browser/ink-text.spec.ts` sets every drawing's text with the site's fonts and fails if a
  label touches another label, a line or a dot, breaks out of its box (a glyph printed on a panel
  or a sheet must stay inside it), or comes within 12px of the frame. It checks every view, every
  beat of each film a quarter, half and three quarters of the way and at its end, each chapter's
  ledger, each jump between chapters part way, and each state of C the reader can set at its end;
  ink fading in or out below 30% is left aside. `tests/ink.test.mjs` checks the counts, the
  schedule, the views' links and the fonts' coverage of every character. It pins each of figure B's
  fractions against a derivation by hand, the plain words for each, and that a film frame is a pure
  function of its time; every effect in a film acts on something its beat draws. For C and D it
  pins the work rules on small programs worked by hand, the block's totals, the plan's casts and
  memory, the edits' identities against the design, the family's recount, that a
  control resizes every edge on its lane, and the measured check's rows against the spectral
  study's data file. Each film's last frame rests in ink, and C dashes only what Remat recomputes.

## The claim record and the search trace

The lab page sets one claim record in type. Every value in it comes from
`src/data/traces/simulated.json`, a simulated search trace in the site's
`niru.search-trace/1` format (`src/lib/trace/schema.ts`), and the sentence
under it comes from `src/data/traces/outcomes.json`.

- `npm run trace` writes the traces from `src/lib/trace/simulate.ts`, which
  applies AMLO's search rules (admission gates, the matched-compute planner,
  the paired BCa gate with Benjamini–Hochberg, the surrogate screen and its
  sampled audit, the freeze and the sealed confirmation) to a synthetic loss
  model. The page says that the run is simulated.
- `src/lib/trace/validate.ts` checks a trace against those rules and reruns
  its arithmetic. This is a check of the site's own traces, not an AMLO
  verifier. The unit tests run it on the committed traces and confirm that
  they regenerate exactly from their seeds.

The site does not read from AMLO. The format is the seam for a later
integration: it follows AMLO's proposal event log (`proposal_events.jsonl`)
and uses AMLO's field names wherever AMLO records the same quantity. Fields
that AMLO does not record today are marked as enrichments in the schema. A
later export would project from AMLO's own event and selection records.
Setting `source.kind` does not make a simulated trace an AMLO record.

## The spectral study

`/research/spectral-diagnostics/` reports a measured toy study, not a test of
AMLO's training path. Its three figures (`src/components/SpectralFigures.astro`)
and its tables are drawn at build time from the data file, and the note
places each figure with an HTML comment such as `<!-- figure:loss -->`.

- `npm run spectral` runs `scripts/spectral-certificate.py` (Python 3.15 and
  NumPy 2.5.4 through uv; about 76 minutes on 6 CPU cores, one worker per
  physical core) and writes `src/data/spectral-certificate.json`. The output
  repeats bit for bit on one machine and NumPy version; another NumPy rounds
  some runs differently, so the numbers move slightly but no verdict does.
- The data records the SHA-256 of the script that wrote it. A change to the
  script's computation needs a new sweep. A change that leaves the
  computation alone (a comment, a format) keeps the old bytes in
  `scripts/spectral-history/`, named by their digest, so the data still binds
  to real source; `tests/python/test_spectral_provenance.py` checks that the
  kept copy and the current script are the same program, docstrings aside.
- The unit tests check the data without a sweep: every scheme, width, grid
  rate and failed run is present, and each ratio and pass flag is recomputed
  from the recorded values.

## Current hosting and this overhaul

The public host observed on 14 October 2026 is GitHub Pages. It does not apply
Cloudflare `_headers` or `_redirects`, and it does not run Pages Functions.
Header-contract tests validate the retained Cloudflare configuration, not live
GitHub response headers. The deployed-header checker is only applicable to a
host configured to satisfy that contract; it does not verify artifact identity.
The scheduled Cloudflare smoke check remains opt-in and is disabled by default.

Publication and a host migration require separate human authorization.
