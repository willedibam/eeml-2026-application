# Figure design — reasoning, style spec, open questions

## Tooling decision

| need | tool | why |
|---|---|---|
| data figures | matplotlib + `figures/style.py` | regenerable from result JSONs, diffable, consistent by construction |
| schematics | matplotlib now; **TikZ if the poster is LaTeX** | TikZ matches poster typography exactly and is parameterised |
| graph layout | hand-placed / networkx for layout only | networkx's default rendering is not poster quality |
| rejected | draw.io, Illustrator | typography drifts from LaTeX, cannot be regenerated, does not diff |

Fig 1 is matplotlib because its content is half real data (the SPI matrices are
genuine outputs of the statistics they claim to be, computed from a real VAR(1)
chain). A pure schematic with no data would be better in TikZ.

## Style spec — one rule, three colour roles

Defined once in `figures/style.py`; no figure script sets rcParams.

- **signed quantities** -> one diverging map, centred at zero (`RdBu_r`)
- **magnitudes** -> one sequential map (custom pale-to-deep blue)
- **emphasis** -> one accent (`#c1440e`), used for nothing else

Two sequential maps in one poster is the most common consistency failure, so
there is deliberately only one. Both maps are reasonable under colour-vision
deficiency; `icefire` was considered and dropped because its endpoints separate
poorly for deuteranopes at low saturation.

Other invariants: no top/right spines, no gridlines, direct labelling instead of
legend boxes wherever a curve can be labelled in place, 8pt base, lowercase
panel titles, bold lowercase panel letters.

`style.LAYERS` is load-bearing: the same three colours identify the same three
statistics in every panel and every later figure. That is what makes the matrix
stack and the graph stack read as one object rather than two illustrations.

## Fig 1 — panel rationale

**Reading order a -> b -> c, one idea per panel.** At A0 a panel gets ~3 seconds.

- **(a) the data.** Heatmap rather than line traces: 5 rows read as a *set* of
  channels, and hairline white gaps reinforce that they are separate series, not
  an image. The generating process is a real VAR(1) chain with self-persistence
  0.62 — at 0.35 the traces looked like white noise and the lagged propagation
  the panel exists to show was invisible.
- **(b) the vocabulary.** A receding deck, not a grid of three, because the
  object is a stack of depth K; the reader must leave believing there are
  hundreds. Three genuine statistics chosen to *disagree* with each other about
  which pairs are coupled — that disagreement is the argument for a vocabulary.
- **(c) the same stack as graphs.** Identical geometry, colours and depth cue to
  (b), so the rhyme carries the claim: an edge is a K-VECTOR, not a scalar.
  Pentagon layout so all 10 pairs get a visible chord; a left-to-right chain
  would hide exactly the pairs the vocabulary is about. Edge width uses a steep
  exponent (w^2.4) — at a gentler one all 10 chords stay visible and three
  superimposed decks read as noise.

**The figure stops before any learning.** `w`, softplus and top-d belong to the
method figure. Panel (c) is already one complete idea.

## Known imperfections in the current draft

- Panel (c)'s content sits slightly low-right in its axes; the three decks could
  be optically centred better.
- The rear matrix plane desaturates toward grey (alpha over white), which costs
  some layer-colour identity. Alternative: no alpha, depth carried by frame
  colour and offset alone.
- Panel (a) is visually heavier than (b) and (c).

## Open design questions — need a decision

1. **Directedness is not shown.** Two of the three statistics drawn are
   symmetric, so panel (c) has undirected chords. But directedness is central
   (fork and collider are Markov-equivalent under any symmetric statistic).
   Options: (i) make one layer explicitly directed with arrowheads, (ii) leave
   Fig 1 undirected and let the case-study figure carry directedness. Leaning
   (i) — it costs one layer and pre-empts the objection.
2. **Poster LaTeX class** fixes fonts and therefore whether schematics move to
   TikZ. Needed before Fig 2.
3. Whether Fig 0 (motivation) is empirical (`latent-directed` vs `spi-mpnn`:
   equal accuracy, unequal insight) or conceptual. Recommend empirical.
