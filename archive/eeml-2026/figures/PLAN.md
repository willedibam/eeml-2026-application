# Poster figure suite — full pass

A0, two columns, **figure-led**: every block below is a figure that carries its
own argument. Text is a caption and a one-line takeaway, written last.

Reading order is a build: *problem -> instrument -> does it work -> what it says
-> when it lies*. Nothing appears before it has been defined.

| col | block | figure | data | status |
|---|---|---|---|---|
| 1 | 1 | **F0** where does the graph come from? | none (conceptual) | to build |
| 1 | 2 | **F1** series -> K statistics -> K-vector edges | local, illustrative | **built** |
| 1 | 3 | **F2** the learned probe | none (schematic) | to build |
| 1 | 4 | **F3** the task, and why direction is necessary | local | to build |
| 2 | 5 | **F4** does it work | `30seeds_main`, `vocab_vs_latent`, `r0_base_s*` | to build |
| 2 | 6 | **F5** what `w` recovers | `r0297_*gl*` (10 runs) | to build |
| 2 | 7 | **F6** the signature moves when the mechanism moves | `r0297_path_gl0.002`, `r1b_lam0.0002` | to build |
| 2 | 8 | **F7** real data (TUH) | pending | blocked |
| 2 | 9 | **F8** knowing when the readout is void | `results/*.json` via `validity_report` | to build |

Seven of nine are buildable from JSONs already on the laptop. Only F7 waits.

---

## F0 — Where does the graph come from? (3 sub-panels, conceptual)

Three routes from the same MTS to a graph, as three rows sharing one input:

- **a. assumed** — threshold `|rho|`. Answer afterwards: *"edges where correlation
  exceeds theta."* Arbitrary, and says nothing about mechanism.
- **b. latent** — learned embeddings, dot-product adjacency. Answer afterwards:
  a `?`. The graph exists but names nothing.
- **c. named vocabulary (ours)** — `A = softplus(b + w'E)`. Answer afterwards:
  *"M05 4.4x enriched, M06 0.5x depleted."*

The panel's whole job is the rightmost column: **what can you say once you have
the graph.** Keep the three inputs pixel-identical so the only difference is the
route.

*Deliberately conceptual.* The empirical version ("equal accuracy, unequal
insight") was planned and dropped: accuracy is not equal (F4), so the comparison
is a result and showing it here would spoil the build.

## F1 — Series -> K statistics -> K-vector edges (3 sub-panels) **[BUILT]**

`figures/fig1_representation.py`. (a) MTS heatmap; (b) receding deck of K
pairwise matrices; (c) the same deck as graphs, lag-1 layer drawn directed.
Stops before any learning.

Outstanding tweak: panel (c) sits slightly low-right; the rear plane desaturates
under alpha.

## F2 — The learned probe (3 sub-panels, schematic)

- **a. the dot product.** One edge's K-vector beside `w` as a horizontal bar
  chart, **aligned row-for-row**, collapsing to one scalar -> `softplus` ->
  edge weight. This is the figure that makes `w` legible; everything else is
  consequence.
- **b. sparsification.** Dense weighted graph -> top-d per node -> the sparse
  directed graph the GNN actually message-passes over.
- **c. the penalty.** The K weights drawn as a column partitioned into modules,
  showing what `l1` does *within* a module and what the group term does
  *between* modules. Annotate the sqrt(|g|) factor -- without it the largest
  module wins by size alone.

Reuse F1's layer colours; the K-vector here must be visibly the same object as
F1(c)'s edge.

## F3 — The task, and why direction is necessary (2 sub-panels)

- **a. the three motifs** — chain `0->1->2`, fork `0->1, 0->2`, collider
  `1->0, 2->0`, drawn directed, with a short example series beside each.
- **b. the ceiling.** Redraw all three **undirected**. Fork and collider become
  *the same picture*. Caption: an undirected statistic sees only the skeleton,
  so no symmetric measure can exceed **2/3** on this task — a proof, not a
  measurement. Mark 2/3 on F4's axis and show `correlation` sitting at 0.59.

This panel earns the vocabulary: it shows *why* a directed statistic is needed
before any accuracy number is shown.

## F4 — Does it work (1-2 sub-panels)

- **a. sample efficiency.** F1 vs `n_train` on log-x for `spi-mpnn`, `fixed-spi`,
  `correlation`, `latent-directed`, `node-only`, `shuffled`. Direct labelling at
  curve ends, no legend box. Horizontal rules at chance (1/3) and the 2/3
  symmetric ceiling.

Three things must be visible and are all real:
- `spi-mpnn` 0.86 at n=20, 0.99 at n=700
- `latent-directed` **at chance at every n** (0.27-0.34, three seeds)
- `shuffled` at chance for n<=100 but **rising to 0.82 by n=1000**

Draw `shuffled` honestly. It bounds the topology claim to small n, and hiding it
would be the one thing a careful reader could catch us on.

Caveat that must appear in the caption: one latent architecture,
capacity-validated but not extensively tuned.

## F5 — What `w` recovers (3 sub-panels)

- **a. module enrichment** against a permutation null, ordered, with the null
  band shaded. M01 and M05 enriched, M06 depleted.
- **b. order specificity** — the headline. Four matched arms of `sgc_parametric`
  (order-1, order-auto, order-20, nonparametric), 6 SPIs each, mean |w| relative
  to the vocabulary average. order-1 **5.22x** vs order-20 **0.65x**, one dot per
  lambda run so all 10 are visible. Matched on estimator, statistic and band, so
  it cannot be a size or directedness artifact.
- **c. lambda robustness** — module shares across the lambda path (50x). Shows
  the ranking does not move where accuracy is on its plateau.

(b) is what makes this more than confirmation: nothing in a VAR(1) encodes a
preference over *model order*. (c) is what stops "you tuned lambda until it
looked right".

**If R2 (lag dose-response) runs**, (b) gains a second axis: true lag on x,
recovered order preference on y, and a diagonal would be the strongest panel on
the poster. Leave room.

## F6 — The signature moves when the mechanism moves (1-2 sub-panels)

Paired R0 -> R1b shift on the pre-registered axes: `directed & linear` **falls**
(-17 to -28pp, 3/3 reference lambdas), `directed & nonlinear` **rises**.
Slopegraph with bootstrap CIs, prediction arrows drawn *before* the measured
values so the falsifiable direction is visible.

Caption states the prediction was in the generator docstring before the result.

## F7 — Real data: TUH (3 sub-panels) — BLOCKED

- **a. controls first** — `node-only`, `shuffled`. Physically first in the panel,
  not an afterthought.
- **b. accuracy** fnsz vs gnsz, patient-disjoint splits.
- **c. recovered signature** — module enrichment, with the prior expectation
  (focal = directed outflow, generalized = symmetric bilateral coherence) marked.

If `node-only` clears chance this becomes a *scope* panel instead: the method
requires classes to differ in how channels couple, and here they do not.
Both versions are honest; only one is exciting.

## F8 — Knowing when the readout is void (1 sub-panel)

Compact verdict table from `docs/validity_report.py`: every run, its gates, and
its verdict. Highlight the three the protocol rejected — R1 (`node-only` 0.913),
Kuramoto (skeleton ceiling), R1b's first pass (signature from a chance-level
model).

Small, and the credibility panel. Most interpretability work reports a signature
with no test that the signature means anything.

---

## Style

Fixed in `figures/style.py`, see `figures/DESIGN.md`. One diverging map (signed),
one sequential map (magnitudes), one accent (emphasis only). `style.LAYERS` must
identify the same statistics across F1, F2 and F5.

## Build order

F3 -> F4 -> F5 -> F6 -> F0 -> F2 -> F8, then F7 when TUH lands.

Results figures first: they are data-driven, so they will force conventions
(axis treatment, direct labelling, null bands) that the schematics should then
inherit rather than contradict. F0 and F2 are the most style-dependent and
cheapest to redo, so they come last.
