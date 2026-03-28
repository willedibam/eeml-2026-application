# Figure Specifications — EEML 2026 Extended Abstract

**For production in Inkscape/Illustrator. Data panels generated in Python (matplotlib/seaborn), composed in vector editor.**

---

## Figure 1: Method Overview

**Purpose:** Show the full pipeline from raw MTS to classification, emphasising what is new (the SPI vocabulary and learned w).

**Layout:** Single row, 4 panels (A–D), full page width (~170mm for two-column, or ~\textwidth for single-column). Height ~55–65mm. White background, minimal borders.

**Font:** Sans-serif (Helvetica/Arial), 7–8pt for labels, 9pt for panel letters.

---

### Panel A — Synthetic Task (width: ~35mm)

**Content:**
- Three small directed graphs arranged vertically, each labelled: **Chain**, **Fork**, **Collider**.
- Each graph: 3 filled nodes (motif, dark blue #1565C0) + 7 open/faint nodes (nuisance, light grey #BDBDBD).
- Motif edges: bold directed arrows (2pt, dark blue).
- Nuisance self-loops: thin grey arcs (0.5pt, #E0E0E0) or omit for clarity.
- Chain: 0→1→2. Fork: 0→1, 0→2. Collider: 1→0←2.
- Below all three: one compact carpet plot (heatmap, T×M, ~25mm wide × 10mm tall) with axis labels "T" and "M". Use a diverging colormap (RdBu). This shows "what the data looks like" — one example only.

**Design notes:**
- The three graphs should be visually compact — the reader identifies the structural difference in <2 seconds.
- Highlight that chain and fork have the same skeleton (same undirected edges) — perhaps draw the skeleton in grey first, then overlay directed arrows in blue.

---

### Panel B — SPI Descriptor Tensor (width: ~45mm)

**Content:**
- Left: One MTS carpet plot (same as Panel A but slightly larger) with a brace/arrow labelled "`pyspi`" pointing right.
- Right: A vertical stack of 5–6 small M×M heatmaps (~8mm each), each labelled with an SPI name:
  - Top: "$r$" (Pearson) — visually **symmetric** (warm colormap, Sequential/Oranges)
  - Middle: "$\rho$" (Spearman), "MI", "Coh" — symmetric, increasingly faint
  - Bottom two: "**SGC**" and "**TE**" — visibly **asymmetric** (upper triangle ≠ lower triangle). Use a diverging colormap (RdBu) to make asymmetry obvious.
  - Vertical dots (...) between some to indicate K ≈ 125 total.
- Far right: One edge (i→j) highlighted across all heatmaps with a coloured bracket, forming the vector $\mathbf{E}_{ij} = [r_{ij}, \rho_{ij}, \ldots, \text{SGC}_{ij}, \text{TE}_{ij}] \in \mathbb{R}^K$. Draw this as a vertical coloured bar extracted from the stack.

**Design notes:**
- **The asymmetric vs symmetric contrast is the most important visual in the paper.** Make it unmissable. Consider a small annotation: "symmetric" (with ↔ icon) for Pearson, "**directed**" (with → icon) for SGC/TE.
- The colour in the "extracted edge vector" bar should match the heatmap cells it came from.

**Data source:** Load any instance from `data/260327_eeml/var-chain/M10_T500_I0/spi_mpis.npz`. Plot the actual matrices for Pearson (`xcorr_Pearson`), SGC (`sgc_parametric_mean_fs-1_fmin-0-25_fmax-0-5_order-1`), TE (`te_kraskov_k-4`).

---

### Panel C — Learned Graph Construction (width: ~45mm)

**Content:**
- Left: The edge vector $\mathbf{E}_{ij}$ (vertical bar from Panel B) enters a dot product with $\mathbf{w}$ (shown as a small horizontal weight vector with colour-coded entries — causal entries tall/dark, others short/faint).
- Formula: $A_{ij} = \text{softplus}(b + \mathbf{w}^\top \mathbf{E}_{ij})$ written compactly.
- Centre: The resulting sparse directed graph (M=10 nodes, top-5 edges shown as bold directed arrows with varying thickness = edge weight $A_{ij}$). Motif nodes highlighted. Non-selected edges shown as very faint dotted lines or omitted.
- Below graph: "top-$d$ sparsification" annotation.

**Design notes:**
- The weight vector $\mathbf{w}$ should visually echo the family bar chart from Figure 2B — tall bars for causal SPIs.
- Show that the graph is directed (arrowheads).

---

### Panel D — MPNN + Classification (width: ~40mm)

**Content:**
- Schematic MPNN: 2 rounds of message passing shown as the graph from Panel C with arrows indicating message flow along edges.
  - Round 1: messages flow along edges, node colours update.
  - Round 2: further update.
- Edge features: small annotation showing $\phi(\mathbf{E}_{ij})$ conditions messages (an MLP icon on one edge).
- Global pooling: all node representations aggregate (converging arrows) to a single graph vector.
- Classifier: graph vector → softmax → predicted class icon (chain/fork/collider graph miniature, with a checkmark on the correct one).

**Design notes:**
- Keep this schematic, not detailed. The reader should see: "messages propagate → pool → classify."
- Don't try to show the full MLP architecture — a rounded rectangle labelled "MPNN (2 layers)" is sufficient.

---

### Panel connections
- Light grey flow arrows between panels: A→B→C→D.
- Panel letters (A, B, C, D) in bold 10pt, top-left of each panel.

---

## Figure 2: Results

**Purpose:** Show (a) sample efficiency, (b) interpretable learned w, and optionally (c) the symmetric limitation.

**Layout:** Two or three panels, full page width. Height ~50–55mm.

---

### Panel A — Sample Efficiency Curves (width: ~85mm if 2-panel, ~55mm if 3-panel)

**Plot type:** Line plot with shaded error bands (±1 std).

**X-axis:** $n$/class, log scale: {20, 50, 100, 200, 500, 1000}. Label: "Training samples per class".

**Y-axis:** Macro F1 (0–1.0). Label: "Macro F1". Grid lines at 0.33 (chance) and 0.67 (symmetric ceiling) — dashed, grey, annotated.

**Lines (7 models), grouped by colour:**
| Model | Colour | Linestyle | Marker |
|---|---|---|---|
| SPI-MPNN | #D32F2F (red) | solid, 2pt | ● |
| Fixed-SPI | #FF7043 (orange) | solid, 1.5pt | ■ |
| MLP-Mix | #FFB300 (amber) | solid, 1.5pt | ▲ |
| Correlation | #1976D2 (blue) | dashed, 1pt | ◇ |
| Latent | #7B1FA2 (purple) | dashed, 1pt | ▽ |
| Shuffled | #78909C (blue-grey) | dotted, 1pt | × |
| Node-Only | #9E9E9E (grey) | dotted, 1pt | + |

**Legend:** Outside plot (right side or below), grouped: "SPI vocabulary" (red/orange/amber) and "Controls" (blue/purple/grey).

**Data source:** `results/sample_efficiency_definitive_w60_new_results.json` (when available). Interim: `results/sample_efficiency_definitive_new_results.json` for non-SPI-MPNN models, `results/sample_efficiency_sens_w60_new_results.json` for SPI-MPNN.

**Key visual:** Two clearly separated bands. Top band (SPI models) rises from ~60% to ~100%. Bottom band (non-SPI) stays near 33%. Correlation slowly rises to ~55%.

---

### Panel B — Learned Statistical Signature (width: ~50mm if 2-panel, ~35mm if 3-panel)

**Plot type:** Horizontal grouped bar chart.

**Y-axis (categorical):** SPI families: Causal, Spectral, Linear, Information, Distance, Rank. Ordered by L2 norm (causal at top).

**X-axis:** $\|\mathbf{w}_g\|_2$ (L2 norm of w within family). Label: "Family weight (L2)".

**Bars:** Colour-coded by family.
| Family | Colour |
|---|---|
| Causal | #D32F2F (red, matching SPI-MPNN) |
| Spectral | #1976D2 (blue) |
| Linear | #388E3C (green) |
| Information | #7B1FA2 (purple) |
| Distance | #FF8F00 (amber) |
| Rank | #78909C (grey) |

**Annotation:** Label top 3 individual SPIs next to the causal bar: "SGC (0.25–0.5Hz)", "GC (Gaussian)", "TLMI".

**Data source:** Learned w from `results/sample_efficiency_definitive_w60_new_results.json`, n=500, averaged across 10 seeds. Compute per-family L2 using `assign_spi_families()`.

---

### Panel C (optional) — Symmetric Limitation Visualisation (width: ~35mm)

**Plot type:** 3×2 grid of small M×M heatmaps.

**Content:** For one chain instance and one fork instance:
- Row 1: Pearson |r| matrix — visually identical for chain and fork (same skeleton → same correlations).
- Row 2: SGC matrix — visibly different (asymmetric structure differs between chain and fork).
- Column labels: "Chain", "Fork". Row labels: "|r| (symmetric)", "SGC (directed)".

**Purpose:** Makes the Markov equivalence argument visual — "these look the same to Pearson, different to SGC."

**Data source:** Load one chain and one fork instance from `data/260327_eeml/`, compute Pearson and extract SGC from npz. Pick instances with clear visual contrast.

**Design note:** If space is tight, this can go to appendix. But it's a powerful visual argument and worth including if it fits.

---

## Table 1 (in-text, Section 4)

Already in the draft. Full 7-model × 6-n comparison. Bold best per row. Include both F1 and std.

---

## Appendix (optional, not required reading)

- Full sensitivity table (all hyperparameter configs tested)
- Per-seed strip plots at n=100 showing bifurcation
- Confusion matrices at n=100 and n=500
- Full learned w bar chart (all 125 SPIs, not just family-level)

---

## Production Notes

1. Generate data panels in Python (matplotlib, `savefig(..., format='pdf', bbox_inches='tight')`).
2. Import PDFs into Inkscape/Illustrator.
3. Add annotations, flow arrows, panel letters in vector editor.
4. Export final figures as PDF (for LaTeX) or high-res PNG (300+ dpi).
5. Figures should be self-contained — a reader should understand the method from Figure 1 alone, and the results from Figure 2 alone.
