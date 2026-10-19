# Hand-off brief: wired and wireless — reading a nervous system's signalling from its spontaneous activity with statistic-biased attention

Written 19 October 2026 for an exploring agent. Return the report described at the end. Everything below marked *verified* was checked at a primary page this session; everything else is a lead to verify. Novelty is **not established**.

## Why this brief exists

The user (a PhD student in complex systems, USyd Physics) wants a NeurIPS/ICML/ICLR workshop-quality paper, ideally stronger. Their bar: a well-motivated, important, interpretable question that survives "who cares?" and "well, obviously", with a colourful scientific application; not incremental "add component Y, gain 5%". Three directions were rejected this week.

| Rejected | Why |
| --- | --- |
| Forecasting selector (does interaction reliability predict when channel mixing helps?) | Pre-empted by Abdelmalak et al. 2025 (arXiv 2502.09683), Choi et al. 2026 (2609.32187), Won 2026 (2610.08118) and Pinet et al. 2026 (2606.02670). |
| Information lattice (exact group-surrogate decomposition of label information) | Sound, and its code works (`research/information_lattice/`), but the user judged it a narrow diagnostic. |
| In-context interaction inference in time-series foundation models | Too bland for the user. |

The user now asks for: **interpretability**; **pyspi as an inductive bias**, ideally yielding **sample efficiency**; and **attention**, because individual heads can be inspected for mechanistic claims.

Assets:

- **pyspi 3.0**: the user maintains a correctness-fixed fork; computation `3.0.0.r9`, fork commit `c2d723f`. The bundled `benchmarked_p90` set has 283 pairwise statistics: linear/nonlinear, directed/undirected, lagged, spectral, information-theoretic. It runs fast enough for systems of hundreds of channels.
- **NCI Gadi** with abundant CPU/GPU. All compute goes there; only small tests run locally. See `HPC.md`. Scratch and gdata are near their **inode** quotas, so keep outputs consolidated.
- **The archived EEML abstract**: a linear probe over named statistics gating a message-passing network. Its interpretation claims were weakened, and its baselines were weak; see `docs/research/assessment.md`.

## The proposal in one paragraph

*C. elegans* is the only animal with a complete wiring diagram that also has a *causal* map of how signals actually propagate between its neurons. That map comes from optogenetic stimulation of 23,433 neuron pairs across 113 worms ([Randi et al., Nature 2023](https://www.nature.com/articles/s41586-023-06683-4), *verified*). The atlas **diverges from the wiring**: many functional links are between unwired neurons and depend on neuropeptides, i.e. "wireless" extrasynaptic signalling. That map took a heroic perturbation campaign. **Can spontaneous activity alone, with no stimulation, reveal who signals to whom, and by which channel (wired versus wireless)?** We propose a transformer whose cross-neuron attention logits are biased, per head, by named pairwise statistics from pyspi. Each head becomes an inspectable, named routing rule ("this head routes along lagged transfer-entropy-like dependence"). We test three things against four ground truths (chemical synapses, gap junctions, the neuropeptide connectome, the causal atlas):

1. whether the statistical prior buys **sample efficiency** across animals;
2. whether individual heads **recover distinct signalling layers**, validated by ablation and *in silico* stimulation compared against the real stimulation atlas;
3. whether **wired and wireless signalling leave different statistical signatures** in spontaneous activity.

## Why this is important and not obvious

- **Who cares:**
  - Connectomes are now being mapped for flies, mice and, in part, humans, but wiring does not determine signalling. Neuromodulators and peptides act off-wire, and the worm shows this is not a footnote.
  - A method that reads *causal signalling layers* from cheap passive recordings would let labs map signalling in mutants, other states, other species and other individuals without optogenetic campaigns.
  - For ML, it is a rare setting where attention-head interpretations can be **checked against causal ground truth** rather than asserted.
- **Why not obvious:**
  - Pairwise activity correlation is known to predict synapse counts poorly.
  - [Creamer, Leifer & Pillow (bioRxiv 2024)](https://www.biorxiv.org/content/10.1101/2024.09.22.614271v1) (*verified*) report that a wiring-constrained linear model fitted to *perturbation* recordings reproduces causal interactions at about 82–92% of data reproducibility; the figure differs between versions, so verify it. Adding off-wiring connections did not help. This sits in tension with Randi's extrasynaptic findings.
  - Whether *spontaneous* activity carries recoverable information about the causal atlas, and especially about its off-wiring part, is open.
  - Whether richer statistics (lagged, directed, nonlinear, spectral) recover more than correlation is unknown.
  - Whether the statistics separate signalling modes is unknown.
- **Why attention, and the caveat.** Heads give per-pair routing maps, per-head named statistic weights, and handles for ablation and activation patching. Attention weights are *not* explanations by default ([Jain & Wallace 2019](https://aclanthology.org/N19-1357/); [Wiegreffe & Pinter 2019](https://aclanthology.org/D19-1002/)). Attention-derived gene regulatory networks in single-cell foundation models reportedly failed stress tests (locate and cite this). The worm is valuable precisely because every head-level claim can be scored against connectome layers and the causal atlas, and checked by intervention.

## Method sketch (to be sharpened, not fixed)

**Inputs.** Per animal: calcium traces of identified neurons (NeuroPAL-type labels), N of roughly 100–200 and a few thousand frames, plus behaviour where available. Neuron identity is conserved across animals, so pairs are indexed by identity and statistics can be pooled.

**Statistic tensor.** For each ordered pair (i, j), s_ij is a vector of K ≈ 283 standardised pyspi statistics. Compare three estimation schemes, keeping only past data relative to every forecast target:

1. per animal, from a disjoint calibration segment;
2. pooled per identified pair over *training* animals (a population prior);
3. both together.

**Statistic-biased attention (SBA).** Per-neuron temporal encoder (patching or a small CNN), then L cross-neuron attention layers with logits

    a^h_ij = (q_i · k_j)/√d  +  w_hᵀ s_ij  (+ b_h)

w_h is a per-head weight vector over named statistics, optionally sparse or grouped by pyspi statistic families (Cliff et al. 2023). Arms, matched in parameters and tuning budget:

- **SBA-additive**: content plus statistic bias, Graphormer-style ([Ying et al. 2021](https://arxiv.org/abs/2106.05234)).
- **SBA-routed**: statistic bias only, no q·k. This is the most interpretable arm: each head is a named statistic graph.
- **Vanilla attention.**
- **Connectome-biased attention**: the anatomical prior instead of the statistical one.
- **Shuffled-statistic and time-reversed-statistic controls.**
- **Single-statistic bias** (Pearson only; best single statistic).
- Non-attention baselines:
  - per-neuron autoregression; vector autoregression or ridge regression;
  - the Creamer linear dynamical system, using its code, if feasible from spontaneous data;
  - a GRU or transformer without cross-neuron mixing.

Note the [Brain Network Transformer](https://proceedings.neurips.cc/paper_files/paper/2022/hash/a408234a9b80604a9cf6ca518e474550-Abstract-Conference.html) argument that graph-transformer edge encodings fit complete correlation graphs poorly. Address it; do not ignore it.

**Relations among the statistics themselves (SPI–SPI versus attention over statistics).** The K×K Pearson matrix of the statistics across channel pairs (the user's SPI–SPI representation) discards pair identity, so it cannot answer the edge-level question of who signals to whom. It is used here in three supporting roles:

1. **Grouping or whitening the statistic basis**, so each w_h is identifiable. Collinear statistics make per-statistic weights non-unique: in the archived study no single statistic reached 0.8 stability-selection frequency.
2. **A direct S2 test**: SPI–SPI computed separately over chemical, gap-junction, peptidergic-only and unconnected pairs. Does the joint organisation of statistics differ by signalling mode? This is the natural bridge to the PhD line.
3. **A baseline descriptor** for recording-level tasks, such as behavioural state.

Attention *between statistic tokens* is a learned, supervised generalisation of SPI–SPI: with linear embeddings, q_a·k_b = x_aᵀWx_b, and W = I recovers Pearson. Consider it only as an optional arm for recording-level tasks, where many recordings and pair-localised or nonlinear agreement could justify it. A cheaper middle ground is a pair-weighted SPI–SPI, x_aᵀ diag(v) x_b with learned or ground-truth-defined pair weights.

**Tasks.** (a) Forecast held-out time and held-out *animals*. (b) Predict a masked neuron from the others (Creamer's held-out-neuron test). (c) Optionally, decode behavioural state.

**Interpretability toolkit:**

- **Head–layer alignment**: area under the precision–recall curve of each head's attention map against chemical synapses, gap junctions, the neuropeptide connectome and the atlas, versus chance, *the best single raw pyspi statistic* and vanilla heads.
- **Per-head statistic signatures**: w_h grouped by statistic family.
- **Head ablation and activation patching**: which predictions depend on which heads.
- **In silico stimulation**: inject a pulse into neuron j's input and read the model's response in neuron i. Compare sign, amplitude and latency with Randi's measured responses.
- **Stability** across seeds and animal subsets.

## Hypotheses and kill criteria (pre-register before scaling)

| ID | Hypothesis | Decisive test | Kill / narrow if |
| --- | --- | --- | --- |
| S1 | Spontaneous-activity statistics recover the causal atlas beyond correlation, including off-wiring edges | Gate A, below, with no deep model | No pyspi statistic beats Pearson/lagged correlation by a meaningful, animal-replicated margin |
| S2 | Wired (chemical, gap junction) and wireless (peptidergic) signalling have distinct dependence signatures (e.g. gap junctions zero-lag and symmetric; chemical lagged and directed; peptidergic slower or nonlinear) | Classify edge type from statistic profiles, controlling for distance, neuron class and activity level | Profiles indistinguishable after controls |
| M1 | The statistic bias buys sample efficiency | Learning curves over the number of training animals, all arms equally tuned | No gain at any data scale over vanilla and connectome-biased attention |
| M2 | SBA heads recover signalling layers better than vanilla heads and better than the best single statistic | Head–layer alignment plus ablation | A raw statistic matches the best head. The paper then becomes a pyspi atlas study — narrower, but still valid |
| M3 | In silico stimulation of the trained model predicts the measured atlas responses | Correlation with Randi's response sign, amplitude and latency, against connectome-model predictions | At chance, or no better than the correlation graph |

**Gate A is the cheapest decisive experiment and comes first.** Compute pyspi p90 on spontaneous recordings of identified neurons, pool per pair across animals, and score each statistic's ability to recover:

- atlas edges, including direction and sign;
- wired versus unwired functional edges;
- connectome layers.

Report animal-bootstrap uncertainty. If Gate A fails, the attention model has nothing to bias toward, so stop or change system before building SBA. **Gate 0**, run in parallel and cheaply, is a synthetic network with two signalling modes of known type: fast wired synapses plus slow diffuse modulation, observed through calcium-like filtering and noise at worm-like N, T and frame rates. Use it to validate SBA's sample-efficiency and head-recovery claims where ground truth is exact.

## Data leads (verify access, licence, identifiers, sampling rate, length, behaviour)

| Resource | What | Status |
| --- | --- | --- |
| [Simeon, Kashyap, Kording & Boyden, arXiv 2411.12091](https://arxiv.org/abs/2411.12091) | Homogenised activity from 12 experiments (~900 worms, ~250 labelled neurons, common ordering and resampling) plus 9 connectome annotations and 1 signal-propagation study. Hugging Face `qsimeon/celegans_neural_data` and `qsimeon/celegans_connectome_data`; CC BY-NC-SA 4.0 | *verified* (abstract page) |
| Randi et al. 2023 atlas | Response kernels, sign and latency for 23,433 pairs; Leifer lab releases; check the `wormneuroatlas` package | paper *verified*; data format to check |
| Creamer et al. 2024 | Code at GitHub `Nondairy-Creamer/Creamer_LDS_2024` | *verified* via search; check it runs |
| Connectomes | White 1986; Cook et al. 2019; Witvliet et al. 2021 (chemical synapses and gap junctions) | verify versions and licences |
| Neuropeptide connectome | Ripoll-Sánchez et al., Neuron 2023 | verify |
| Spontaneous whole-brain recordings with identified neurons | Atanas et al., Cell 2023 (Flavell lab, WormWideWeb, with behaviour); Kato et al. 2015; Yemini et al. 2021; Uzel et al. 2022; Dag et al. 2023 | verify which are in the homogenised set and which have behaviour |
| Related analyses | [Dvali, Seguin, Betzel & Leifer, PRX Life 2025](https://journals.aps.org/prxlife/pdf/10.1103/6wgv-b9m6); [Toyoshima et al., PLOS CB 2024](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10942262/); decomposed linear dynamical systems for *C. elegans* (2025, PMC12350842) | *verified* exist; read for overlap |

**Possible second systems**, for ML generality. Evaluate; don't commit:

- **Mouse visual cortex, MICrONS**: co-registered calcium imaging and electron-microscopy synapses; cellular ground truth, but sparse proofread axons and weak activity–synapse links.
- **Collective animal motion**: simulated rules plus real fish or pigeon trajectories. [Heras et al., PLOS CB 2019](https://journals.plos.org/ploscompbiol/article?id=10.1371/journal.pcbi.1007354) is a precedent for attention revealing interaction rules.
- **Fly whole-brain imaging with the FlyWire connectome**: identity mapping is coarser.

## Feasibility questions to answer with numbers

- Per dataset: frame rate, recording length, identified-neuron count, number of animals. Which of the 283 statistics are estimable at those lengths? Spectral, convergent-cross-mapping and k-nearest-neighbour estimators may be unreliable at roughly 1–2 Hz and ~1,000–4,000 frames. Measure estimator stability on split halves.
- pyspi p90 runtime and memory at N ≈ 150 and T ≈ 2,000 per animal, and a Gadi job plan with file-count-conscious outputs.
- How many animals overlap each atlas pair? What fraction of atlas edges are wired, gap-junction or peptidergic-only?
- Leakage hazards: statistics from overlapping windows, neuron-identity confusion, and atlas and spontaneous data drawn from the same animals or strains.

## Overlap and positioning to settle

- **The user's PhD line** (sibling repository `mts-spi-study-cluster`) characterises *dependence character* via SPI–SPI representations. S2 touches the same theme. Propose a clean boundary or a synergy, e.g. the worm as a ground-truth case study of character. Do not silently duplicate it.
- The paper must differ clearly from:
  - Creamer 2024: perturbation-trained, wiring-constrained, linear;
  - Randi 2023: the atlas itself;
  - Dvali 2025: network architecture comparison;
  - Brain Network Transformer and brain-graph transformers: phenotype prediction, not causal validation;
  - Heras 2019: attention for collective-motion rules.
- If someone has already inferred the signalling atlas from spontaneous activity, or biased attention with pairwise statistics for neural populations, say so plainly.

## What to return

A markdown report (≤ ~4 pages) in this repository at `docs/research/wired-wireless-exploration.md`, containing:

1. **Verdict**: go, narrow, or no-go, with the decisive reasons.
2. **Literature**: closest prior work with verified links, and what is genuinely open. Label every claim verified, inferred or unverified.
3. **Data**: a table of usable datasets with N, T, frame rate, animals, labels, behaviour, licence and access steps.
4. **Refined hypotheses and kill criteria**, and a pre-registrable Gate 0 and Gate A protocol.
5. **Compute plan for Gadi**: core-hours, GPU-hours, storage and inode budget.
6. **The strongest objections** ("well, obviously"; attention is not explanation; Creamer already explains the atlas; calcium dynamics too slow for lag statistics), each with a response or an admission.
7. **Recommended title-level framing** and target workshops.

Do not launch Gadi jobs or bulk downloads without the user's go-ahead. Small inspection downloads are fine. Keep the repository's conventions: one paragraph per line, concise, no duplicated status documents.

## Prompt to give the agent

> Read `docs/research/wired-wireless-brief.md` in `eeml-2026-application` (branch `codex/research-foundation`), plus `CLAUDE.md` and `HPC.md`. Explore and stress-test the proposal as instructed. Verify literature and data access, quantify feasibility, sharpen hypotheses and the Gate 0 and Gate A protocols, and give an honest go/narrow/no-go verdict. Write the report to `docs/research/wired-wireless-exploration.md` and commit it to the same branch. Do not submit Gadi jobs or bulk-download data without asking the user.
