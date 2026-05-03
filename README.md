# Sploink Research-Engineer Assessment

This repository contains my full submission for the Sploink Research Engineer assessment: graph detection of agent-session health and a proposal for real-time intervention.

## Repository Layout

```
.
├── README.md
├── systems_design.md
├── simulate.py
├── solution.py
├── research_proposal.md
├── data/
│   ├── sessions.jsonl
│   └── labels.jsonl          ← ground-truth labels (classifier never reads this)
├── predictions.csv
├── features.csv              ← per-session feature dump (debugging aid)
```


## Reproducing the Result

```bash
# Generate the dataset (deterministic given --seed).
python simulate.py --out data/sessions.jsonl --seed 42

# Classify every session and dump predictions + per-class metrics.
python solution.py \
    --events data/sessions.jsonl \
    --truth  data/labels.jsonl \
    --out    predictions.csv
```

## Headline Result

**~ 90% accuracy across 400 sessions.**

```

arpbansal@DESKTOP-NP5V1SB:~/code/sploink$ python3 simulate.py 
wrote 22292 events across 400 sessions to data/sessions.jsonl
wrote ground-truth labels to data/labels.jsonl
  session kinds: {'phase_transition': 103, 'pure': 240, 'locally_mixed': 57}
  hard labels (second-half-dominant): {'progressing': 119, 'looping': 89, 'drifting': 91, 'failing': 101}
arpbansal@DESKTOP-NP5V1SB:~/code/sploink$ python3 solution.py 
loaded 21465 unique events across 400 sessions (dropped 827 exact duplicates)
wrote 400 predictions to predictions.csv

Accuracy: 369/400 = 0.922

Confusion matrix (rows = actual, cols = predicted):
actual \ pred  | progressing |     looping |    drifting |     failing
----------------------------------------------------------------------
progressing    |         114 |           0 |           1 |           4
looping        |           3 |          84 |           0 |           2
drifting       |           8 |           0 |          72 |          11
failing        |           2 |           0 |           0 |          99

Per-class precision / recall / F1:
  progressing   P=0.898  R=0.958  F1=0.927
  looping       P=1.000  R=0.944  F1=0.971
  drifting      P=0.986  R=0.791  F1=0.878
  failing       P=0.853  R=0.980  F1=0.912

Accuracy by session kind:
  locally_mixed      50/57 = 0.877
  phase_transition   96/103 = 0.932
  pure               223/240 = 0.929
arpbansal@DESKTOP-NP5V1SB:~/code/sploink$ python3 test_solution.py 
Running edge-case demonstration tests:
------------------------------------------------------------
  PASS  baseline (no-noise progressing session classified as progressing)
  PASS  duplicate events: deduped, label unchanged
  PASS  step collisions: siblings preserved, not deduped
  PASS  late events: timestamp inversion, hybrid sort recovers correct order
  PASS  missing metadata: latency / target / status missing on subset, no crash
  PASS  interleaved sessions: globally shuffled file, each grouped and labeled
  PASS  burst: 100+ events sharing a near-identical timestamp
  PASS  silent stall: long gap mid-session, has_silent_stall fires
  PASS  combined: burst + step-collisions + duplicates in one session
------------------------------------------------------------
Total: 9    Failures: 0
arpbansal@DESKTOP-NP5V1SB:~/code/sploink$ 
```

The dataset includes every adversarial condition the spec requires:
- 843 exact-duplicate events (handled via content hashing in `solution.py`)
- 1,481 step-collision keys (preserved as siblings, not deduped)
- 1,529 step-vs-timestamp inversions (handled via hybrid sort key)
- 1,126 events with missing metadata fields (`.get()` defaults)
- 99.7% session-to-session adjacency in the file (full interleaving)

## Design Choices Worth Calling Out

- **Graph-native, not CRUD-native.** The classifier's discriminating features are over the (action, target) n-gram graph: cycle pressure (3-gram repetition rate), unique-node ratio, output entropy, and Jensen-Shannon divergence between the first and second halves of the target distribution. The same graph schema appears in `systems_design.md` and `research_proposal.md` so the same mental model carries end-to-end.
- **Transparent classifier, not a black-box ML model.** Threshold-based rules with per-class evidence scores. The choice is deliberate and defended in `solution.py`'s docstring — the signal is structural and trees overfit the simulator's quirks. In production the same features would feed a calibrated GBM; the features are the long-lived asset.
- **Hybrid ordering key over timestamps alone.** `(step, timestamp, content_hash)` — neither timestamp nor step number alone is reliable, and the hybrid is what survives the bursts and late arrivals.
- **Content-hash dedup that excludes timestamp.** Two emissions of the same logical event collapse correctly even when the second arrives with a different ingest timestamp.

---

# Part D — Ownership Reflection

## What's complete vs partial vs skipped

**Fully completed:**
- Part A: `systems_design.md`, 3,300 words, ASCII architecture diagram, all required sections.
- Part B: `simulate.py` generates 100 sessions per class with the required noise types injected; `solution.py` classifies all 400 sessions at 92.2% accuracy on the realistic-overlap dataset, with a confusion matrix and per-class P/R/F1 reported. `predictions.csv` produced.
- Part C: `research_proposal.md` presents a graph-conditioned POMDP with simulator-trained PPO, an LLM semantic diagnoser plus action parameterizer, DR off-policy evaluation, staged rollout, failure modes with safeguards, and a dedicated graph-vs-sequence comparison.

**Partial / skipped:**
- Task B - Partial, in sense that, I put less time in it, so didn't went for exploring ML, sequence model or other approach.
- `optional_ui/` - skipped, I don't do UI/frontend.
- `training.log` - not produced. The classifier is rule-based, not ML-trained. The `features.csv` dump serves the same auditing purpose for this submission. I didn't spend time fabricating a training artifact for a non-trained model. In production, the natural next step would be a calibrated GBM or a learned graph-conditioned encoder.

## Time spent

| Section | Time |
|---|---|
| Part A systems design | ~ 5.5 hrs |
| Part B simulator + classifier (incl. tuning the progressing/drifting boundary) | ~2 hrs |
| Part C research proposal | ~3.5 hrs |
| Part D reflection + repo polish | ~1.5 hrs |
| **Total** | **~12.5 h** |

The single biggest time sink was diagnosing why progressing was being misclassified as drifting on the first classifier run. Looking at per-class feature percentiles immediately exposed that `output_entropy` was the cleanest separator (progressing p50 = 3.17, drifting p50 = 5.91), and re-tuning around that took the classifier from an unusable first pass to a strong result on the clean synthetic split. I then made the simulator harder with realistic overlap, which dropped the final reported score to 92.2% but produced a more honest evaluation.

## Top two weakest assumptions

1. **The simulator is still simpler than real agent behavior, even after I made it harder.** The final 92.2% result is on a more real synthetic dataset than the earlier 98.8% clean split, but it is still synthetic. Real traces will have messier semantics, weaker class boundaries, and more ambiguity between drifting and failing than this generator produces. I would still expect a rule-based classifier like this to land closer to the 75–85% range on real production data, which is why the production path should move toward a calibrated GBM or a learned graph-conditioned encoder.
2. **The intervention proposal assumes enough historical intervention data exists to calibrate the simulator and evaluate the policy.** The event log and reverse control plane in `systems_design.md` make that data path plausible going forward, but the hard part is coverage: rare, high-value actions like `rewind`, `terminate`, and `request_human_approval` will be sparse at bootstrap time. The proposal does not literally require gold labels for "the right action," but it does rely on enough `(session, intervention, outcome)` triples and enough human-reviewed diagnosis data to fit the simulator, train the semantic layer, and make DR evaluation credible.

## Usage of AI

I use AI in extreme boundations with one core principle: AI follows a plan, but the plan is mine, not AI's. I don't "Vibe Code", I do AI assisted coding. 
I review the code against a set of expectations, and making sure that AI MUST NOT DEVIATE FROM THE PLAN. If it does, I stop and correct the course.

If making system design, make sure that we go through alternatives, we know what are the options, what aligns with system etc.

## What v2 looks like in 30 days

- **Implement the streaming feature aggregator** (HyperLogLog for cycle counts, Misra-Gries for target entropy) in the Go/NATS scoring path described in `systems_design.md` and benchmark the actual p99 latency of that pipeline. Confirms whether the §6 latency claims survive contact with reality.
- **Stand up the simulator-calibration and shadow-mode intervention path from Part C** over the existing reverse control plane. Even before executing actions, shadow-mode decisions plus operator review would start producing the intervention data needed for DR evaluation and later policy training.

- **Honest stress test of the simulator's noise model.** Compare distributions of duplicates, gaps, and step collisions in `data/sessions.jsonl` against a small sample of real production traces (from any open agent log we can find) and adjust the noise knobs accordingly.

The classifier and the proposal are both written to assume v1 is the *interpretable, demonstrable, defensible* version, and v2 is the *higher-capacity, better-calibrated* version that comes once the v1 is in production and the data shape is understood. That ordering is deliberate and I'd defend it in the walkthrough.

## Something i thought on last minute

- **I personally think we can try a webhook based approach rather than websockets, due to optimization needed for handling million websockets. websockets are resource intensive in nature. 
If webhook based system can work for cli to backend call, I think will be better to handle and save dev time too.**

- Best intervention initially will have prompt injection, prompts designed per task (i mean special prompts for code, sales etc.)

### Why accuracy dropped from 98.8% to 92.2%.

Initially without giving much thought i went for hard labels, no overlap, which was not realisitic.

Three reasons, in order of contribution:
1. Realistic-overlap sessions are genuinely harder to label, and this is most of the drop, roughly 5 of the 6.6 points. The original dataset was mostly pure sessions with crisp graph signatures, while the harder dataset mixes classes inside a session, so signals like cycle pressure and output entropy partially cancel. The second-half-dominant rule still helps, which is why phase-transition sessions stay strong at 93.2%, but locally mixed sessions are much harder and fall to 87.7%.

2. The simulator's hard label is itself debatable on edge cases, probably about 1 point of the drop. The second-half-dominant rule forces one label even when a session meaningfully contains two behaviors, so a session that loops early and recovers late is marked as progressing even if the looping signal is still obvious. Some of the confusion-matrix errors are true misses, but some are really disagreements with the labeling rule rather than the classifier missing the pattern.

3. Drifting and failing overlap on late-session features, probably another 0.5 point. Some drifting sessions accumulate clustered failures late enough that their second-half features look failing, which explains many of the 11 drifting-to-failing errors. I could tune thresholds harder to reduce that, but doing so would mostly optimize for this simulator; 92.2% is a more honest estimate of what would hold up on messier real traces than the earlier 98.8%.