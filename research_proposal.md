# Sploink Intervention Engine — A Graph-Conditioned POMDP with Simulator-Trained Policy and LLM-Grounded Semantic Diagnosis

**Status:** Proposal
**Scope:** Real-time selection of the next-best intervention when an autonomous-agent session degrades.

---

## 1. What We're Solving

The classifier in `solution.py` answers *is this session healthy?* The system we propose answers *given that it isn't, what should we do about it?*

The action set is given:

1. `rewind(checkpoint)` — restore graph state to an earlier step.
2. `reset_context` — clear the agent's working memory.
3. `spawn_helper(specialist)` — instantiate a side agent.
4. `reroute_tools(toolset)` — change the available tool registry.
5. `decompose(subtasks)` — break the current task into smaller pieces.
6. `request_human_approval(question)` — gate further execution behind review.
7. `terminate` — stop the run.
8. `switch_planning_mode(mode)` — change the agent's planning strategy.
9. `noop` — do nothing, observe further. We add this; without it, every classifier blip becomes an action.

The decision is sequential, the world is partially observed (we see events, never the agent's internal beliefs), and the right action depends on the *structure* of the session graph, not just summary statistics.

What makes this proposal different from a generic RL-for-control writeup. I argue three claims, each defended below:

1. **The intervention engine is two components, not one.** A graph-structural classifier handles "is this session degrading?" An LLM-based semantic diagnoser handles "is the agent doing the *right thing*?" Both feed the policy.
2. **The policy is trained against a simulator, not against live customer agents.** Online RL on customer traffic is a commercial risk we should not take when a calibrated simulator is achievable.
3. **Graph-native reasoning is non-optional.** Linear-sequence formulations of session state lose exactly the information the policy needs.

---

## 2. Formal Framing

### 2.1 Why POMDP, Not MDP or Bandit

We model the intervention engine as a **discounted partially observable Markov decision process** (POMDP):

$$
\mathcal{M} = \big( \mathcal{S},\, \mathcal{A},\, \mathcal{O},\, \mathcal{T},\, \Omega,\, \mathcal{R},\, \gamma \big)
$$

The decision problem is partially observed: even when we have access to the agent's chain-of-thought tokens and planner state, we do not know how the next reasoning step will resolve, and we do not know the agent's true beliefs about the task. POMDP is the formalism that admits this honestly.

We considered three alternatives and rejected each:

- **MDP.** Assumes full observability. Wrong: even the agent itself is uncertain about its next decision.
- **Contextual bandit.** Treats each decision as independent. Throws away the sequential structure. A bandit cannot reason about "I rewound to step 20; was that the right choice?" — that decision can only be evaluated against the rest of the trajectory.
- **Pure graph decision process.** Less standardized in the literature [Bertsekas, 2019]. We borrow the *graph-structured state and observation* from this framing while keeping the formal apparatus of POMDP.

### 2.2 Graph-Conditioned POMDP

We're explicit that this is a POMDP whose state, observation, and action representations are all graph-structured. Recent work has formalized this hybrid carefully [Battaglia et al., 2018; Schlichtkrull et al., 2018], and we adopt their conventions.

**State** $s_t = (G_t, b_t, \tau_t, c_t)$ where:

- $G_t = (V_t, E_t)$ is the typed execution subgraph for the session up to step $t$. Vertices are typed by `action` (read_file, retry, etc.); edges are typed in $\{\textsc{next}, \textsc{reads}, \textsc{branched\_from}, \textsc{retried}, \textsc{validated\_by}\}$.
- $b_t \in \mathbb{R}^k$ is the agent's latent belief — its planner state, its working hypothesis. We never observe this directly; it is the *partially observable* part.
- $\tau_t$ is the task descriptor — the user's original goal, encoded once at session start.
- $c_t$ is the chain-of-thought / planner trace observed up to time $t$. Available but not fully revealing of $b_t$.

**Observation** $o_t = (G_t, c_t, \text{score}_t, \text{semantic\_diag}_t)$ where:

- $\text{score}_t \in \Delta^4$ is the per-class evidence distribution from the structural classifier (`solution.py`).
- $\text{semantic\_diag}_t$ is the LLM-based semantic diagnoser's structured output (§4.2).

The policy operates on belief states $\beta_t = p(s_t \mid o_{1:t}, a_{1:t-1})$, approximated via a recurrent encoder over the observation history.

**Action space** $\mathcal{A}$ is hierarchical with a discrete top level over the 9 action types and a continuous-or-discrete sub-action conditioned on type:

$$
a_t = (a^{\text{type}}_t,\; a^{\text{param}}_t)
$$

For example, `rewind` carries $a^{\text{param}} \in [0, t]$ (which step to rewind to); `reroute_tools` carries a categorical over toolsets; `terminate` has no parameter. Hierarchical action spaces with this structure are well-studied [Bacon et al., 2017].

### 2.3 The Two-Component Architecture

A key contribution of this proposal: we do not model the intervention engine as a monolithic policy. The engine is two distinct components feeding a meta-controller.

```
                ┌────────────────────┐
                │  Event stream      │
                └──────────┬─────────┘
                           │
              ┌────────────┴────────────┐
              ▼                         ▼
   ┌─────────────────────┐   ┌──────────────────────────┐
   │  Structural         │   │  Semantic diagnoser      │
   │  classifier         │   │  (open-source LLM,       │
   │  (graph features    │   │   fine-tuned on session  │
   │   + rules)          │   │   diagnosis tasks)       │
   │                     │   │                          │
   │  → per-class scores │   │  → structured diagnosis  │
   │    + evidence       │   │    (missing reqs, etc.)  │
   └──────────┬──────────┘   └────────────┬─────────────┘
              │                           │
              └───────────┬───────────────┘
                          ▼
              ┌─────────────────────────┐
              │  RL meta-controller     │
              │  (graph-conditioned     │
              │   POMDP policy)         │
              │  → action_type          │
              └────────────┬────────────┘
                           ▼
              ┌─────────────────────────┐
              │  LLM action             │
              │  parameterizer          │
              │  → full action          │
              └─────────────────────────┘
```

The structural classifier detects structural pathologies (looping, drifting, failing, stalling) cheaply, on every event. The LLM-based semantic diagnoser answers questions the classifier cannot — *did the agent miss writing A/B tests? did it misunderstand the requirement to use Postgres rather than SQLite?* These are semantic observations that require reading natural language and code, not graph features.

The RL meta-controller chooses *whether* to intervene and *which* high-level action type. The LLM parameterizer then chooses *how* to instantiate the chosen action — when "spawn_helper" is selected, the LLM picks the helper's specialty and prompt; when "rewind" is selected, the LLM (informed by the graph and diagnosis) suggests the right step.

This separation puts each component where it adds most value, makes the RL problem tractable (small discrete top-level action space), and makes the system auditable: at every decision point, we can show *why* the structural classifier flagged the session, *what* the semantic diagnoser identified, *which* action type the policy chose, and *how* the LLM parameterized it.

### 2.4 Transition Model

$\mathcal{T}(s_{t+1} \mid s_t, a_t)$ is **not analytically known** — it depends on how the underlying agent reacts to the intervention, which depends on the agent's prompt, model weights, available tools, and the task. We do not assume access to a closed-form transition kernel. Instead we *learn* an approximate transition model in simulation (§4) and we *robustify* the policy against simulator-to-reality gap via the methods in §6.

### 2.5 Reward

This is the load-bearing modeling decision. We propose a **sparse terminal reward with potential-based shaping** [Ng, Harada, & Russell, 1999]:

$$
r_t = r^{\text{outcome}}_t + r^{\text{shape}}_t
$$

**Outcome reward.** Sparse and terminal:

$$
r^{\text{outcome}}_T = \begin{cases}
+1 & \text{if the session ultimately succeeded against its task descriptor} \\
-1 & \text{if the session ultimately failed} \\
\,\,\,\,0 & \text{if the session was terminated by intervention}
\end{cases}
$$

The "neither credit nor blame" branch for terminations matters. Without it, the policy learns to never `terminate`, because terminating loses the chance at $+1$. With it, terminating is better than letting a doomed session run all the way to failure, but still strictly worse than success — the right semantics for a controlled loss-cutting action.

We deliberately do **not** include cost-aware terms for expensive interventions. Sploink's business cares about successful outcomes; an intervention that costs more compute but saves a customer's run is a good intervention.

**Shaping reward.** Potential-based, with the standard guarantee that the optimal policy is unchanged regardless of the choice of $\Phi$:

$$
r^{\text{shape}}_t = \gamma \,\Phi(s_{t+1}) - \Phi(s_t)
$$

The potential function $\Phi(s)$ is a learned value-function estimate of "how likely is this state to lead to a successful outcome," trained separately on historical session data using supervised regression to terminal outcomes. The shaping term is a *telescoping sum* over any complete trajectory — its total contribution depends only on the start and end states, not on actions taken — which is the property that prevents reward hacking.

This formulation gives us per-step learning signal and faster convergence without the failure mode of dense reward functions derived from learned components. A naive dense reward like "+0.1 every time the classifier score improves" would let the policy hack the classifier (e.g., `reset_context` makes the post-reset session look superficially healthy because all the bad signal is gone). Potential-based shaping is the fix.

We considered and rejected: *pure sparse terminal* (clean; learns too slowly at 100+ step horizons), *dense classifier-based reward* (fast learning, hackable), *cost-aware reward* (misaligned with business priorities).

### 2.6 Discount

$\gamma \in [0.95, 0.99]$. Sessions are long; a $\gamma$ much below 0.95 makes the policy myopic, ignoring the long-horizon consequences of `rewind` decisions.

---

## 3. State Representation

### 3.1 Hybrid Graph + Recent-Window Encoding

We use a hybrid encoder that combines a graph encoding for structural awareness with a recent-window encoding for temporal recency.

**Graph encoder.** A relational graph convolutional network [Schlichtkrull et al., 2018] with action-typed message passing:

$$
h_v^{(\ell+1)} = \sigma\!\left( W_{\text{self}}^{(\ell)} h_v^{(\ell)} + \sum_{\text{type}\, e} \frac{1}{|\mathcal{N}_e(v)|} \sum_{u \in \mathcal{N}_e(v)} W_e^{(\ell)} h_u^{(\ell)} \right)
$$

Per-edge-type weight matrices $W_e$ are crucial — a `RETRIED` edge is not the same as a `BRANCHED_FROM` edge and the encoder must not collapse them. The session-level graph representation $z^{\text{graph}}_t$ is a permutation-invariant pooling over $\{h_v^{(L)}\}$.

**Recent-window encoder.** A causal transformer over the last $K = 50$ events, producing $z^{\text{recent}}_t$. Captures the temporal information the graph throws away — which retry chain is *currently* active, which file was *just* modified.

**Fused representation.**

$$
z_t = [z^{\text{graph}}_t \,;\, z^{\text{recent}}_t \,;\, \text{score}_t \,;\, \text{embed}(\text{semantic\_diag}_t)]
$$

Concatenation of: graph encoding, recent-window encoding, structural classifier output, semantic diagnoser output. This is the input to the recurrent belief tracker:

$$
\hat\beta_t = \text{GRU}(z_t,\; \hat\beta_{t-1})
$$

Recurrence over the observation history serves as a tractable approximation to the true belief update [Hausknecht & Stone, 2015]. It works well when the latent dimension of $b_t$ is bounded, which we believe it is — agents have bounded prompt size and bounded planner state.

### 3.2 Why Pure Sequence Encoding Is Insufficient

A natural alternative is to flatten the session into a token sequence and run a transformer end-to-end, treating the problem as sequence-to-action. This would be the obvious move in 2024. We argue it's the wrong choice for Sploink, for three concrete reasons.

1. **Cycles are first-class structure.** A transformer over the linearized sequence sees a loop as a repeated n-gram and has to *learn* that the repetition is structural. A GNN encoder sees the cycle as an actual cycle in the input graph — no learning required to detect it. The classifier in `solution.py` already exploits this; the policy should too.
2. **Causal edges aren't temporal.** A `RETRIED` edge connects a retry to its predecessor, which may have been many steps ago. In the linearized sequence, that's a long-range dependency the transformer must learn to attend across. In the graph, it's a single hop.
3. **Branches break the linearization.** When a session forks (sub-agents in the swarm exploring different paths), the linear sequence has to either pick one branch (losing information) or interleave them all (creating spurious adjacencies). The graph encoder handles branches structurally.

The general claim — *structure matters when the structure is the input* — is not novel. Graph-structured RL has been making this argument for half a decade [Wang et al., 2018]. Our contribution is bringing the same insight to autonomous-agent intervention specifically: the agent's execution graph is the right state representation for deciding what to do next, and reducing it to a sequence loses information that is exactly the information needed for the decision.

### 3.3 The Semantic Diagnoser's Output

The LLM-based diagnoser produces structured output — not free-form text. The schema includes a task-understanding score, a list of detected missing requirements, a list of wrong abstractions, an optional "obviously stuck" reason, an optional recommended action type, and a confidence score. Each field is fed into the policy's state representation as an embedding plus the scalar fields directly. The LLM is fine-tuned on (session, ground-truth-diagnosis) pairs derived from historical operator decisions and post-hoc human review.

The diagnoser does NOT directly choose an action. Its `recommended_action_type` field is a *suggestion*, given equal weight to other state features by the policy. The policy is free to override the recommendation when its learned value estimates disagree. This separation prevents the policy from collapsing to "do whatever the LLM says," which would inherit all of the LLM's failure modes (hallucination, prompt sensitivity, overconfidence on out-of-distribution sessions).

The diagnoser runs on session start, on every structural-degradation event from the classifier, and on a 5-minute timer for active sessions. It does not run on every event — too expensive.

---

## 4. Learning Algorithm — Simulator-Trained, Deployed Without Online Updates

### 4.1 Why Not Online RL on Customer Traffic

We deliberately reject online RL on live customer agents. Three reasons:

- **Exploration on production = customer churn.** A `terminate` chosen during exploration on a paying customer's run is a churned account.
- **Sample efficiency is a forcing function.** Even with off-policy methods, online RL needs millions of episodes to converge on hard problems. We don't have millions of degraded customer sessions to spare.
- **Iteration speed.** Online policies are bound to the rate at which production data accumulates. Simulator-trained policies iterate at the speed of GPU compute.

The standard counter-argument — "simulators always have a sim-to-real gap" — is true, but the gap is mitigatable, the alternative has worse expected cost, and the gap shrinks with each calibration cycle. We choose the path with bounded downside.

### 4.2 The Simulator

We extend `simulate.py` from Part B into a full agent-environment simulator. The Part B simulator generates session traces; the Part C simulator additionally models *agent response to interventions*:

- **Pre-intervention dynamics.** Same as Part B — sessions evolve through phases (progressing, looping, drifting, failing) with realistic graph signatures.
- **Intervention response model.** When the policy takes an action, the simulator transitions stochastically. The transition distribution is parameterized and fit to historical (session, intervention, outcome) triples from real operator decisions. `rewind` reverts the graph and resamples a continuation; `reset_context` clears recent context but keeps the agent's goal; `spawn_helper` injects helper-agent events.
- **Outcome model.** Each simulated session has a true terminal outcome drawn from a learned distribution over (initial-task-difficulty, observed-trajectory) → outcome.

**Domain randomization** [Tobin et al., 2017]. During training, simulator parameters (response distributions, latency distributions, noise rates, agent error patterns) are randomized within plausible ranges drawn from historical-data calibration. The policy must work across the full randomization range, which makes it robust to calibration error.

### 4.3 Algorithm: PPO with Recurrent Encoder

We propose **Proximal Policy Optimization** [Schulman et al., 2017] as the base learning algorithm. It is well-suited to:

- Policies with recurrent state (the GRU belief tracker).
- Hierarchical action spaces (with separate type and parameter heads).
- The relatively small effective action dimension (9 top-level types, parameterized within each).
- Stable training, which matters for a system whose reproducibility we'll have to defend.

Specifically: PPO with a separate value head trained against the potential-shaped reward, with a clipped surrogate objective:

$$
\mathcal{L}^{\text{CLIP}}(\theta) = \mathbb{E}_t\!\left[ \min\!\Big( \rho_t(\theta) A_t,\; \text{clip}(\rho_t(\theta), 1-\epsilon, 1+\epsilon) A_t \Big) \right]
$$

where $\rho_t(\theta) = \pi_\theta(a_t \mid o_t) / \pi_{\theta_{\text{old}}}(a_t \mid o_t)$ is the importance ratio and $A_t$ is the generalized advantage estimate.

We also propose, as an ablation rather than a primary method, **offline-first pre-training** via behavioral cloning on historical operator decisions, used to initialize the policy before the simulator-based training begins. This provides a useful prior — operators have seen these patterns before and made reasonable choices — and reduces simulator training cost.

### 4.4 Why PPO Over Alternatives

We considered **CQL / IQL** [Kumar et al., 2020; Kostrikov et al., 2022] — strong choices for *offline* RL where new data cannot be generated. Less natural here because we *can* generate new data via the simulator; the conservatism property is less useful in a controlled environment. We considered **DQN-family** methods — discrete actions match, but the action space is hierarchical and DQN handles hierarchy awkwardly compared to PPO's actor-critic structure. We considered **SAC / continuous-action methods** — the action space is fundamentally discrete at the top level, so fitting it into continuous-action methods adds complexity without benefit. PPO is the boring-correct choice. Its ubiquity is a feature: any engineer joining the project can debug it.

### 4.5 Hindsight Experience Replay

We additionally use **Hindsight Experience Replay** [Andrychowicz et al., 2017] to extract more learning signal from failed simulator trajectories. When the simulator generates a session that ends in failure, we relabel it as if its actual terminal state were the goal — the policy still gets to learn "actions that led to *that* outcome" even though *that* wasn't the desired outcome. This helps especially with the sparse-reward problem; failed trajectories are not wasted training data.

---

## 5. Why This Approach Fits Sploink

**Cost of exploration is decisive.** In a market where product correctness directly determines retention, online exploration on customer traffic is unaffordable. Every algorithm requiring online interaction during training is therefore disqualified, regardless of theoretical appeal. PPO + simulator + domain randomization is the only path that produces a deployable policy under this constraint.

**The data shape matches simulator construction.** Sploink's storage architecture (per `systems_design.md`) already produces `(state, action, reward, next_state)` tuples for any historical session that contained an operator intervention. Those tuples are the calibration data for the simulator's response model. We do not need to instrument anything new; the data exists.

**Two-component architecture matches the actual problem.** The intervention decision genuinely has two parts — *is the structure pathological?* and *is the agent doing the right thing?* Folding both into a single end-to-end model would either undertrain the structural part (semantic signal dominates loss) or undertrain the semantic part (structural signal is far more frequent). Separating them lets each be trained on its native data.

---

## 6. Offline Evaluation

The simulator-trained-policy approach has a fundamental risk: the simulator-trained-and-evaluated policy is not the same as the production-deployed policy. We address this with three layers of evaluation, each more expensive and more faithful than the last.

### 6.1 In-Simulator Evaluation

The first cut. We hold out a fixed evaluation suite of simulator scenarios — phase-transition sessions, locally-mixed sessions, and adversarially-generated scenarios designed to test specific failure modes. The policy must beat three baselines:

- *Always-noop.* Lower bound. Any policy doing worse than always-noop is broken.
- *Behavioral cloning.* Imitates historical operator decisions. The PPO policy must beat this on simulated outcomes.
- *Single-rule baselines.* Hand-written rules ("always rewind on retry chain >5", etc.) — the heuristics a smart engineer would write without RL. The policy must beat them too, otherwise we're not earning the complexity.

### 6.2 Doubly-Robust Off-Policy Evaluation on Real Data

Before any production deployment, we evaluate the simulator-trained policy against the historical data it was *not* trained on, using **doubly-robust off-policy evaluation** [Jiang & Li, 2016; Thomas & Brunskill, 2016]:

$$
\hat{V}_{\text{DR}}(\pi)
=
\frac{1}{N}\sum_{i=1}^{N}
\left[
\hat V\!\left(s_0^{(i)}\right)
+
\sum_{t=0}^{T_i-1}
\gamma^t\,
\rho_{0:t}^{(i)}
\Big(
r_t^{(i)}
+
\gamma \, \hat V\!\left(s_{t+1}^{(i)}\right)
-
\hat Q\!\left(s_t^{(i)}, a_t^{(i)}\right)
\Big)
\right]
$$

where $\rho_{0:t} = \prod_{k=0}^{t} \pi(a_k \mid s_k) / \mu(a_k \mid s_k)$ is the cumulative importance ratio of the new policy $\pi$ to the historical policy $\mu$, and $\hat Q, \hat V$ are model-based estimates that act as control variates. This is the standard step-wise DR construction: it is unbiased if either the importance weights or the model is correct, and it has lower variance than naive importance sampling in the regimes we care about.

The DR estimate is the gating metric. The PPO policy must produce a DR-estimated return higher than the historical operator's return, with confidence intervals that don't overlap, before deployment proceeds.

### 6.3 Counterfactual Human Review

For high-stakes interventions — particularly `terminate` — we never observe the counterfactual. The only way to evaluate is human review: present the session and the policy's chosen action to a domain expert, and ask whether the action would have been better than the action the operator actually took. This is expensive and slow. We propose it for a sample of sessions involving rare actions, not as a primary evaluation method.

---

## 7. Online Rollout

Even after a policy passes offline evaluation, we do not flip a switch. The deployment plan, in increasing order of exposure:

1. **Shadow mode (4 weeks).** Policy runs on every degraded session but outputs are logged, not executed. The system records what action *would* have been taken and whether the session subsequently improved. Builds out-of-distribution trust before any customer-visible action.
2. **Tier-3 tenants only (2 weeks).** Limited-blast-radius tenants opted into beta. Real interventions, kill-switch, tight per-tenant rate limits. Manual review of every `terminate` action.
3. **Tier-2 expansion (4 weeks).** Per-action rollout — `rewind` and `noop` first, `request_human_approval` second, `terminate` last and only with double sign-off.
4. **All tenants, conservative defaults.** Policy on by default with **pessimistic action selection**: when action-value uncertainty exceeds a threshold, fall back to `noop`. Handles sim-to-real gap on novel session shapes — when in doubt, do nothing.

Throughout the rollout, we collect production data and re-calibrate the simulator periodically (not the policy directly). Every 3-6 months, we take recent production data, update the simulator's response and outcome models, retrain the policy in simulation, deploy through the same staged rollout. **This is not online RL.** The policy is fixed between releases; only the offline pipeline updates.

If the DR-estimated return on production data drops below the BC baseline, affected tenants auto-roll-back to advisory-only mode.

---

## 8. Failure Modes and Safeguards

### 8.1 Sim-to-Real Gap

The simulator's response model is fit to historical data. When the policy is deployed on agent workloads or task domains the simulator hasn't seen, its predictions of intervention outcomes are unreliable.

**Safeguard.** Pessimistic action selection at deployment (§7). The policy maintains a Q-value uncertainty estimate (via ensemble or Monte Carlo dropout). When uncertainty exceeds a threshold, fall back to `noop`. Threshold set conservatively, tightened only as production data confirms calibration. Domain randomization during training (§4.2) is the complementary safeguard.

### 8.2 LLM Diagnoser Hallucination

The semantic diagnoser is an LLM, fine-tuned but still subject to hallucination — confidently asserting "the agent forgot to write A/B tests" when it didn't. If the policy heavily weights `recommended_action_type` from the diagnoser, hallucinations propagate.

**Safeguards.** The diagnoser's `recommended_action_type` is one feature among many; the policy is free to override it. The diagnoser's `confidence` field is an explicit input — the policy learns (in simulation) to trust it more when confidence is high. During simulator training, we corrupt diagnoser outputs a fraction of the time so the policy learns robustness to bad diagnoses. In production, sampled outputs are human-reviewed; persistent hallucination patterns trigger LLM re-fine-tuning.

### 8.3 Distribution Shift From Deployment

Once deployed, the intervention policy changes which sessions reach which states. Naive iterative recalibration could converge to a degenerate fixed point.

**Safeguard.** A **stratified calibration buffer** preserves a fraction of pre-intervention historical data indefinitely. New production data displaces old data within strata, never across them — same mechanism off-policy methods use to prevent catastrophic forgetting [Rolnick et al., 2019]. We also retain a **canary cohort** of opt-in tenants who accept reduced intervention frequency in exchange for serving as a control group; their data is the unbiased reference for distribution-shift detection.

---

## 9. Why Graph-Native Reasoning Matters Here, Concretely

Three concrete reasons specific to *this* problem.

**1. `rewind` selection requires graph reasoning.** The right rewind target is "the most recent step from which a different choice would have avoided the current failure mode." That's a question about the causal graph — finding the lowest common ancestor of failing nodes in the `RETRIED`/`READS` subgraph. A linear-sequence policy picks rewind targets by step distance, which is wrong; the right step might be 50 events ago in the sequence but one structural hop in the graph.

**2. `decompose` and `spawn_helper` require subgraph identification.** When you decompose a stuck task, you decompose it into parts that each correspond to a subgraph of the current execution. A graph-aware policy can identify which subgraphs are healthy (and should be preserved) versus which are degenerate (and should be re-attempted). A sequence policy has no notion of subgraphs at all.

**3. Cycle detection is exact in graphs, approximate in sequences.** A transformer over linearized events sees a loop as a repeated n-gram, with all the false positives (legitimate retries that look like repetition) and false negatives (loops with parameter variation) that come with that. The classifier in `solution.py` already exploits exact cycle detection for diagnosis; the intervention policy operates on the same representation for the same reason.

---

## Graph-Native Reasoning vs Linear-Sequence Assumptions: A Comparison

The dominant paradigm in 2024-2025 for modeling sequential agent behavior is to flatten everything into a token sequence and run a transformer. This is what every modern RL-on-text and LLM-policy paper does. We argue this paradigm is wrong for the intervention problem specifically. Here is the head-to-head comparison.

### What Each Representation Actually Encodes

A **linear-sequence representation** of a session is an ordered list of events: `[e_1, e_2, ..., e_t]`. Each event is a token (or a small token group). The model sees position in the sequence as the only relational information.

A **graph-native representation** of the same session is a typed graph `G = (V, E)` where vertices are events and edges are typed causal relationships (`NEXT`, `READS`, `RETRIED`, `BRANCHED_FROM`, `VALIDATED_BY`). The model sees explicit relationships between events that may be far apart in time.

The two representations carry different information. The graph contains everything the sequence contains (you can walk the `NEXT` edges to recover the sequence) plus the typed causal edges. The sequence does not contain the causal edges — they have to be *inferred* by the model from positional patterns.

### The Three Things the Graph Has That the Sequence Doesn't

**1. Cycles as first-class objects.**

Consider a session where the agent reads file X, calls the LLM with X's content, writes a modification, runs a test, the test fails, reads X again, calls the LLM again, etc. — a tight 4-event loop repeated 5 times.

- *Sequence view:* a transformer sees a 20-token sequence with a repeating 4-token motif. To detect the loop, it must learn an attention pattern that recognizes the repetition. This is learnable but requires data; it can also produce false positives (legitimate retries with intentional repetition look the same) and false negatives (a loop with parameter variation — same 4 actions but each time on a slightly different file — doesn't pattern-match).
- *Graph view:* the cycle is a literal cycle in the (action, target) graph. One graph operation — `find_strongly_connected_components` — detects it exactly, with no learning, no false positives from one-off repetitions, and no false negatives from parameter variation (because the graph aggregates over distinct nodes).

The structural classifier in `solution.py` already exploits exact cycle detection. The intervention policy operates on the same representation for the same reason.

**2. Causal distance ≠ temporal distance.**

Consider a `RETRIED` edge that connects retry event #50 to its original failed attempt at event #2.

- *Sequence view:* this is a 48-step long-range dependency. A transformer with limited context window may not even hold both endpoints simultaneously. With a long enough window, the model can in principle attend across, but it must learn that the connection exists from positional patterns alone.
- *Graph view:* a single edge. The retry node has a direct neighbor 48 steps away in the sequence but 1 hop away in the graph. Anything that does message passing over the graph reaches both endpoints in a single layer of computation.

This matters operationally because **the right rewind target is exactly this kind of thing**. To pick a rewind point, the policy needs to find "the most recent step from which a different choice would have avoided the current failure" — which is a lowest-common-ancestor query over the causal subgraph. Graph operation: cheap. Sequence-attention approximation: lossy.

**3. Branches and swarms break linearization.**

When a session forks (a sub-agent in the swarm starts working in parallel), there is no single sequence. Two reasonable linearizations exist:

- *Pick one branch:* throws away information from the other.
- *Interleave both:* creates spurious adjacencies (event from branch A and event from branch B end up next to each other in the sequence even though they're causally unrelated).

Either way, the linear sequence is lying about the structure. The graph isn't — the swarm's events are nodes with `BRANCHED_FROM` edges back to the parent, and the policy sees the actual structure.

### Where Sequences Genuinely Win

To be honest, sequences have one real advantage: **recency is intrinsic**. A retry chain that ended 80 steps ago and one that's still active are obviously distinguishable in a sequence model — the latter is at the end of the input, the former is buried in the middle. In the graph, this distinction is lost in a flat encoding; you have to add it back via temporal attention or explicit recency features.

This is exactly why our state representation is **hybrid** rather than pure-graph. We use the GNN encoding for structural awareness and a separate recent-window encoder over the last K events for recency. Pure graph would lose recency; pure sequence loses structure; the hybrid is honest about needing both.

### A Concrete Decision Where the Choice Matters

The action `decompose(subtasks)` requires the policy to look at the current execution and identify subgraphs corresponding to coherent sub-tasks — which sub-graphs are healthy and should be preserved, which are degenerate and should be re-attempted under decomposition.

A graph-native policy can do this directly. Subgraph identification is a graph operation; community detection algorithms run on the (action, target) graph give us candidate decompositions for free.

A sequence policy cannot. There's no notion of "subgraph" in a sequence. To approximate, the model would have to learn implicit clustering from attention patterns — a much harder problem with much less inductive bias.

`spawn_helper(specialist)` has the same property: choosing what kind of helper to spawn requires identifying which part of the execution is failing and what kind of expertise it needs. That's structural reasoning over subgraphs again.

### Summary

| Property | Linear Sequence | Graph-Native |
|---|---|---|
| Cycle detection | Learned, approximate, error-prone | Exact, single graph operation |
| Long-range causal links | Long-range attention required | Single edge hop |
| Branches / swarms | Forced linearization, lossy | Natural representation |
| Subgraph identification | Implicit, hard to learn | Direct graph operations |
| Recency | Intrinsic | Requires explicit modeling |
| Inductive bias for *this* problem | Generic sequence prior | Causal graph structure |
| Performance ceiling on small data | Lower | Higher |

The summary claim: **the agent's execution is causal-graph-shaped, not sequence-shaped, and the right representation matches the data.** The intervention decisions we care about — rewind targeting, decomposition, helper spawning — are all expressed naturally as graph operations and unnaturally as sequence operations. We choose the representation that admits these operations directly.

This is not a claim that graphs are *always* better than sequences. For language modeling, sequences are obviously right; for protein structure, graphs are obviously right. For autonomous agent execution, we argue graphs are right — and we argue this for specific, problem-grounded reasons rather than as a general aesthetic preference.

---

## 10. Implementation Plan and Risks

In rough order of work:

1. Define the intervention command schema and integrate it with the existing control plane so actions can be delivered over the CLI's reverse command stream and recorded with command lifecycle events. (~2 weeks)
2. Instrument the existing event log so intervention intents, executions, and outcomes are first-class events — converts existing data into a partial replay buffer and simulator-calibration set immediately. (~1 week)
3. Train the LLM-based semantic diagnoser against ground-truth diagnoses derived from historical sessions and human review. Fine-tune an open-source LLM (Llama-3-70B or comparable) hosted by Sploink. (~6 weeks, mostly data labeling)
4. Extend `simulate.py` to a full agent-environment simulator with intervention-response and outcome models. Calibrate against historical data. (~4 weeks)
5. Implement the graph-conditioned state encoder (graph encoder + recent-window encoder), using the structural classifier's labels as a bootstrap signal where helpful. (~2 weeks)
6. Train the PPO meta-controller in the simulator with domain randomization, then pair it with the LLM action parameterizer for concrete action instantiation. Iterate against in-simulator baselines. (~6 weeks, mostly debugging)
7. DR-evaluate against historical operator data, gated by BC baseline. If it doesn't beat BC, return to step 6.
8. Shadow-mode deploy over the existing control plane so actions are logged but not executed. (~4 weeks of operating before next stage)
9. Tier-3 rollout with conservative action gating and manual review of `terminate` decisions. (Per §7)

The largest risk is **simulator calibration**. The intervention-response model is the load-bearing component of the whole approach. If predictions of "what happens after a `rewind`" are systematically wrong, the policy will be too. Mitigations: domain randomization, DR off-policy evaluation, gradual rollout.

The second-largest risk is **semantic-layer quality**. The diagnoser can hallucinate and the action parameterizer can instantiate a sensible action type badly. Mitigations are described in §8.2, and the rollout path in §7 further limits blast radius.

The third risk is **scope.** The system as described is roughly six months of engineering work for a small team. We're explicit that it should ship in tiers — the structural classifier alone is already valuable, the policy adds value on top, and the LLM diagnoser adds value on top of the policy. Each layer can ship independently.

---

## References

- **Andrychowicz, M., Wolski, F., Ray, A., et al.** (2017). *Hindsight Experience Replay.* NeurIPS 2017.
- **Bacon, P.-L., Harb, J., & Precup, D.** (2017). *The Option-Critic Architecture.* AAAI 2017.
- **Battaglia, P. W., Hamrick, J. B., Bapst, V., et al.** (2018). *Relational inductive biases, deep learning, and graph networks.* arXiv:1806.01261.
- **Bertsekas, D. P.** (2019). *Reinforcement Learning and Optimal Control.* Athena Scientific.
- **Hausknecht, M., & Stone, P.** (2015). *Deep Recurrent Q-Learning for Partially Observable MDPs.* AAAI Fall Symposium 2015.
- **Jiang, N., & Li, L.** (2016). *Doubly Robust Off-policy Value Evaluation for Reinforcement Learning.* ICML 2016.
- **Kostrikov, I., Nair, A., & Levine, S.** (2022). *Offline Reinforcement Learning with Implicit Q-Learning.* ICLR 2022.
- **Kumar, A., Zhou, A., Tucker, G., & Levine, S.** (2020). *Conservative Q-Learning for Offline Reinforcement Learning.* NeurIPS 2020.
- **Ng, A. Y., Harada, D., & Russell, S.** (1999). *Policy Invariance Under Reward Transformations: Theory and Application to Reward Shaping.* ICML 1999.
- **Rolnick, D., Ahuja, A., Schwarz, J., Lillicrap, T. P., & Wayne, G.** (2019). *Experience Replay for Continual Learning.* NeurIPS 2019.
- **Schlichtkrull, M., Kipf, T. N., Bloem, P., et al.** (2018). *Modeling Relational Data with Graph Convolutional Networks.* ESWC 2018.
- **Schulman, J., Wolski, F., Dhariwal, P., Radford, A., & Klimov, O.** (2017). *Proximal Policy Optimization Algorithms.* arXiv:1707.06347.
- **Thomas, P. S., & Brunskill, E.** (2016). *Data-Efficient Off-Policy Policy Evaluation for Reinforcement Learning.* ICML 2016.
- **Tobin, J., Fong, R., Ray, A., et al.** (2017). *Domain Randomization for Transferring Deep Neural Networks from Simulation to the Real World.* IROS 2017.
- **Wang, T., Liao, R., Ba, J., & Fidler, S.** (2018). *NerveNet: Learning Structured Policy with Graph Neural Networks.* ICLR 2018.
