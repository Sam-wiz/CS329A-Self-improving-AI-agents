# CS329A — Lecture 3 Notes
## Verification

**Lecture theme:** Lecture 2 showed that models can generate correct answers among many samples. Lecture 3 asks how to automatically identify those answers or guide generation toward them. This is the **generation–verification gap**. The lecture presents four papers as a progression: outcome verification → process verification → automatic process supervision → weak-verifier ensembles.

---

## 1. Core problem: generation vs verification

Suppose a generator produces:

\[
s_1,s_2,\ldots,s_N \sim G_\theta(q)
\]

for question \(q\). Even if one candidate is correct, the system still needs a selector:

\[
s^*=\arg\max_{s_i} V_\phi(q,s_i)
\]

where \(V_\phi\) is a verifier.

The central question is therefore:

> **How can we automatically select correct solutions, or guide the model during solution generation?**

The lecture uses GSM8K as an early math-reasoning benchmark: about 8,500 grade-school math problems designed to require multi-step reasoning. fileciteturn7file0L20-L58

---

# 2. Paper 1 — Training Verifiers to Solve Math Problems

### Goal

Train a verifier that predicts whether a generated solution is correct.

Given a question \(q\), candidate solution \(s\), and binary label \(y\):

\[
y\in\{0,1\}
\]

Train:

\[
V_\phi(q,s)\rightarrow P(y=1\mid q,s)
\]

### Training procedure

1. Start with problems with known ground-truth answers.
2. Fine-tune a generator.
3. Generate many solutions per problem (the lecture describes 100 completions).
4. Compare each completion with the known answer.
5. Label each completion correct/incorrect.
6. Train the verifier.

The verifier is itself a language model with a scalar prediction head. The paper also combines a binary verification objective with a language-modeling objective. fileciteturn7file0L66-L125

### Test-time procedure

```text
Question
   ↓
Generator
   ↓
N candidate solutions
   ↓
Verifier scores candidates
   ↓
Highest-scoring solution
```

This is learned **Best-of-N** selection.

### Token/sentence supervision

The lecture discusses sentence-level and token-level correctness labels. Token-level labels are noisy because individual tokens do not map cleanly to semantic correctness. The final token's score can be used as the score of the entire solution. fileciteturn7file0L136-L166

### Important results

Verification becomes more useful as verifier training data grows and can outperform fine-tuning-only baselines in the presented experiments. With small verifier datasets, verification may not help much. fileciteturn7file0L186-L203

The lecture also compares generator/verifier capacity. The reported result favors **larger generator + smaller verifier** over the reverse configuration, motivating the research question:

> **How should model capacity be divided between generation and verification under a fixed compute budget?**

The lecture notes that verification gains can eventually flatten or decline as the number of sampled candidates becomes large: the verifier has increasing difficulty distinguishing very similar candidates, even though the candidate coverage may continue to rise. fileciteturn7file0L214-L245

### Key lesson

A strong generator is not enough. A system also needs a sufficiently capable selector.

---

# 3. Paper 2 — Let's Verify Step by Step

The first verifier mainly asks:

> **Is the final solution correct?**

The second paper asks:

> **Can we verify the reasoning process itself?**

## ORM — Outcome Reward Model

An ORM scores the whole solution:

\[
ORM(q,s)\rightarrow r_{outcome}
\]

It mainly evaluates final correctness.

## PRM — Process Reward Model

A PRM evaluates intermediate reasoning steps:

\[
PRM(q,s_1)\rightarrow r_1
\]

\[
PRM(q,s_1,s_2)\rightarrow r_2
\]

and so on.

The lecture explicitly distinguishes outcome-based reward from process-based reward. fileciteturn7file0L367-L386

### Why PRM?

A model might produce:

```text
Step 1 → correct
Step 2 → correct
Step 3 → wrong
Step 4 → wrong
Final answer → accidentally correct
```

An ORM can simply see a correct final answer. A PRM can identify the bad intermediate step.

This provides finer-grained information for:

- credit assignment
- search
- pruning
- trajectory selection
- reasoning diagnosis

The lecture calls this **process supervision**. fileciteturn7file0L427-L450

---

## PRM training and PRM800K

Human annotators label individual reasoning steps as correct/incorrect/neutral. This produced **PRM800K**, an open dataset with roughly 800k step-level labels.

An especially useful annotation strategy prioritized **convincing wrong** solutions: cases where the final answer could be correct while intermediate reasoning was incorrect. The lecture says this made annotation substantially more data-efficient. fileciteturn7file0L453-L475

### PRM trajectory score

The lecture describes the final process score as based on the product of stepwise rewards:

\[
R_{trajectory}\propto\prod_{t=1}^{T}r_t
\]

Thus a weak intermediate step can strongly affect the overall trajectory score. fileciteturn7file0L400-L426

### Reported behavior

The lecture reports PRM outperforming ORM and majority voting in the presented experiments and notes that PRM can identify rare correct solutions, including cases where correct solutions appear in less than 5% of sampled generations. fileciteturn7file0L503-L535

---

# 4. Critical distinction: PRM ≠ search algorithm

This is important terminology.

**PRM provides a score.**

A search procedure uses that score to decide what to expand/prune/select.

```text
Generator
   ↓
Candidate reasoning branches
   ↓
PRM scores
   ↓
Search algorithm
   ↓
Expand / prune / select
```

Do not say “the PRM searches the tree.” More precisely, the **search algorithm is guided by PRM scores**.

---

# 5. PRM failure modes

### 5.1 Miscalibration

A correct step can receive a low score; an incorrect step can receive a high score.

### 5.2 Early pruning

A correct trajectory can look bad early. If it is pruned, search loses a potentially correct solution.

The tradeoff is:

\[
\text{compute saved}
\quad vs \quad
\text{correct trajectories accidentally pruned}
\]

### 5.3 Human-label bias

PRM800K uses human annotations. Therefore the PRM learns human judgments of reasoning quality. A valid but unusual reasoning strategy could be scored poorly.

### 5.4 Reward hacking / optimization risk

If the generator itself is optimized against a learned PRM, the generator may learn to exploit weaknesses in the reward model rather than genuinely improve reasoning. The lecture notes that this becomes a caveat when using the PRM for generator training. fileciteturn7file0L588-L650

---

# 6. Paper 3 — Math-Shepherd
## Verify and reinforce LLMs step-by-step without human annotations

### Motivation

PRMs are promising, but step-level human annotation is expensive.

Question:

> **Can we automatically generate process-supervision labels?**

Math-Shepherd defines the quality of a partial reasoning step by its potential to eventually reach the correct final answer. From a given step, sample multiple continuations and inspect their final outcomes. fileciteturn7file0L661-L703

---

## Hard estimate

A step gets a positive label if **any** sampled continuation eventually reaches a correct answer:

\[
R_{hard}(s)=\mathbf{1}[\exists\text{ correct continuation}]
\]

Example:

```text
wrong
correct
wrong
```

\[
R_{hard}=1
\]

## Soft estimate

Use the fraction of successful continuations:

\[
R_{soft}(s)=\frac{\#\text{correct continuations}}{N}
\]

Example:

```text
wrong
correct
correct
```

\[
R_{soft}=2/3
\]

---

# 7. Math-Shepherd failure modes

## Rare valid trajectories

Suppose:

\[
P(\text{eventual success}\mid s)=0.01
\]

but only 3 continuations are sampled. We can easily observe zero successes and incorrectly estimate the step as useless.

The lecture explicitly identifies this problem: unusual valid paths can receive low scores if the rollout budget is too small. fileciteturn7file0L704-L727

## Wrong intermediate steps with correct final answer

A trajectory can contain incorrect reasoning and still reach the correct final answer. Automatic labels based on eventual success can therefore assign positive credit to a flawed intermediate step. More samples may expose this, but it is not guaranteed. fileciteturn8file6

---

# 8. Math-Shepherd + RL

The automatically generated PRM can become a reward model for the generator.

```text
Generator
   ↓
Generate reasoning
   ↓
Automatic rollout labels
   ↓
Train PRM
   ↓
PRM reward
   ↓
RL / PPO generator
   ↓
Improved generator
```

The lecture reports that Math-Shepherd used PPO with the PRM and obtained stronger results than using an ORM as the RL reward in the presented experiments. The approach therefore creates multiple levels of self-improvement: the model helps generate supervision, the PRM is trained from it, and the PRM is then used to improve the generator. fileciteturn8file4L1-L18

The lecture also notes that improvements eventually plateaued.

---

# 9. Our research idea: adaptive rollout allocation

We developed this during the lecture.

Suppose two partial trajectories have estimated success probabilities:

\[
P(A)=0.40,
\qquad
P(B)=0.05
\]

Instead of allocating all inference compute uniformly:

1. Give every branch an initial exploration budget.
2. Estimate success probability.
3. Allocate additional compute preferentially to promising branches.
4. Keep some exploration budget for uncertain/rare branches.

Example:

```text
Initial:
A → 20 rollouts → 8 successes = 40%
B → 20 rollouts → 1 success  = 5%

Remaining compute:
Allocate more to A,
but do not necessarily give B zero additional exploration.
```

Why not discard B completely? Because an observed rate of 0/20 does **not** prove the true success probability is zero.

### Stronger research formulation

Instead of simply choosing the highest estimated probability, consider:

\[
\text{priority}(s)
\approx
\frac{\Delta\text{expected accuracy}}
{\Delta\text{compute}}
\]

This turns the problem into **adaptive inference-time resource allocation**.

### Hypothesis

> Under a fixed inference budget, adaptive rollout allocation based on estimated future success can outperform uniform rollout allocation.

### Baseline

Uniform allocation:

\[
B/N
\]

rollouts per trajectory.

### Metrics

- final accuracy
- inference tokens/FLOPs
- success rate
- rare-trajectory recovery
- calibration
- performance vs compute

### Ablations

- no uncertainty modeling
- no exploration reserve
- different initial budgets
- different allocation rules
- different problem difficulties

This is the kind of formulation we want to develop throughout CS329A: **idea → hypothesis → baseline → controlled experiment → ablations → failure analysis**.

---

# 10. Paper 4 — Shrinking the Generation–Verification Gap with Weak Verifiers

The final paper takes a different approach.

Instead of training one stronger verifier:

> **Can we combine many imperfect verifiers into a stronger aggregate verifier?**

“Weak” means imperfect, not intentionally bad. Candidate verifier types include:

- ORMs
- PRMs
- LLM judges
- rubric-based judges
- tool-assisted evaluators

Each produces a noisy signal correlated with true correctness. fileciteturn9file3

---

# 11. Weaver

The lecture describes **Weaver** using:

> **score → weight → select**

### Score

Run candidate solutions through multiple verifiers.

### Normalize

Put verifier outputs onto comparable scales.

### Filter

Remove very low-quality verifiers using limited labeled data.

### Weight

Estimate how much each verifier should contribute.

### Select

Combine the verifier outputs and choose the best candidate.

The lecture specifically discusses simple weighting methods such as logistic regression and Naive Bayes. fileciteturn10file0turn10file1

---

# 12. Why not simply average verifiers?

Naive ensemble:

\[
S(s)=\frac{1}{M}\sum_{j=1}^{M}V_j(s)
\]

But different verifiers have different quality.

A weighted ensemble is:

\[
S(s)=\sum_{j=1}^{M}w_jV_j(s)
\]

where \(w_j\) estimates how useful verifier \(j\) is.

The lecture reports that learned weighting improves over naive ensembling in the presented experiments. fileciteturn10file0

---

# 13. Weak supervision and verifier diversity

Suppose:

- \(n\) queries
- \(k\) solutions per query
- \(m\) verifiers

Then we obtain:

\[
n\times k\times m
\]

weak labels.

The goal is to estimate the probability that a candidate is correct using the collective verifier signals.

A key assumption is that different verifiers capture somewhat different aspects of correctness. If every verifier makes exactly the same mistake, the ensemble gains little information.

Therefore:

> **More verifiers do not automatically mean more useful information. Verifier diversity matters.**

The lecture describes the approach as weak-to-strong aggregation and connects it to weak-supervision work such as Snorkel. fileciteturn10file3

---

# 14. Verification compute can be scaled in multiple ways

You can spend more inference compute by:

1. Sampling more candidate generations.
2. Using larger generators.
3. Using larger verifiers.
4. Increasing the number of verifiers.

Thus a broader optimization problem is:

\[
\boxed{\text{How should limited inference FLOPs be allocated between generation and verification?}}
\]

The lecture compares Weaver against a Pass@K oracle, supervised weighting, an unsupervised/low-label setting, naive ensembles, majority voting, and multi-agent verification. fileciteturn10file2

---

# 15. Reported lecture results

The lecture reports notable gains from weak-to-strong verifier aggregation on harder datasets including GPQA Diamond, MATH, and MLU Pro.

It also describes an 8B generator with a pool of 8B-and-below verifiers reaching around 70% average accuracy in the cited setup, approaching the performance of much larger-class systems. For the 70B-class setup, the lecture reports about 86.2% average accuracy on the cited datasets and compares it with o3-mini. These are **lecture-reported experimental results**, not universal current-model claims. fileciteturn10file4turn9file2

---

# 16. Distilling the verifier ensemble

An ensemble is expensive.

If there are:

\[
100\text{ candidates}\times20\text{ verifiers}=2000
\]

verifier evaluations.

The lecture describes distilling Weaver into a much smaller model.

Reported result:

- original pool included models around the 70B range;
- distilled model was around 400M parameters;
- distilled model captured about 97% of the accuracy of the large verifier pool;
- it used substantially less test-time compute, with the lecture reporting a 99%+ reduction in the relevant compute comparison.

The original and distilled versions were described as open-sourced in the lecture. fileciteturn10file9

---

# 17. Four-paper progression

| Paper | Main idea | Limitation it addresses |
|---|---|---|
| Training Verifiers to Solve Math Problems | Outcome verification / ORM | Which final answer is correct? |
| Let's Verify Step by Step | Process verification / PRM | Where did reasoning go wrong? |
| Math-Shepherd | Automatic process supervision | Human step annotation is expensive |
| Shrinking the Generation–Verification Gap with Weak Verifiers | Weaver / verifier ensemble | Individual verifiers are imperfect |

### Evolution

```text
Outcome verification
        ↓
Process verification
        ↓
Automatic process supervision
        ↓
Weak verifier ensembles
```

The lecture closes by emphasizing that verification can improve both training and inference; process rewards are powerful but may benefit from combination with outcome rewards; and Weaver uses weak-supervision aggregation rather than training a single new verifier. fileciteturn9file0

---

# 18. Major failure modes to remember

### 1. Verifier error
Correct candidate receives low score.

### 2. False positive
Incorrect candidate receives high score.

### 3. Rare correct trajectory
Correct solution is generated rarely and is missed.

### 4. Early pruning
A correct trajectory looks bad early and gets discarded.

### 5. Correlated verifier errors
Many verifiers make the same mistake, so ensembling adds little information.

### 6. Human-label bias
PRM learns human preferences about reasoning.

### 7. Reward hacking
Generator learns to exploit the verifier rather than genuinely improve.

### 8. Compute explosion
More samples × more verifiers can become prohibitively expensive.

---

# 19. Research questions from Lecture 3

1. **Generator/verifier allocation:** How should capacity be divided between generator and verifier under fixed compute?
2. **Adaptive verification:** Can inference compute be dynamically allocated according to difficulty or marginal benefit?
3. **Rare trajectories:** How can search preserve rare but valid reasoning paths?
4. **PRM reliability:** How accurate must a PRM be before PRM-guided search beats simpler inference scaling?
5. **ORM + PRM:** Can combining outcome and process rewards improve reliability?
6. **Verifier diversity:** How much does diversity matter compared with simply adding more verifiers?
7. **Self-improving verification:** Can a verifier generate useful training data without amplifying its own errors?
8. **Tool-assisted verification:** Can calculators, symbolic solvers, unit tests, simulators, or other tools provide stronger signals?

The lecture explicitly points toward tool use, test-time scaling, and agentic approaches as ways to add supervision and improve verification. fileciteturn8file7

---

# 20. Research methodology takeaway

Lecture 3 reinforces the research loop:

```text
Existing method
      ↓
Identify limitation
      ↓
Research question
      ↓
Hypothesis
      ↓
Baseline
      ↓
Controlled experiment
      ↓
Ablations
      ↓
Failure analysis
      ↓
Conclusion
```

The critical habit is to ask:

- Why does this work?
- When does it fail?
- What assumptions does it make?
- What happens under a fixed compute budget?
- Is the improvement from the proposed method or simply more compute/data?
- Can the result be reproduced?
- Which component actually causes the gain?

---

# 21. Final mental model

The entire lecture can be compressed into one chain:

\[
\boxed{
\text{Generate}
\rightarrow
\text{Verify}
\rightarrow
\text{Select/Guide}
\rightarrow
\text{Improve}
}
\]

### ORM
**“Is the final solution correct?”**

### PRM
**“Are the intermediate steps good?”**

### Automatic process supervision
**“Can rollout success tell us whether a partial trajectory is promising?”**

### Weak verifier ensemble
**“Can many imperfect signals be combined into a stronger verifier?”**

### Central research problem

> **How should limited inference compute be spent to maximize the probability of finding and correctly selecting a valid solution?**

This connects Lecture 2's **inference-time scaling** directly to Lecture 3's **verification**.
