# CS329A — Lecture 2: Inference Scaling

## 1. Core idea

Lecture 2 is about **inference scaling / test-time scaling**: improving an LLM's usefulness at inference time without changing its parameters or fine-tuning it.

The central question is:

> Given a limited inference-compute budget, how should we spend that compute to maximize final answer quality?

The lecture progresses from repeated sampling to verification, reward models, search, fusion, and finally Archon, which treats inference scaling as an architecture-search problem.

---

## 2. Three stages of LLM development

### Pre-training
- Historically the most compute-intensive stage.
- Can take months and use very large GPU clusters.
- Can involve trillions of tokens.

### Fine-tuning
- Historically much less compute-intensive than pre-training.
- Uses much less data.

### Inference
- The trained model is used to solve problems.
- Lecture focus: improve performance here **without changing parameters and without fine-tuning**.

---

# Part I — Repeated Sampling

## 3. Large Language Monkeys

The **Large Language Monkeys** paper starts with a simple idea:

Instead of generating one answer:

```text
Question → Model → Answer
```

generate many:

```text
Question → Model → Answer 1
                 → Answer 2
                 → Answer 3
                 → ...
                 → Answer N
```

Then use a verifier to select a correct response.

Repeated sampling can make smaller/open models substantially stronger in some benchmark settings because a correct solution that is not produced on the first attempt may appear in later samples.

The lecture describes examples in:
- difficult mathematics;
- coding;
- question answering;
- agentic benchmarks such as SWE-bench.

---

## 4. Eliciting / surfacing correct answers

A key intuition from the lecture is:

> A model may already be capable of generating a correct answer, but a single attempt may fail to surface it.

Repeated sampling gives the model more opportunities to produce different trajectories.

This is not parameter learning. The model is not becoming permanently smarter.

Instead:

```text
same model
   ↓
more attempts
   ↓
more candidate solutions
   ↓
higher chance of encountering a correct solution
```

---

## 5. Coverage

**Coverage** is the fraction of problems for which at least one generated sample is correct.

For a benchmark of 100 problems, if at least one correct solution is generated for 70:

\[
Coverage = 70\%
\]

Coverage answers:

> Did the model generate a correct answer somewhere?

It does **not** answer:

> Did the system successfully select that correct answer?

This distinction is fundamental.

---

## 6. Pass@N

If one independent sample has probability \(p\) of being correct, then the probability that at least one of \(N\) samples is correct is:

\[
oxed{P(	ext{at least one correct})=1-(1-p)^N}
\]

Example:

\[
p=0.1,\quad N=20
\]

gives:

\[
P(	ext{at least one correct})=1-(0.9)^{20}
\]

The formula explains why increasing inference samples can increase the probability of finding a correct solution.

---

## 7. Long-tail difficulty

Benchmark problems are not equally difficult.

- Easy problems may be solved by many samples.
- Hard problems may be solved only rarely.
- Extremely hard problems may have a correct solution appear only once, twice, or three times among 1,000–10,000 samples.

This long tail explains why coverage can continue increasing with additional inference compute.

It also explains why selection becomes difficult.

---

# Part II — Verification

## 8. Why verification matters

Suppose 1,000 answers are generated and one is correct.

If the selector cannot identify that answer, the theoretical coverage does not become final system accuracy.

Therefore:

\[
oxed{	ext{Generation capability} 
eq 	ext{Final system capability}}
\]

The verifier/selector is a critical component.

---

## 9. Domains with strong verification

Some domains have relatively objective correctness signals.

### Mathematics
Formal or objective checking can sometimes verify correctness.

### Coding
Generated code can be executed against unit tests.

```text
Generated code
     ↓
Unit tests
     ↓
Pass / Fail
```

The lecture discusses CUDA generation as an example: generated CUDA code can be compared against the behavior of a source implementation such as PyTorch.

### Translation
Programs translated between languages can sometimes be checked by comparing behavioral equivalence.

---

## 10. Domains without perfect verifiers

Many tasks do not have an objective, cheap, perfect verifier.

Possible alternatives include:

- majority voting;
- model-based rankers;
- reward models;
- verifier ensembles.

These alternatives can leave a large gap between the model's potential coverage and final accuracy.

---

## 11. Majority voting

Generate multiple responses and choose the most frequent answer.

Example:

```text
A
A
B
A
C
A
B
```

Majority vote chooses A.

### Failure mode

The correct answer may be rare.

For example:

```text
Wrong A    → 900
Wrong B    → 97
Correct C  → 3
```

Majority voting returns A even though the model generated the correct solution.

This is particularly problematic for hard problems where correct solutions can be very rare.

---

## 12. Generation–Verification Gap

The lecture calls this the:

\[
oxed{	ext{Generation–Verification Gap}}
\]

It is the gap between:

1. what correct solutions the model can generate; and
2. what correct solutions the verification/selection mechanism can successfully identify.

Conceptually:

```text
Generated candidates
       ↓
Potentially many correct answers
       ↓
   verification
       ↓
Correct answers successfully captured
```

A strong generator with a weak verifier can therefore waste substantial inference compute.

---

## 13. Reward models

A reward model is an LLM-based model trained to score response quality.

Example:

```text
Answer A → 0.62
Answer B → 0.91
Answer C → 0.34
```

The system can select the highest-scoring answer.

However, reward models are imperfect, especially when applied to new domains.

Thus they do not automatically eliminate the generation–verification gap.

---

## 14. Verification research directions

The lecture proposes several directions:

### Better verifiers
Improve the ability to identify correct candidates.

### Asymmetric verification
Look for domains where:
- proving correctness is hard/expensive;
- detecting incorrectness is easier.

### Simulation
Use domain-specific simulations as verification signals, including examples such as physics or molecular-dynamics simulations.

### Tool use
Use external tools to filter incorrect answers.

### Ensembles of verifiers
Use multiple imperfect verifiers together.

The lecture mentions work with roughly 10–20 verifiers and notes the computational cost.

### Verifier quality
A verifier itself can be wrong.

For example, a unit-test suite may not fully cover a generated program. Passing the tests therefore does not necessarily establish global correctness.

---

# Part III — Beyond Repeated Sampling

## 15. Other forms of test-time scaling

The lecture mentions:
- self-study;
- search;
- tool use.

The important transition is:

> Instead of simply generating more independent answers, use inference compute to improve the reasoning/search process.

---

## 16. Parallel vs sequential scaling

### Parallel sampling

Generate multiple independent candidate solutions.

```text
Question
 ├── Solution 1
 ├── Solution 2
 ├── Solution 3
 └── ...
```

This provides **breadth**.

### Sequential revision

Generate a solution and repeatedly improve it.

```text
Initial solution
      ↓
Revision 1
      ↓
Revision 2
      ↓
Revision 3
```

This provides **depth**.

---

## 17. Fixed-budget allocation

Suppose we have a budget of 64 generation units.

Possible allocations include:

```text
64 × 1
32 × 2
16 × 4
8 × 8
4 × 16
1 × 64
```

These represent different breadth/depth tradeoffs.

The central question becomes:

> For a fixed inference budget, how should computation be divided between parallel exploration and sequential refinement?

---

## 18. Difficulty-dependent intuition

The lecture discusses an intuitive hypothesis:

### Easier problems
Sequential refinement may be sufficient because many trajectories can reach the correct answer.

### Harder problems
Parallel exploration may be more useful because finding the right trajectory may be difficult.

This is a hypothesis, not a universal rule.

The optimal allocation depends on:
- problem difficulty;
- model;
- verifier;
- inference budget;
- task.

---

# Part IV — ORM and PRM

## 19. Outcome Reward Model (ORM)

An **Outcome Reward Model (ORM)** scores the final outcome.

```text
Question
   ↓
Solution
   ↓
ORM
   ↓
Final score
```

For best-of-N:

1. Generate N candidates.
2. Score each with the ORM.
3. Select the highest-scoring candidate.

---

## 20. Process Reward Model (PRM)

A **Process Reward Model (PRM)** scores intermediate steps.

For a solution:

```text
Step 1
Step 2
Step 3
Step 4
Step 5
```

the PRM gives step-level scores.

The lecture explicitly notes:

> PRM scoring is **per step**, not simply per token.

---

## 21. ORM vs PRM

| | ORM | PRM |
|---|---|---|
| Full name | Outcome Reward Model | Process Reward Model |
| Evaluates | Final outcome | Intermediate steps |
| Signal | Final quality | Step quality |
| Typical use | Candidate selection | Search guidance |

Memorize:

\[
oxed{ORM ightarrow Outcome}
\]

\[
oxed{PRM ightarrow Process}
\]

---

## 22. PRM-guided beam search

Suppose at each level we generate four candidate next steps:

```text
A → 0.9
B → 0.8
C → 0.3
D → 0.2
```

The PRM scores them.

Keep the top two:

```text
A
B
```

Then expand A and B and continue.

This is a beam-search-style approach where the PRM guides which branches survive.

Important distinction:

> **The PRM provides scores; the search procedure uses those scores to decide which branches to retain and expand.**

The PRM itself is not the search algorithm.

---

## 23. PRM tradeoff

PRM-guided search saves computation by pruning bad branches.

But it can also make mistakes.

If a correct trajectory receives a low intermediate score:

```text
Correct trajectory
       ↓
Low PRM score
       ↓
Pruned
       ↓
Correct final answer never reached
```

So there is a tradeoff:

```text
More pruning → lower compute, higher risk of losing good branches
Less pruning  → higher compute, more exploration
```

A natural research question is:

> How reliable must a PRM be before PRM-guided search outperforms simpler inference-scaling strategies?

---

## 24. PRM training and generalization

PRMs are generally fine-tuned from language models.

A PRM trained on in-domain data can work particularly well on that task.

However, because PRMs are language-model based, they can also generalize across tasks.

The lecture describes an experiment with:
- approximately 12k training questions;
- approximately 500 test questions;
- a PaLM model;
- five difficulty bins based on Pass@1.

This setup allows researchers to study how different test-time scaling strategies behave as problem difficulty changes.

---

# Part V — Combining Scaling Strategies

## 25. Parallel + sequential scaling

An important open problem is how to combine:

- parallel sampling;
- sequential revision.

For example:

```text
Parallel sampling
       ↓
Select/filter
       ↓
Sequential revision
       ↓
Parallel sampling
       ↓
Final selection
```

The goal is not simply maximum accuracy.

The desired objective is often:

> **For each target accuracy, minimize the required inference-generation budget.**

This is the **quality–compute frontier**.

---

# Part VI — Fusion, Critic, Ranker, Verifier

## 26. Fusion

**Fusion** takes K candidate responses and asks an LLM to synthesize one new answer using all of them.

```text
Answer 1 ─┐
Answer 2 ─┤
Answer 3 ─┤
Answer 4 ─┤ → Fusion model → New final answer
Answer K ─┘
```

The model sees the original question plus the candidate responses.

Fusion is therefore different from simple selection.

### Selection

```text
A → score
B → score
C → score
↓
Choose B
```

### Fusion

```text
A ─┐
B ─┤
C ─┤ → model → NEW answer
```

The lecture reports that fusion can outperform Oracle selection in the cited experiment.

---

## 27. Critic

A **critic** examines a response and describes its:

- strengths;
- weaknesses;
- problems.

```text
Response
   ↓
Critic
   ↓
Strengths / weaknesses
```

The critique can be used by later inference operations.

---

## 28. Ranker

A **ranker** takes candidate responses and orders them by perceived quality.

```text
A
B
C
D
↓
Ranker
↓
B > D > A > C
```

These operations are generally prompting-based inference-time optimization; the model does not necessarily need special training specifically for the operation.

---

## 29. Verifier

A **verifier** produces a correctness/quality assessment.

In the described Archon setup, it can also explain why it assigned its score.

The ideal verifier would identify correctness perfectly, but practical verifiers can be imperfect.

---

# Part VII — Comparing Inference-Time Operations

## 30. Random selection

Generate candidates and randomly select one.

This is a baseline.

---

## 31. Model ranking

Generate candidates, ask a model to rank them, and return the top-ranked response.

The lecture reports this improves over random selection in the cited experiment.

---

## 32. Oracle selection

Assume an ideal verifier can identify the correct/best generated response.

This is an upper-bound-style reference.

---

## 33. Fusion

Give multiple candidates to an LLM and ask it to synthesize a final answer.

The lecture reports that fusion can outperform Oracle selection in the cited experiment.

This is notable because fusion can create a new response instead of merely choosing an existing one.

---

## 34. Filtering + fusion

Another approach is:

```text
Generate many responses
        ↓
Filter/select promising responses
        ↓
Fuse selected responses
        ↓
Final answer
```

The cited experiment found this filtering-plus-fusion strategy useful.

---

# Part VIII — Multiple Models

## 35. Model ensembles

Inference scaling can use multiple models instead of one.

For example:

```text
Model A → Answer A
Model B → Answer B
Model C → Answer C
...
```

The outputs can then be:

- ranked;
- critiqued;
- fused;
- verified.

The lecture compares configurations using different numbers of models and observes similar trends in the cited experiments.

The core idea:

> Different models can provide diverse candidate solutions that later inference operations can combine.

---

# Part IX — Archon

## 36. Archon: inference architecture search

**Archon** is introduced as an:

> **Architecture search framework for inference-time scaling.**

The key shift is:

Instead of asking:

> Which inference technique should I use?

ask:

> **What entire inference-time architecture should I construct?**

For example:

```text
Generate → Critic → Ranker → Fusion
```

versus:

```text
Generate → Verify → Fusion
```

versus a much deeper architecture.

---

## 37. Archon's optimization problem

Archon tries to optimize the:

\[
oxed{	ext{capability/correctness vs inference cost}}
\]

frontier.

The lecture frames this as:

> How can we assign inference compute to different tasks so that we get high-quality answers without wasting tokens and generations?

---

## 38. Archon inputs

Archon receives:

### Target benchmarks
The benchmarks on which performance should be optimized.

### Inference compute budget
The available inference resources.

### Available LLMs
Multiple models can be mixed and matched.

### Inference-time techniques
Such as:
- generation;
- fusion;
- critic;
- ranker;
- verifier;
- unit-test generation;
- evaluation.

---

## 39. itest

Archon's optimizer is called:

\[
oxed{	ext{itest}}
\]

and is described as **inference-time architecture search**.

Its job is to determine how to combine:

- models;
- inference-time techniques;
- architecture layers;

to obtain high-quality results under the available inference budget.

Conceptually:

```text
Models
  +
Inference techniques
  +
Budget
      ↓
    itest
      ↓
Optimized architecture
```

---

# Part X — Archon Operations

## 40. Generation

Sampling from a model.

If \(N\) generations are requested, the model produces \(N\) different candidate responses.

---

## 41. Fusion

Given \(K\) candidate responses, an LLM synthesizes one new answer using them.

---

## 42. Critic

Produces strengths/weaknesses for candidate responses.

---

## 43. Ranker

Ranks candidate responses by quality.

---

## 44. Verifier

Provides a correctness/quality signal and can explain its score.

---

## 45. Unit-test generation and evaluation

For coding problems, Archon can use inference-time generation of unit tests.

Example problem:

> Check whether round brackets in a string are balanced.

Possible generated tests include:
- strings with an odd number of brackets should be invalid;
- a closing bracket should match the most recently unmatched opening bracket.

The model can also generate code to implement the tests.

Pipeline:

```text
Candidate solution
       ↓
Generate unit tests
       ↓
Evaluate tests
       ↓
Correctness signal
```

---

# Part XI — Archon Search Space

## 46. Layered architectures

Archon represents inference architectures as layers.

Example:

```text
Layer 1: Generation
Layer 2: Critic
Layer 3: Ranker
Layer 4: Fusion
```

Another architecture could be:

```text
Generation
   ↓
Verifier
   ↓
Fusion
```

The search space becomes very large as the number of operations, models, and layers increases.

---

## 47. Preprocessing / search-space reduction

Searching every possible architecture is expensive because each configuration requires inference calls.

Therefore the lecture describes reducing the search space using prior observations.

Examples:

- first layer is always generation;
- one inference-time technique per layer;
- critic placed before ranker/fuser based on prior local testing;
- unit-test generator followed by evaluator.

These constraints make the architecture search tractable.

---

# Part XII — Bayesian Optimization

## 48. Why Bayesian optimization?

Archon uses a **Bayesian optimizer** to search the architecture space efficiently.

General loop:

```text
Search space
     ↓
Bayesian optimizer
     ↓
Candidate architecture
     ↓
Evaluate
     ↓
Accuracy / cost
     ↓
Use result to choose next candidate
```

The goal is **sample efficiency**:

> Reach a strong architecture after evaluating relatively few configurations.

The lecture states that this performs better than greedy search or random selection in the described setup.

---

## 49. Archon objective

The optimizer can consider different objectives, including:

- accuracy;
- available models;
- inference time;
- inference compute budget.

A simple formulation is:

\[
\max_A Accuracy(A)
\]

subject to:

\[
Cost(A) \leq B
\]

where:
- \(A\) = inference architecture;
- \(B\) = available inference budget.

---

# Part XIII — Depth of Inference Architectures

## 50. Stacking inference layers

The lecture reports an interesting result:

> Stacking more inference-time layers can improve accuracy.

The cited experiments compare:
- using the best model once;
- using the best model multiple times with one fusion layer;
- deeper architectures containing multiple critique/fusion layers.

A deeper architecture can perform significantly better in the cited tasks.

The lecture compares this conceptually to adding layers in deep learning, while emphasizing that these are inference-time layers rather than learned neural-network layers.

---

## 51. Example deep architecture

One cited ensemble contains:

- multiple models;
- multiple critique/fusion layers;
- a final fusion layer.

Conceptually:

```text
Multiple models
      ↓
Generation
      ↓
Critique
      ↓
Rank / Fuse
      ↓
Critique
      ↓
Fuse
      ↓
...
      ↓
Final fusion
      ↓
Final answer
```

The lecture reports that this type of deeper architecture significantly outperformed simpler alternatives in the cited experiments.

---

# Part XIV — Task-Specific vs General-Purpose Archon

## 52. Task-specific optimization

The architecture search can focus on one particular task/benchmark.

```text
Task X
  ↓
Bayesian optimization
  ↓
Architecture optimized for Task X
```

This lets the search exploit task-specific characteristics.

---

## 53. General-purpose optimization

The architecture can instead be optimized to work well across many tasks.

```text
Multiple tasks
     ↓
Architecture search
     ↓
General-purpose architecture
```

The lecture reports that the general-purpose setup also performed strongly across tasks.

---

# Part XV — Reported Archon Results

## 54. Open-source vs closed-source models

The lecture reports that, in the cited experiments, Archon using open-source models could match or exceed contemporary closed-source frontier models across several tasks.

The lecture reports an average **Pass@1 improvement of approximately 14.1%** over GPT-4o or Claude 3.5 Sonnet across the cited instruction-following, reasoning, mathematics, and coding tasks.

This should be understood as the result reported for the specific experiment and setup, not a universal claim about open-source models.

---

# Part XVI — Research Methodology Lessons

## 55. Fix the inference budget

When comparing inference-scaling strategies, control the available compute.

Otherwise a method may appear better simply because it was allowed more computation.

---

## 56. Measure quality and cost

Do not measure only accuracy.

Track:

\[
(	ext{quality}, 	ext{inference cost})
\]

or plot:

\[
	ext{accuracy vs generation budget}
\]

The important object is often the **quality–compute frontier**.

---

## 57. Compare against strong baselines

Useful baselines include:

- single sample;
- random selection;
- majority voting;
- best-of-N;
- model-based ranking;
- Oracle selection/verifier where applicable.

---

## 58. Analyze by difficulty

Average accuracy can hide important behavior.

Use Pass@1 or another measure to divide problems into difficulty groups.

Then ask:

> Does the best inference-scaling strategy change with difficulty?

---

## 59. Evaluate the verifier

Do not automatically treat the verifier as ground truth.

A unit-test suite may be incomplete.

A reward model may be miscalibrated.

A PRM may incorrectly prune a correct trajectory.

Therefore, verification itself needs evaluation.

---

# Part XVII — Failure Modes

## 60. Generation failure

The model almost never generates a correct answer.

More samples may not solve the underlying capability problem.

---

## 61. Verification failure

A correct answer exists, but the system fails to identify it.

This contributes to the generation–verification gap.

---

## 62. Majority-vote failure

Correct answers may be rare and therefore lose to frequent incorrect answers.

---

## 63. Reward-model failure

An ORM can give a high score to an incorrect answer.

---

## 64. PRM misranking

A correct intermediate step can receive a low process score.

---

## 65. Early pruning

A search algorithm can eliminate a branch that would eventually have produced the correct solution.

---

## 66. Incomplete tests

A candidate program may pass all generated tests while still being wrong on unseen cases.

---

## 67. Architecture-search cost

Searching over many architectures itself requires substantial inference computation.

---

# Part XVIII — Research Questions

Lecture 2 naturally suggests:

1. **How can we shrink the generation–verification gap?**
2. How should inference scaling work with imperfect or expensive verifiers?
3. Can ensembles of imperfect verifiers outperform a single verifier?
4. How should compute be allocated according to problem difficulty?
5. When should we use parallel exploration versus sequential refinement?
6. How reliable must a PRM be for PRM-guided search to help?
7. Which search algorithm makes best use of process-level reward signals?
8. When does fusion beat candidate selection?
9. Can automatically searched inference architectures generalize across tasks?
10. How should inference architectures optimize accuracy under strict compute budgets?

---

# Part XIX — Core Mental Model

Think of inference compute as a budget that can be spent in different ways.

### Spend it on breadth
Generate many independent solutions.

### Spend it on depth
Revise/refine solutions.

### Spend it on evaluation
Critique, rank, or verify candidates.

### Spend it on synthesis
Fuse information from multiple candidates.

### Spend it on architecture
Search for the best composition of all these operations.

Therefore:

\[
oxed{
	ext{Inference scaling}
=
	ext{how we allocate and structure inference compute}
}
\]

---

# Part XX — Lecture 2 in One Diagram

```text
                         INFERENCE SCALING
                                |
                +---------------+---------------+
                |                               |
        Repeated Sampling                Structured Scaling
                |                               |
            Coverage                    +-------+-------+
                |                       |       |       |
          Verification                 ORM     PRM   Revision
                |                       |       |       |
    Generation–Verification Gap       Outcome Process  Depth
                |                       |       |       |
        +-------+-------+               +-------+-------+
        |       |       |                       |
     Majority Ranker  Verifier                  |
      Vote                                     Search
        |       |       |                       |
        +-------+-------+-----------------------+
                        |
                      Fusion
                        |
                 Multiple Models
                        |
             Inference Architecture
                        |
                      Archon
                        |
              Bayesian Optimization
                        |
          Accuracy ↔ Inference Cost
```

---

# 21. Most Important Equations

## At least one correct sample

\[
oxed{
P(	ext{at least one correct})=1-(1-p)^N
}
\]

where:
- \(p\) = probability one sample is correct;
- \(N\) = number of samples.

## Benchmark coverage

For \(M\) problems:

\[
oxed{
Coverage =
rac{\#	ext{problems with at least one correct sample}}{M}
}
\]

---

# 22. Terminology Checklist

Be able to define precisely:

- inference scaling;
- test-time scaling;
- repeated sampling;
- coverage;
- Pass@N;
- verification;
- generation–verification gap;
- majority voting;
- reward model;
- Outcome Reward Model (ORM);
- Process Reward Model (PRM);
- parallel sampling;
- sequential revision;
- beam search;
- PRM-guided search;
- fusion;
- critic;
- ranker;
- verifier;
- unit-test generation;
- ensemble;
- inference architecture;
- Archon;
- itest;
- Bayesian optimization;
- inference budget;
- quality–compute frontier;
- task-specific optimization;
- general-purpose optimization.

---

# 23. Final takeaway

The deepest idea in Lecture 2 is not simply:

> "Generate more answers."

It is:

> **An LLM's inference computation can be deliberately allocated, structured, evaluated, searched, and optimized.**

The progression is:

\[
oxed{
	ext{More samples}
ightarrow
	ext{Better verification}
ightarrow
	ext{Better search}
ightarrow
	ext{Better synthesis}
ightarrow
	ext{Better inference architecture}
}
\]

And the central research question is:

\[
oxed{
	extbf{Given limited inference compute, what should we do with it?}
}
\]
