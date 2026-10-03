# CS329A — Lecture 5: Planning and Multi-Step Reasoning

> Study notes based on the Lecture 5 transcript and our interactive discussion.
>
> Core theme: how agents can reason, plan, search, execute, and learn on multi-step tasks while using compute efficiently.

## 1. Lecture Overview

Three papers:

1. **Language Agent Tree Search (LATS)** — MCTS-style search/planning for language agents.
2. **SPRINT** — parallel planning/execution using dependencies between reasoning steps.
3. **SWiRL** — training multi-step tool-using agents from offline synthetic trajectories and step/outcome feedback.

Unifying question:

> How should an agent allocate computation across alternative reasoning paths, independent work, and feedback so that additional compute actually improves task success?

---

# 2. Why Multi-Step Reasoning Is Different

A multi-step task may look like:

```text
Goal → Reason → Action/Tool → Observation → Reason → Action → … → Result
```

Later decisions depend on earlier decisions and observations. A poor early choice can send the trajectory into a bad region of the search space.

This motivates:
- alternative trajectories
- planning/search
- environment feedback
- verification
- adaptive compute allocation

---

# 3. Paper 1 — Language Agent Tree Search (LATS)

## 3.1 Core idea

LATS brings ideas from **Monte Carlo Tree Search (MCTS)** and multi-step RL into language-model agents.

Instead of committing to one trajectory, the agent maintains and explores multiple possible trajectories and allocates more computation to promising branches.

It combines:
- reasoning
- acting
- planning
- trajectory search
- environment feedback
- reflection

## 3.2 Why a tree?

A single trajectory:

```text
S0 → A → B → C → Result
```

can fail because of an early decision.

A search tree allows alternatives:

```text
             S0
          /   |   \\
         A    B    C
        / \\   |   / \\
      A1  A2 B1 C1  C2
```

This creates a **breadth vs depth** tradeoff:
- breadth = explore alternatives
- depth = develop promising alternatives

---

# 4. LATS and MCTS

LATS uses a search procedure inspired by MCTS.

The lecture describes six stages:

1. Selection
2. Expansion
3. Evaluation
4. Simulation
5. Backpropagation
6. Reflection

## 4.1 Selection

Choose which existing node/branch to explore next.

The search balances:
- **exploitation** — use branches that currently look good
- **exploration** — investigate branches that have not been explored much

## 4.2 UCT

One exploration/exploitation mechanism is **UCT (Upper Confidence Bound applied to Trees)**.

Conceptually:

\[
UCT(s,a)=\text{exploitation}+\text{exploration}
\]

A common form is:

\[
V(s,a)+c\sqrt{\frac{\ln N(s)}{N(s,a)}}
\]

where:
- `V(s,a)` = current search value estimate
- `N(s)` = visits to parent/state
- `N(s,a)` = visits to action/child
- `c` = exploration coefficient

A low-visit branch can receive exploration pressure even if its current value is lower.

### Important RL terminology

Do **not** automatically identify a LATS tree value with the standard RL \(V^\pi(s)\). LATS uses search-tree statistics/value estimates accumulated during search; they are conceptually related to value estimation but are not necessarily the same mathematical object.

---

# 5. LATS — Expansion

After selecting a node, sample possible actions/continuations.

Example:

```text
Current state
   ↓
Sample actions
   ├── Search flights
   ├── Search hotels
   └── Search recommendations
```

Each action can produce a new state/observation and new tree branches.

---

# 6. LATS — Evaluation

Candidate states/trajectories are evaluated.

The lecture describes:

### LM-as-a-Judge

The model evaluates a state/trajectory with a score, e.g. 0–1.

### Self-consistency

Multiple actions/continuations are sampled and their frequency/agreement can be used as an additional signal.

Key distinction:

> Evaluation provides a value signal; MCTS uses that signal to decide where to search next.

A better evaluator does not automatically mean a better search algorithm.

---

# 7. LATS — Simulation

After selecting and expanding a promising node, the system can continue the trajectory, often greedily, until:
- success
- failure
- or the search budget is exhausted.

The purpose is to discover downstream consequences of the branch.

---

# 8. LATS — Backpropagation

If a rollout produces:

```text
S0 → A → B → C → SUCCESS
```

information about the outcome is propagated back through the visited nodes.

```text
C ← outcome
↑
B
↑
A
↑
S0
```

This updates search statistics and influences future selection.

### Important distinction

**MCTS backpropagation ≠ PRM.**

- MCTS backpropagation: updates search-tree information from rollout outcomes.
- PRM: a learned process-level evaluator that scores intermediate reasoning.

They can be combined, but they are different mechanisms.

---

# 9. LATS — Reflection

After a trajectory, the model can reflect on:
- why it succeeded
- why it failed
- what could be changed
- what should be tried next

Reflection provides an additional signal for future search.

Backpropagation propagates outcome/value information; reflection can provide a more explicit interpretation of what happened.

---

# 10. LATS Example — Maze

Initial state: dimly lit room.

Possible actions:
- go left
- go right
- inspect room

LATS can:
1. sample actions
2. execute them
3. observe resulting states
4. evaluate states
5. expand promising branches
6. simulate deeper
7. backpropagate outcomes
8. reflect
9. search again

---

# 11. LATS vs Math-Shepherd

**Math-Shepherd:** focuses on verification/scoring of reasoning trajectories, particularly process-level reasoning.

**LATS:** uses environment outcomes, observations, reflection, and search to decide how to explore trajectories.

Useful distinction:

> Math-Shepherd is primarily about process verification; LATS uses feedback as part of active search.

---

# 12. LATS Evaluation and Limitations

The lecture discusses evaluation on:
- **HotPotQA** — multi-hop questions requiring multiple Wikipedia pages.
- **WebShop** — web-based interaction.

The lecture reports improved performance with more trajectories/test-time compute and discusses strong WebShop performance.

### Limitations

1. Tree expansion and repeated rollouts are expensive.
2. Cost-benefit of additional search compute needs analysis.
3. Irreversible actions are difficult: e.g. payments or transactions.
4. UCT is only one exploration/exploitation method; other bandit/search algorithms could be investigated.

---

# 13. LATS — Breadth vs Depth

### Pure depth

```text
S → A → A1 → A2 → A3 → … → FAIL
```

Problem: little breadth; an early mistake can consume large amounts of compute.

### Pure breadth

```text
S → A
S → B
S → C
S → D
```

Problem: many possibilities but little refinement.

### Adaptive search

```text
Explore → identify promising branches → deepen → re-evaluate → allocate more compute
```

This balances breadth and depth.

---

# 14. LATS Search Failure Modes

## Premature exploitation

The search repeatedly selects a high-value branch and neglects alternatives.

More precise terms:
- premature exploitation
- insufficient exploration
- search bias

This is not the same as ordinary model overfitting.

## Bad evaluation signal

Even a good search algorithm can fail if the evaluator is wrong.

Example:

```text
Estimated: A=.95, B=.60, C=.55
Actual:    A=.30, B=.60, C=.80
```

So:

\[
\text{Search failure}
=\text{bad exploration policy}+\text{bad evaluation/value signal}
\]

---

# 15. Paper 2 — SPRINT

## 15.1 Core idea

SPRINT focuses on **parallel planning and execution**.

Reasoning models may spend many steps on difficult tasks, but not every step depends on every previous step.

Therefore:

> Independent reasoning work can potentially be executed in parallel.

---

# 16. Sequential vs Parallel Reasoning

Sequential:

```text
A → B → C → D → E
```

Parallel where possible:

```text
A ──┐
B ──┼──→ D → E
C ──┘
```

The key concept is **dependency**, not whether tasks merely look separate.

---

# 17. Dependency DAG

Reasoning can be represented as:

\[
G=(V,E)
\]

where:
- \(V\) = reasoning/execution steps
- \(E\) = dependency relationships

Example:

```text
Step 1 ─────┐
Step 2 ─────┼──→ Step 4 ──→ Step 5
Step 3 ─────┘
```

Steps 1, 2, and 3 can potentially execute concurrently.

Step 4 depends on earlier outputs.

Step 5 depends on Step 4.

Important nuance: parallelization disappears only for the affected dependency chain, not necessarily for the entire problem.

---

# 18. SPRINT's Main Insight

Instead of:

\[
1\rightarrow2\rightarrow3\rightarrow4\rightarrow5
\]

SPRINT tries to expose:

\[
\{1,2,3\}\rightarrow4\rightarrow5
\]

The transformer architecture does not need to change. The underlying model remains a next-token prediction model; SPRINT changes how reasoning/planning blocks are organized and executed.

---

# 19. SPRINT and Reasoning Traces

The lecture describes using existing reasoning traces and an LLM to:
- identify plans
- distinguish planning and execution
- identify dependencies
- construct a DAG
- identify independently executable steps

Independent steps can then be packed for parallel execution.

The lecture discusses SFT using reformatted trajectories.

---

# 20. SPRINT Training Setup

The lecture describes:
- generating around **6k MATH trajectories**
- selecting trajectories with high parallelization potential
- reformulating trajectories
- SFT training a DeepSeek-R1 Distill-Qwen-7B model

The lecture reports accuracy/efficiency improvements over the relevant baseline and out-of-domain generalization to tasks including Countdown and GPQA Diamond.

---

# 21. SPRINT — Parallelism Over Time

The lecture observes that:

> Early reasoning can have more parallelism, while later reasoning tends to converge toward fewer/deeper plans.

### Early

Information gathering, alternative approaches, independent calculations, and evaluations can often be parallel.

### Later

The agent combines results, refines a candidate, verifies conclusions, and produces a final answer. Dependencies increase.

Conceptually:

```text
Early                    Later

A ──┐
B ──┼──→ D ──→ E ──→ Answer
C ──┘
```

High width early → higher dependency/depth later.

---

# 22. SPRINT — Parallelism Has Overhead

Parallelization costs include:
- dependency estimation
- plan generation
- scheduling
- synchronization
- result aggregation

For easy tasks:

\[
C_{overhead}>C_{savings}
\]

For complex tasks with substantial independent work:

\[
C_{savings}\gg C_{overhead}
\]

Therefore:

> Parallelism should be adaptive and task-dependent, not maximized blindly.

---

# 23. Dependency Estimation — Research Direction

A possible extension is to estimate:

\[
D(i,j)=P(\text{step }i\text{ depends on step }j)
\]

rather than treating dependency as perfectly known.

Example:

\[
D(i,j)=0.92
\]

Possible policy:
- high confidence → enforce sequential dependency
- low confidence → parallelize
- medium confidence → speculative execution
- dependency discovered → replan affected branch

Research question:

> Can uncertainty-aware dependency estimation improve parallel reasoning efficiency without hurting accuracy?

---

# 24. SPRINT Failure Mode

Incorrect dependency assumptions can cause:
- invalid parallel execution
- stale inputs
- wasted computation
- synchronization problems
- incorrect reasoning

A robust pipeline could be:

```text
Planning
   ↓
Dependency estimation
   ↓
DAG construction
   ↓
Parallel execution
   ↓
Dependency violation?
   ├── No → continue
   └── Yes → pause/replan affected branch
```

---

# 25. LATS vs SPRINT

| LATS | SPRINT |
|---|---|
| Search over possible trajectories | Schedule independent reasoning |
| MCTS / UCT | Dependency/DAG structure |
| Breadth + depth | Parallelism + sequential dependencies |
| Adaptive search allocation | Adaptive execution |
| Focuses on **where** compute goes | Focuses on **when/how** work executes |

Common question:

\[
\boxed{\text{How can we get more useful reasoning per unit of inference cost?}}
\]

---

# 26. Paper 3 — SWiRL

## 26.1 Core problem

SWiRL focuses on **multi-step reasoning and tool use**.

Typical agent loop:

```text
Problem
  ↓
Reason
  ↓
Tool call
  ↓
Observation
  ↓
Reason
  ↓
Tool call
  ↓
Observation
  ↓
Final answer
```

Challenges:
1. errors compound across steps
2. live tool interactions can be expensive during training
3. tools/environments can be slow or unreliable
4. final-answer-only reward is sparse
5. the agent must learn when/how to call tools
6. the agent should recover from errors and know when to stop

---

# 27. Error Compounding

Example:

```text
Step 1 → small error
Step 2 → uses incorrect result
Step 3 → builds on Step 2
Step 4 → final failure
```

The policy needs to learn to:
- choose actions
- form tool queries
- interpret observations
- recover from mistakes
- maintain accuracy across steps

---

# 28. Why Live Tools Are Difficult During Training

Live RL looks like:

```text
Policy → Tool → Real environment → Observation → Policy update
```

This can be expensive, slow, and unreliable.

SWiRL instead uses **offline synthetic multi-step trajectories**.

---

# 29. Offline Synthetic Trajectories

At each step the model can:
- reason
- call a tool
- receive an environment response
- continue reasoning
- produce a final answer

Different action/tool choices create different trajectories.

```text
Problem
  ├── reason → tool A → observation → answer
  ├── reason → tool B → observation → answer
  └── reason → tool A → tool C → observation → answer
```

Main benefit:

> Many experiences can be generated/evaluated without requiring every RL update to interact with expensive live tools.

---

# 30. LLM-as-a-Judge in SWiRL

An LLM judge can assign quality/reward at each step.

Example:

```text
Step 1 → 0.9
Step 2 → 0.8
Step 3 → 0.2
Step 4 → 0.3
Final  → 0.1
```

Compared with only:

\[
R_{final}=0.1
\]

step-level scores provide more local information about where the trajectory degraded.

---

# 31. Step-Level vs Outcome Filtering

## Step-level filtering

Keep trajectories where every step is judged good.

```text
✓ → ✓ → ✓ → ✓ → final ✓
```

Pros:
- cleaner process signal
- avoids retaining poor intermediate behavior

Cons:
- can discard successful trajectories
- can lose recovery examples
- produces less data diversity

## Outcome filtering

Keep trajectories whose final answer is correct regardless of intermediate step labels.

```text
✗ → ✗ → ✓ → ✓ → final ✓
```

Pros:
- retains more successful trajectories
- more diverse training signal
- includes successful recovery trajectories

Cons:
- intermediate behavior may be noisy
- can reinforce undesirable behavior if used blindly

---

# 32. Process Quality vs Outcome Quality

Core tradeoff:

\[
\boxed{\text{Process quality} \quad vs. \quad \text{Outcome quality}}
\]

Step-level filtering asks:

> Is the trajectory consistently good?

Outcome filtering asks:

> Did the trajectory ultimately succeed?

Neither should automatically be assumed universally superior.

A useful experiment compares:
1. step-filtered training
2. outcome-filtered training
3. mixed training

Evaluate:
- final accuracy
- tool-use accuracy
- error recovery
- number of steps
- inference cost

---

# 33. SWiRL and RL

High-level pipeline:

\[
\text{Offline trajectory generation}
\rightarrow
\text{Step/outcome judging}
\rightarrow
\text{Filtering}
\rightarrow
\text{RL training}
\]

### Inference-time feedback

```text
Bad observation
   ↓
Reason again
   ↓
Different action
```

### Training-time feedback

```text
Bad step reward
   ↓
Learning signal
   ↓
Future policy changes
```

The training process does not necessarily fix the same trajectory immediately; the reward becomes a signal for learning better future behavior.

---

# 34. Why Per-Step Feedback Matters

Final reward only says whether the overall trajectory succeeded.

Per-step feedback provides:

\[
r_1,r_2,r_3,\ldots,r_T
\]

This can provide more useful local information for learning over a long trajectory.

Important:

> Per-step feedback is not automatically the same thing as a PRM.

A PRM is a learned process-level evaluator. A judged step reward is a training signal/evaluation signal; the two concepts should not be conflated.

---

# 35. Unified View

## LATS

> **Where should inference compute go?**

Mechanisms:
- search
- MCTS
- UCT
- exploration/exploitation
- rollout
- backpropagation
- reflection

## SPRINT

> **Which reasoning work can happen simultaneously?**

Mechanisms:
- planning
- dependency detection
- DAG construction
- parallel execution
- sequential execution where dependencies exist

## SWiRL

> **How can we train a multi-step policy to make better decisions and use tools reliably?**

Mechanisms:
- offline trajectories
- tool-use experiences
- step/outcome feedback
- filtering
- RL training

---

# 36. Lecture 5 Progression

A useful mental model:

\[
\boxed{\text{Search}\rightarrow\text{Efficient execution}\rightarrow\text{Learning better policies}}
\]

- LATS → search better
- SPRINT → execute better
- SWiRL → learn better

---

# 37. Adaptive Compute — Unifying Research Theme

A major research direction:

> How do we maximize useful progress per unit of inference/training compute?

For search:

> Where should the next compute unit go?

For parallelism:

> Which work can run concurrently?

For feedback:

> Which feedback source gives the highest information value per cost?

A useful conceptual objective:

\[
\max\frac{\Delta P(\text{success})}{\Delta\text{inference cost}}
\]

---

# 38. Combining LATS + SPRINT + SWiRL

Possible architecture:

```text
                    Problem
                       ↓
              Difficulty / uncertainty
                       ↓
             ┌─────────┴─────────┐
             ↓                   ↓
          Simple               Hard
             ↓                   ↓
      Direct reasoning         LATS
                                 ↓
                         Candidate branches
                                 ↓
                       Dependency analysis
                                 ↓
                    ┌────────────┴────────────┐
                    ↓                         ↓
               Independent                 Dependent
                    ↓                         ↓
                 SPRINT                  Sequential
                    └────────────┬────────────┘
                                 ↓
                              Verify
                                 ↓
                       Tool / environment
                       or LLM-as-a-Judge
                                 ↓
                               Result
```

Training could use SWiRL-style offline trajectories and feedback to improve the underlying policy.

---

# 39. Research Directions Identified

## 39.1 Adaptive rollout allocation

Under a fixed inference budget:
1. explore several trajectories
2. estimate success probability and uncertainty
3. allocate more compute to promising/high-value trajectories
4. preserve some exploration
5. stop when additional compute has low expected value

Conceptually:

\[
a^*=\arg\max_a
\frac{\mathbb{E}[\Delta P(\text{correct}\mid s,a)]}{C(a)}
\]

## 39.2 Uncertainty-aware dependency scheduling

Estimate:

\[
P(\text{dependency})
\]

and use uncertainty to decide whether to serialize, parallelize, speculate, or replan.

## 39.3 Search + parallelism

Combine LATS for promising branches with SPRINT for independent work.

Research question:

> Can search and parallel execution jointly reduce wall-clock inference cost while maintaining or improving accuracy?

## 39.4 Search + learned feedback

Research question:

> How should search allocate compute when the verifier itself is uncertain or systematically biased?

## 39.5 Adaptive feedback selection

Possible policy:

```text
Objective environment available?
        │
       yes → execution / verifier
        │
       no
        ↓
Can outcome be objectively evaluated?
        │
       yes → outcome verifier
        │
       no
        ↓
LLM-as-a-Judge
```

Research objective:

\[
\max\frac{\text{information gained}}{\text{feedback cost}}
\]

---

# 40. Important Terminology Distinctions

### MCTS backpropagation ≠ PRM

Backpropagation updates search-tree statistics. A PRM is a learned process-level evaluator.

### Search value ≠ automatically \(V^\pi(s)\)

LATS search values are search estimates/statistics and should not automatically be called the standard RL policy value function.

### Process feedback ≠ credit assignment

Knowing a trajectory succeeded is different from determining exactly which action caused the success.

### Parallel ≠ independent forever

Two steps may initially appear independent but later reveal dependencies. Robust systems need dependency detection/replanning.

### Premature exploitation ≠ ordinary overfitting

Repeatedly following explored branches while neglecting alternatives is more precisely insufficient exploration/premature exploitation.

---

# 41. Final Mental Model

### LATS

> **Where should I search?**

### SPRINT

> **What can I execute in parallel?**

### SWiRL

> **How can I learn a better multi-step policy from these experiences?**

### Lecture-wide research question

> **How can an AI agent spend limited compute where it creates the most probability of solving the task?**

---

# 42. Interactive Session Self-Assessment

Approximate understanding from the lecture discussion:

- **LATS:** ~9/10
- **SPRINT:** ~9.5/10
- **SWiRL:** ~8/10
- **Research thinking:** strong
- **Main remaining gap:** precise separation of search statistics, reward signals, process evaluators, and RL policy learning.
