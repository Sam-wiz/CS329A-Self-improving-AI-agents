# CS329A — Self-Improving AI Agents
## Lecture 1 Notes

> **Status:** Lecture 1 completed
> **Focus:** Scaling, reasoning, test-time compute, verifiers, agents, workflows, and self-improvement

## 1. Big Picture

The lecture introduces the progression from large language models toward reasoning systems, agents, and self-improving agents.

```text
Scaling → Reasoning / test-time compute → Sampling + verification
→ Successful behavior → Training → Self-improvement
→ Agents → Tools + memory + environment + feedback
```

The central theme is that capability can improve not only by making models larger, but also by spending more computation at inference time and feeding useful experience back into training.

## 2. Scaling

Three major scaling dimensions:

1. Parameters / model scale
2. Training data
3. Compute

Increasing these resources has historically produced better model performance and lower loss.

Important distinction:

> More training compute can change model parameters, while more inference compute normally does not.

## 3. Zero-Shot and Few-Shot

### Zero-shot

The model receives an instruction without worked examples and must infer the task from the instruction.

### Few-shot

The prompt contains examples demonstrating the desired task. Few-shot does **not** normally modify model parameters; it is an inference-time/context effect.

## 4. Reasoning and Chain-of-Thought

A model can produce intermediate reasoning steps rather than immediately producing a final answer. Reasoning can involve:

- problem analysis
- task decomposition
- self-correction
- alternative approaches
- backtracking
- longer reasoning

The lecture distinguishes historical Chain-of-Thought from modern reasoning models. Original CoT showed that prompting intermediate reasoning could improve performance. Modern reasoning models increasingly have reasoning deliberately trained or reinforced as a behavior.

## 5. Test-Time / Inference-Time Scaling

Instead of changing the model, we can spend more computation during inference:

- longer reasoning
- multiple candidate solutions
- evaluation
- revision
- search

Normally:

\[
\theta_{before}=\theta_{after}
\]

### Repeated sampling

If one attempt has probability \(p\) of being correct, then with \(N\) independent attempts:

\[
P(\text{at least one correct})=1-(1-p)^N
\]

More samples can increase the probability that at least one candidate is correct. But finding a correct candidate is not enough: we need a way to identify it.

## 6. Verifiers

A verifier evaluates candidate solutions and provides a correctness signal.

```text
Problem → Generate candidates → Verifier → Correct / incorrect
```

Verifiers are especially useful for objectively checkable domains such as:

- mathematics
- programming
- unit tests
- compilation
- execution-based evaluation

A verifier is a powerful reward/feedback source, but it is not the only possible RL signal; human preferences and learned reward models are other possibilities.

## 7. Best-of-N vs Self-Improvement

### Best-of-N / inference scaling

```text
Model → N solutions → Verifier → Select best
```

The model remains \(M_\theta\). This improves the current inference result but does not itself improve the model.

### Self-improvement

```text
Generate → Evaluate → Keep successful trajectories
→ Create training data → Fine-tune / train → New model
```

Now:

\[
\theta_{after}\neq\theta_{before}
\]

The model can potentially internalize useful behavior so future attempts have higher success probability. Fine-tuning does not guarantee correctness; generalization still matters.

## 8. LLM → Agent

A normal chatbot can be approximated as:

```text
Prompt → Tokenization → Embeddings → Transformer
→ Logits → Probabilities → Sample → Response
```

An agent introduces a closed-loop interaction:

```text
Goal → Plan → Action → Environment → Observation / feedback
→ Correction → Action → … → Completion
```

The fundamental distinction is that an agent can **act, observe consequences, and use feedback to decide what to do next**. Tools and memory support this process.

## 9. Agent Components

- **Goal:** what the system is trying to accomplish
- **Plan:** strategy or sequence for accomplishing it
- **Action:** operation chosen by the agent
- **Environment:** external system with which it interacts
- **Observation / feedback:** information returned after actions
- **Correction:** changing the plan or next action based on feedback
- **Stopping condition:** determining whether the goal is achieved or cannot be completed
- **Tools:** external capabilities
- **Memory:** information retained or made available across steps/tasks

## 10. Agentic Workflows

The lecture discusses:

- prompt chaining
- routing
- parallelization
- orchestration
- evaluator/judge
- verifiers
- tool calls

### Prompt chaining

```text
LLM 1 → LLM 2 → LLM 3
```

### Routing

```text
Input → Router → Path A / Path B
```

### Parallelization

```text
             → LLM 1 →
Input →      → LLM 2 → Aggregate
             → LLM 3 →
```

### Orchestration

A controller determines which steps or agents should run.

### Evaluator / Judge

A component evaluates an output.

### Verifier

A correctness mechanism, especially useful when correctness is objectively checkable.

## 11. Static Workflows vs Open-Ended Agents

A static workflow has a largely predetermined structure:

```text
A → B → C → D
```

An agent can have a more open-ended loop:

```text
Goal → Choose action → Observe → Decide next action → …
```

The more open-ended the task, the more important planning, feedback, memory, and evaluation become.

## 12. Why Verifiable Domains Matter

Math and programming are attractive environments because they often provide objective feedback:

\[
\text{solution} → \text{verifier} → \{pass, fail\}
\]

For code:

```text
Generated code → Run tests → Pass / Fail
```

This is easier to automate than subjective evaluation such as whether a poem is beautiful or emotionally deep.

## 13. Coding Agents

Coding agents have observable consequences from actions:

```text
Task → Inspect repository → Identify component → Edit code
→ Run tests → Diagnose failure → Fix → Test → Complete
```

They may need to understand the codebase, identify affected files, formulate a plan, make changes, run tests, inspect errors, correct mistakes, and verify the final result.

Important idea:

> Goal ≠ complete procedure.

## 14. Research Agents

A research workflow can be:

\[
\text{Question} → \text{Find references} → \text{Read} → \text{Summarize}
→ \text{Synthesize} → \text{Write}
\]

The LLM becomes a component in a multi-step information workflow rather than only an answer generator.

## 15. AI Scientist / Automated Research

The lecture discusses a more ambitious direction:

```text
Idea generation → Experiment → Analyze results
→ Modify idea/method → Experiment again → Paper writing
```

However:

\[
\text{Idea generation} \neq \text{idea correctness}
\]

Generated ideas can be incorrect, redundant, uninteresting, or based on false assumptions. Therefore research agents need strong evaluation and experimental feedback.

## 16. Reasoning, Self-Correction and Backtracking

Important capabilities include:

- multi-step reasoning
- planning
- self-correction
- alternative proposals
- backtracking

A useful loop:

```text
Problem → Attempt → Evaluate → Failure?
                         ↓
              Alternative / correction
```

Important research principle:

> Observing a capability does not automatically tell us why it exists.

Possible contributing factors include pretraining, scaling, fine-tuning, and RL. Understanding causality requires experiments.

## 17. RL Foundations

### MDP

\[
s_t \rightarrow a_t \rightarrow r_{t+1},s_{t+1}
\]

where \(s\) is state, \(a\) is action, and \(r\) is reward.

### Trajectory

\[
\tau=(s_0,a_0,r_1,s_1,a_1,r_2,\ldots,s_T)
\]

A trajectory is a sequence of states, actions, and rewards.

### Return

\[
G_t=r_{t+1}+\gamma r_{t+2}+\gamma^2r_{t+3}+\cdots
\]

where \(\gamma\) is the discount factor.

## 18. Value, Q-Value and Advantage

### State-value function

\[
\boxed{V^\pi(s)}
\]

**Exact term:** State-value function.

Expected return starting from state \(s\) while following policy \(\pi\):

\[
V^\pi(s)=\mathbb{E}_\pi[G_t\mid S_t=s]
\]

Think: **How good is this situation overall?**

### Action-value function

\[
\boxed{Q^\pi(s,a)}
\]

**Exact term:** Action-value function.

Expected return starting from state \(s\), taking action \(a\), and then following policy \(\pi\):

\[
Q^\pi(s,a)=\mathbb{E}_\pi[G_t\mid S_t=s,A_t=a]
\]

Think: **How good is this particular action from here?**

Do not define Q as "how good the next state is."

### Advantage function

\[
\boxed{A^\pi(s,a)}
\]

**Exact term:** Advantage function.

\[
A^\pi(s,a)=Q^\pi(s,a)-V^\pi(s)
\]

It measures how much better or worse an action is than the policy's baseline expectation at that state.

Example:

\[
V(s)=8,\quad Q(s,a)=12
\]

\[
A(s,a)=4
\]

Positive advantage → increase action probability. Negative advantage → decrease action probability.

## 19. Policy

A policy describes how actions are selected:

\[
\pi(a|s)
\]

For an LLM, it can be viewed as the probability distribution over actions/outputs given the current context/state.

## 20. Delayed Reward and Credit Assignment

Suppose an agent performs 30 actions and receives reward only at the end:

```text
a1 → a2 → … → a30 → +1
```

The key question is:

> Which actions deserve credit for the successful outcome?

This is the **credit assignment problem**.

Reward hacking is separate: it occurs when an agent maximizes the specified reward without achieving the intended objective.

## 21. Policy Gradient Intuition

A policy can be written as:

\[
\pi_\theta(a|s)
\]

Basic intuition:

### Positive advantage

\[
A(s,a)>0 \Rightarrow P(a|s)\uparrow
\]

### Negative advantage

\[
A(s,a)<0 \Rightarrow P(a|s)\downarrow
\]

A simplified policy-gradient update is:

\[
\theta \leftarrow \theta+\alpha A(s,a)\nabla_\theta\log\pi_\theta(a|s)
\]

The intuition matters more initially than memorizing the equation:

> Increase probability of actions that performed better than expected; decrease probability of actions that performed worse than expected.

## 22. Two Loops: Inference vs Learning

### Inference loop

```text
Think → Act → Observe → Correct → Act
```

Model parameters can remain unchanged.

### Learning loop

```text
Trajectories → Evaluate → Train → Better policy/model
```

Now parameters change.

## 23. Research Methodology

A research project should go beyond simply building an agent.

A useful structure is:

\[
\boxed{\text{Research question} → \text{Hypothesis} → \text{Experiment} → \text{Results} → \text{Analysis}}
\]

When evaluating a claimed improvement, investigate:

### Baseline

What happens without the proposed method?

### Fair comparison

Control relevant variables such as:

- model
- compute budget
- sampling budget
- verifier
- prompting
- training data

### Statistical reliability

Determine whether the improvement is robust rather than noise.

### Generalization

Test on unseen problems or distributions.

### Ablation

Remove components to determine what caused the improvement.

### Failure analysis

Study where and why the method fails.

## 24. Core Mental Models

### Inference scaling

\[
\boxed{\text{More compute now} → \text{better current answer}}
\]

No parameter update.

### Self-improvement

\[
\boxed{\text{Experience} → \text{Evaluation} → \text{Training} → \text{Better model}}
\]

Parameters change.

### Agent

\[
\boxed{\text{Goal} → \text{Plan} → \text{Action} → \text{Environment} → \text{Feedback} → \text{Correction}}
\]

Closed-loop interaction.

### RL

\[
\boxed{\text{State} → \text{Action} → \text{Reward} → \text{Policy update}}
\]

### Research

\[
\boxed{\text{Question} → \text{Hypothesis} → \text{Experiment} → \text{Evidence} → \text{Conclusion}}
\]

## 25. High-Priority Terminology

| Term | Precise meaning |
|---|---|
| Zero-shot | Perform a task without examples in the prompt |
| Few-shot | Perform a task using examples provided in the prompt |
| Inference-time scaling | Spend more computation during inference without normally changing parameters |
| Verifier | Mechanism for evaluating candidate outputs/trajectories |
| Best-of-N | Generate N candidates and select among them using evaluation |
| Self-improvement | Use experience/outputs to update the model or policy |
| Agent | System that pursues goals through actions and environment interaction |
| State \(s\) | Current relevant situation/context |
| Action \(a\) | Choice made by the agent |
| Trajectory \(\tau\) | Sequence of states, actions, and rewards |
| Return \(G_t\) | Cumulative future reward |
| Policy \(\pi(a\|s)\) | Strategy/probability distribution for selecting actions |
| State-value \(V(s)\) | Expected return from a state under a policy |
| Action-value \(Q(s,a)\) | Expected return from taking an action in a state and then following a policy |
| Advantage \(A(s,a)\) | \(Q(s,a)-V(s)\); how much better/worse the action is than the baseline |
| Credit assignment | Determining which actions contributed to an eventual reward |
| Reward hacking | Maximizing the specified reward without achieving the intended objective |

## 26. Lecture 1 Takeaway

The central story is:

\[
\boxed{
\text{Scale models}
\rightarrow
\text{reason longer}
\rightarrow
\text{sample more}
\rightarrow
\text{verify}
\rightarrow
\text{learn from successful behavior}
\rightarrow
\text{build agents}
\rightarrow
\text{close the feedback loop}
}
\]

The increasingly important research question is:

> **How can we make a system generate, evaluate, learn from, and improve its own problem-solving behavior?**

This connects Lecture 1 directly to reinforcement learning, verifiers, test-time compute, agentic systems, and self-improving AI.
