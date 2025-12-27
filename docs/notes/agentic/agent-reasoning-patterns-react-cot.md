---
title: "Agent Reasoning Patterns (ReAct, CoT)"
date: "2025-12-27"
tags: ["agents", "genai", "prompt-engineering"]
related: []
slug: "agent-reasoning-patterns-react-cot"
category: "agentic"
---

# Agent Reasoning Patterns (ReAct, CoT)

## Summary
LLMs are not just text generators; they can be reasoning engines. Techniques like **Chain of Thought (CoT)** improve reasoning by breaking down problems, while **ReAct (Reason + Act)** enables agents to interact with the world (using tools) to solve complex tasks.

## Details

### 1. Chain of Thought (CoT)
- **Concept**: Prompting the model to "think step by step" before giving an answer.
- **Mechanism**: The intermediate generation acts as a "scratchpad," allowing the model to compute intermediate results and reducing logical errors.
- **Zero-Shot CoT**: Just adding "Let's think step by step."
- **Few-Shot CoT**: Providing examples of (Question -> Reasoning -> Answer).

### 2. ReAct (Reason + Act)
- **Concept**: Interleaves reasoning (Thinking) with action execution (Acting).
- **Loop**:
    1. **Thought**: "The user wants to know the weather in Tokyo. I should check the weather API."
    2. **Action**: `SearchTool.run("weather Tokyo")`
    3. **Observation**: "Tokyo is 20°C and rainy." (Output from tool).
    4. **Thought**: "I have the answer. I will reply to the user."
    5. **Final Answer**: "It is currently rainy and 20°C in Tokyo."
- **Significance**: Without ReAct, models hallucinate facts. With ReAct, they ground answers in external data.

### 3. Other Patterns
- **Plan-and-Solve**: The agent first generates a full plan (Step 1, Step 2, Step 3) and then executes it. Good for complex, multi-step tasks.
- **Reflection / Self-Correction**: The agent reviews its own output ("Does this code actually solve the problem?") and generates a critique, then regenerates the solution.

## Examples / snippets

### ReAct Prompt Template (Simplified)

```text
Answer the following questions as best you can. You have access to the following tools:

[Search]: Useful for questions about current events.
[Calculator]: Useful for math.

Use the following format:

Question: the input question you must answer
Thought: you should always think about what to do
Action: the action to take, should be one of [Search, Calculator]
Action Input: the input to the action
Observation: the result of the action
... (this Thought/Action/Observation can repeat N times)
Thought: I now know the final answer
Final Answer: the final answer to the original input question

Begin!

Question: Who is the wife of the actor who played 'Neo' in Matrix?
Thought: I need to find out who played Neo, then find his wife.
Action: Search
Action Input: "actor played Neo in Matrix"
Observation: Keanu Reeves played Neo.
Thought: Now I need to find Keanu Reeves' wife.
Action: Search
Action Input: "Keanu Reeves wife"
Observation: Keanu Reeves is not married.
Thought: I have the answer.
Final Answer: Keanu Reeves is not married.
```

## Flashcards

- What is the core idea of Chain of Thought (CoT)? ::: It encourages the LLM to generate **intermediate reasoning steps** before the final answer, which improves performance on complex tasks.
- What does ReAct stand for? ::: **Re**asoning + **Act**ing.
- How does ReAct differ from standard CoT? ::: CoT is for internal reasoning only. ReAct allows the model to **take actions** (call tools) and use the **observations** from those actions to inform the next reasoning step.
- What is "Zero-Shot CoT"? ::: Triggering reasoning capabilities simply by appending the phrase "**Let's think step by step**" to the prompt, without providing examples.
- What is the "Observation" step in the ReAct loop? ::: It is the output/result returned by an **external tool** (e.g., API response) that the model reads to update its context.

## Quizzes

### Pattern Selection
Q: You are building an agent to book flight tickets. This requires checking availability, comparing prices, and then booking. Which pattern is most appropriate?
Options:
- A) Zero-Shot Classification
- B) Chain of Thought (CoT)
- C) ReAct (Reason + Act)
- D) Sentiment Analysis
Answers: C
Explanation: The task requires interacting with external systems (Flight API) and making decisions based on the results (if price > X, check another flight). ReAct is designed exactly for this loop of Reasoning -> Tool Use -> Observation.

### Limitations
Q: What is a common failure mode of ReAct agents?
Options:
- A) They never hallucinate.
- B) Getting stuck in a loop (Thought -> Action -> Same Observation -> Thought -> Same Action...).
- C) They cannot perform math.
- D) They work too fast.
Answers: B
Explanation: ReAct agents can get stuck in infinite loops if the Action doesn't yield new info or if the model fails to recognize it has enough info. Limits on "max steps" are usually required.

## Learning Sources
- [ReAct Paper (ICLR 2023)](https://arxiv.org/abs/2210.03629) - The original paper "ReAct: Synergizing Reasoning and Acting in Language Models".
- [Chain-of-Thought Prompting Elicits Reasoning (NeurIPS 2022)](https://arxiv.org/abs/2201.11903) - The seminal CoT paper.
- [LangChain Agents Documentation](https://python.langchain.com/docs/modules/agents/) - How to implement ReAct in code.
- [Prompt Engineering Guide: ReAct](https://www.promptingguide.ai/techniques/react) - Detailed tutorial.
