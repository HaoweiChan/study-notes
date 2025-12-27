---
title: "DSPy (Programming with LLMs)"
date: "2025-12-27"
category: "agentic"
tags: ["dspy", "prompt-engineering", "llm-ops", "python", "stanford"]
related: ["agent-reasoning-patterns-react-cot", "mcp-model-context-protocol"]
slug: "dspy-programming-llms"
flashcards:
  - q: "What is the core philosophy of DSPy?"
    a: "Separating the flow of the program (logic) from the parameters (prompts/weights), allowing the system to 'compile' and optimize prompts automatically."
  - q: "What is a DSPy 'Signature'?"
    a: "A declarative definition of input/output behavior (e.g., `Question -> Answer`), replacing manual prompt strings."
  - q: "What does a DSPy 'Teleprompter' (Optimizer) do?"
    a: "It takes a program, a training set, and a metric, and optimizes the prompts (e.g., by selecting the best Few-Shot examples) to maximize the metric."
  - q: "Why use DSPy over manual prompting?"
    a: "Manual prompts are brittle and don't transfer across models. DSPy programs are modular and can be re-optimized for different models (e.g., moving from GPT-4 to Llama-3)."
quizzes:
  - q: "In DSPy, if you want to add Chain of Thought reasoning to a step, what do you do?"
    options: ["Write a prompt saying 'Think step by step'", "Change `dspy.Predict` to `dspy.ChainOfThought` in your module", "Manually implement a loop", "Use a larger model"]
    answers: [1]
    explanation: "You simply switch the module class. DSPy handles the underlying prompt structure ('Let's think step by step') automatically."
  - q: "What is 'BootstrapFewShot' in DSPy?"
    a: "A prompting technique",
    options: ["An optimizer that self-generates examples (demonstrations) for the prompt based on teacher inputs", "A way to restart the LLM server", "A specific type of GPU optimization", "A manual list of examples"]
    answers: [0]
    explanation: "It runs the pipeline, keeps the traces that lead to correct answers (based on your metric), and uses them as Few-Shot examples in the final prompt."
---

# DSPy (Programming with LLMs)

## Summary
**DSPy** (Declarative Self-improving Language Programs) is a framework from Stanford that shifts the paradigm from **"Prompt Engineering"** to **"Programming"**. Instead of tweaking string prompts manually, you define the *logic* (Signatures) and *modules*, and DSPy **compiles** the program to optimize the prompts automatically using a dataset and a metric.

## Details

### 1. The Problem with Prompts
*   Prompts are brittle: A prompt that works for GPT-4 might fail for Llama-3.
*   Prompts are opaque: Hard to version control or systematically improve.
*   "Prompt Engineering" is trial-and-error.

### 2. DSPy Core Concepts

#### A. Signatures (The "Type System")
Instead of writing "Please summarize this text...", you define the inputs and outputs:
```python
class Summarize(dspy.Signature):
    """Summarize the text into 3 bullet points."""
    text = dspy.InputField()
    summary = dspy.OutputField()
```

#### B. Modules (The "Layers")
DSPy provides standard layers that use Signatures:
*   `dspy.Predict`: Basic prompt.
*   `dspy.ChainOfThought`: Adds reasoning steps.
*   `dspy.ReAct`: Adds tool use loops.

#### C. Teleprompters (The "Optimizers")
This is the magic. An Optimizer takes:
1.  Your Program (Modules + Signatures).
2.  A Training Set (Inputs + Expected Outputs).
3.  A Metric (e.g., "Is the answer correct?").
It runs the program, tries to improve the "weights" (which in DSPy are **Prompts** and **Few-Shot Examples**), and outputs a compiled program.
*   **BootstrapFewShot:** Finds successful examples in your data and adds them to the prompt.
*   **MIPRO:** Optimizes the instructions themselves.

### 3. The Workflow
1.  **Define**: Write your pipeline using `dspy.Module`.
2.  **Evaluate**: Run it on dev set.
3.  **Compile**: Use `BootstrapFewShot` to optimize.
4.  **Deploy**: Save the compiled program (which contains the optimal few-shot examples).

## Examples / snippets

### A Simple RAG Pipeline in DSPy

```python
import dspy

# 1. Configure LM
turbo = dspy.OpenAI(model='gpt-3.5-turbo')
dspy.settings.configure(lm=turbo)

# 2. Define Signature
class GenerateAnswer(dspy.Signature):
    """Answer questions with short factoid answers."""
    context = dspy.InputField(desc="may contain relevant facts")
    question = dspy.InputField()
    answer = dspy.OutputField(desc="often between 1 and 5 words")

# 3. Define Module
class RAG(dspy.Module):
    def __init__(self, num_passages=3):
        super().__init__()
        self.retrieve = dspy.Retrieve(k=num_passages)
        self.generate_answer = dspy.ChainOfThought(GenerateAnswer)
    
    def forward(self, question):
        context = self.retrieve(question).passages
        prediction = self.generate_answer(context=context, question=question)
        return dspy.Prediction(context=context, answer=prediction.answer)

# 4. Compile (Optimize)
from dspy.teleprompt import BootstrapFewShot

# We need a metric (e.g., Exact Match)
def validate_answer(example, pred, trace=None):
    return example.answer.lower() == pred.answer.lower()

# Optimize!
teleprompter = BootstrapFewShot(metric=validate_answer)
compiled_rag = teleprompter.compile(RAG(), trainset=my_dataset)

# 5. Run
compiled_rag("What is the capital of France?")
```

## Learning Sources
- [DSPy GitHub Repository](https://github.com/stanfordnlp/dspy) - Official source.
- [DSPy: Compiling Declarative Language Model Calls (Paper)](https://arxiv.org/abs/2310.03714) - The academic paper.
- [Omar Khattab's Twitter/X](https://twitter.com/lateinteraction) - Creator of DSPy (and ColBERT), posts updates.
