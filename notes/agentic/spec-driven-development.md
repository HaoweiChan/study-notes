---
title: "Spec-Driven Development (SDD)"
date: "2025-12-27"
category: "agentic"
tags: ["sdd", "spec-kit", "agentic-workflow", "ai-coding"]
related: ["spec-kit-interview-strategy", "agent-reasoning-patterns-react-cot"]
slug: "spec-driven-development"
flashcards:
  - q: "What is the core philosophy of Spec-Driven Development (SDD)?"
    a: "Specifications don't serve code; code serves specifications. The spec is the source of truth that generates the implementation."
  - q: "What is the 'Library-First Principle' in SDD?"
    a: "Every feature must begin as a standalone library with clear boundaries, ensuring modularity and reusability."
  - q: "What is the purpose of the `/speckit.specify` command?"
    a: "To transform a vague idea into a structured feature specification with user stories and acceptance criteria."
  - q: "What are the 'Phase -1 Gates' in the SDD planning process?"
    a: "Checks for simplicity (e.g., max 3 projects) and anti-abstraction (using frameworks directly) to prevent over-engineering."
  - q: "What is the 'Test-First Imperative' in the SDD Constitution?"
    a: "No implementation code is written before unit tests are written, approved, and confirmed to fail (Red phase)."
quizzes:
  - q: "In the SDD workflow, which command is responsible for converting a high-level plan into executable steps?"
    options: ["/speckit.specify", "/speckit.plan", "/speckit.tasks", "/speckit.generate"]
    answers: [2]
    explanation: "/speckit.tasks analyzes the plan and design documents to generate a specific, parallelizable task list."
  - q: "How does SDD handle 'What-If' scenarios or pivots?"
    options: ["Manually refactoring the code base", "Updating the specification and regenerating the implementation plan", "Creating a new git branch and rewriting code", "Ignoring the change until the next release"]
    answers: [1]
    explanation: "Because code is a generated artifact, changes are made to the specification, and the implementation plan and code are systematically regenerated."
---

# Spec-Driven Development (SDD)

## Summary
Spec-Driven Development (SDD) is a methodology where **Specifications** are the primary source of truth, and **Code** is a generated artifact. By using AI agents to strictly follow a "Constitution" and structured templates, SDD ensures that software is built with architectural discipline, modularity (Library-First), and testability (Test-First) from the ground up.

## Details

### 1. The Power Inversion
Traditionally, documents (PRDs, specs) become stale as soon as coding begins. In SDD:
*   **Code serves Specifications:** The Spec generates the code.
*   **Gap Elimination:** There is no gap between intent and implementation because the translation is automated and constrained by AI.
*   **Iterative Regeneration:** Refactoring means updating the Spec and regenerating the Plan/Code.

### 2. The Workflow (Spec Kit)

#### Step 1: Specify (`/speckit.specify`)
*   **Input:** Natural language idea (e.g., "Real-time chat system").
*   **Output:** `spec.md` with structured requirements, User Stories, and Acceptance Criteria.
*   **Key Constraint:** Focus on **WHAT**, not HOW.

#### Step 2: Plan (`/speckit.plan`)
*   **Input:** The `spec.md`.
*   **Output:** `plan.md` (Architecture, Tech Stack), `contracts/` (APIs), `data-model.md`.
*   **Gates:** Enforces "Simplicity Gate" (e.g., don't use Kubernetes for a ToDo app) and "Anti-Abstraction Gate".

#### Step 3: Task (`/speckit.tasks`)
*   **Input:** `plan.md`.
*   **Output:** `tasks.md`.
*   **Function:** Breaks down the plan into parallelizable, executable steps for a coding agent.

### 3. The Constitution (Architectural DNA)
The system is governed by a `constitution.md` that defines immutable principles:
*   **Article I: Library-First:** Every feature is a standalone library. No monolithic glue code.
*   **Article II: CLI Interface:** Every library must expose functionality via CLI (stdin/stdout) for testability.
*   **Article III: Test-First:** Tests must be written and fail **before** implementation code is generated.
*   **Article VII & VIII:** Simplicity & Anti-Abstraction. Use frameworks directly; avoid wrapper hell.
*   **Article IX: Integration-First:** Prefer real environments (e.g., real DB) over mocks for integration tests.

## Examples / snippets

### SDD Command Flow

```bash
# 1. Create a Feature Spec
# Generates: specs/003-chat/spec.md
/speckit.specify "Real-time chat with WebSocket and Redis"

# 2. Generate Implementation Plan
# Generates: specs/003-chat/plan.md, data-model.md, contracts/
/speckit.plan "Use FastAPI for backend and React for frontend"

# 3. Create Executable Tasks
# Generates: specs/003-chat/tasks.md
/speckit.tasks

# 4. Agent Execution
# The agent reads tasks.md and implements them one by one.
```

### Example Constitution Check (Pre-Implementation Gate)

The AI must check these boxes before writing code:

```markdown
### Phase -1: Pre-Implementation Gates

#### Simplicity Gate
- [x] Using ≤3 projects?
- [x] No future-proofing? (YAGNI)

#### Anti-Abstraction Gate
- [x] Using framework directly? (No "MyCustomWrapper")
- [x] Single model representation?

#### Integration-First Gate
- [x] Contracts defined?
- [x] Contract tests written?
```

## Learning Sources
- [Spec-Driven Development Manifesto](https://github.com/github/spec-kit/blob/main/spec-driven.md) - The official philosophy and guide.
- [Spec Kit Repository](https://github.com/github/spec-kit) - Tools to implement SDD.
