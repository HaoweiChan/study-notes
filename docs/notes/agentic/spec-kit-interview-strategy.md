---
title: "Spec Kit 30-Min Interview Strategy"
date: "2025-12-27"
tags: ["interview-prep", "agentic-workflow", "cursor", "productivity"]
related: []
slug: "spec-kit-interview-strategy"
category: "agentic"
---

# Spec Kit 30-Min Interview Strategy

## Summary
This strategy uses **Spec Kit** within Cursor to demonstrate **Architectural Control over AI** during a 30-minute coding interview. By "programming the AI" via a Constitution and Specs before generating code, you demonstrate the ability to lead an AI workflow rather than just typing code.

## Details

### Phase 1: Pre-Interview Setup (Do this NOW)
You do not want to be installing tools or defining rules during the 30 minutes. You need a "Pre-Loaded" environment.

#### 1. Installation & Initialization
Run these commands in your terminal to set up the toolchain:

```bash
# 1. Install uv (fast Python package manager)
curl -LsSf https://astral.sh/uv/install.sh | sh

# 2. Install Spec Kit CLI
uv tool install specify-cli --from git+https://github.com/github/spec-kit.git

# 3. Initialize your Interview Directory
mkdir interview_prep_demo
cd interview_prep_demo
specify init
# Select "Cursor" as your AI tool when prompted.
# Select "One-time usage" or "Persistent" (Persistent is better if you practice).
```

#### 2. The "Constitution" Hack (Crucial)
Spec Kit relies on a `constitution.md` file to govern how the AI behaves. **Edit this file immediately.** Do not use the default one; it's too generic. Replace the content of `.specify/memory/constitution.md` with an **Interview-Optimized Constitution** (see Examples section).

### Phase 2: The 30-Minute Live Workflow
When the clock starts, do **not** just start coding. Follow this "Speed-Run" protocol.

#### Step 0: The Setup (Min 0-2)
* Open your `interview_prep_demo` folder in Cursor.
* **Paste the Problem:** Paste the interviewer's prompt into a new file `problem.txt` (so it's in context).
* **Open Cursor Chat (Cmd+L / Ctrl+L).**

#### Step 1: Specify (Min 2-5)
* **Command:** `/speckit.specify`
* **Your Prompt:** Keep it brief. Reference the problem file.
    > "Build a \[System\] based on `problem.txt`. Focus on handling \[Key Constraint, e.g., 100ms latency\]. strictly follow the Constitution."
* **The Win:** The AI will generate a `spec.md`. **READ IT.**
* **Verbalize to Interviewer:** "I am generating a spec to ensure we handle the edge cases before we write code. I see the AI captured the latency requirement here..."

#### Step 2: Plan (Min 5-8)
* **Command:** `/speckit.plan`
* **Your Prompt:** (Usually empty, or add a specific tech constraint).
    > "Use \[Library X\] for this."
* **The Win:** It generates a `plan.md`.
* **Control Check:** If the AI suggests a complex database setup, **STOP IT**.
    * *Correction:* "The plan suggests using PostgreSQL. For this 30-min interview, that's overkill. I'll instruct it to use SQLite or In-Memory Dict."
    * *Prompt:* "Refine plan: Use in-memory data structures instead of a DB for speed."

#### Step 3: Implement (Min 8-25)
* **Command:** `/speckit.implement`
* **The Magic:** Cursor will start generating the code files based on the plan.
* **Your Job:** Watch the code generate in real-time.
* **Spot Bugs:** "Wait, line 45 looks like an off-by-one error."
* **Interrupt if needed:** If it goes off rails, hit Stop and refine the task.

#### Step 4: Verify & Polish (Min 25-30)
* **Manual Review:** Run the code.
* **Refactor:** Use Cursor's "Edit" (Cmd+K) to fix small things.
    > "Refactor this function to be O(n) instead of O(n^2)."

### Phase 3: Contingencies (The "Oh Sh*t" Button)

#### Scenario A: The Task is too simple (e.g., "Reverse a string")
* **Do NOT use Spec Kit.** It's too heavy.
* **Fallback:** Just use Cursor Composer (Cmd+I) or standard Chat. "Write a function to reverse string, handle unicode."

#### Scenario B: The AI Hallucinates a Complex Folder Structure
* **Fix:** Immediately use the `constitution` concept to your advantage.
* *Say to Interviewer:* "The agent is trying to over-engineer this. I'm going to override it."
* *Prompt:* "IGNORE previous plan. Rewrite code into a SINGLE `main.py` file. Keep it simple."

#### Scenario C: Playable Ad Task (HTML5)
* Your `constitution.md` should be ready for this.
* **Prompt:** `/speckit.specify Create a single-file HTML5 playable ad. It must include CSS, JS, and HTML in one file. No external assets.`
* (This ensures you don't get broken image links or CORS errors).

## Examples / snippets

### Interview-Optimized Constitution

Paste this into `.specify/memory/constitution.md`:

```markdown
# INTERVIEW CONSTITUTION

1. **Speed & MVP**: Prioritize "Minimum Viable Product". Do not over-engineer. Use the simplest working solution.
2. **Single-File Preference**: Unless the architecture demands otherwise, keep code in one file (or minimal files) for easy review during a 30-min session.
3. **Defensive Coding**: ALWAYS handle edge cases (null inputs, empty lists, API timeouts). Add comments explaining WHY you handled them.
4. **Explainability**: Code must be heavily commented with "Architectural Intent". Explain the Big-O complexity in comments.
5. **Stack**: Use Python (for data/backend) or HTML5/Canvas (for Playable Ads) unless specified otherwise.
6. **No Placeholder**: Never leave "TODO" or "pass". Write working code.
```

## Learning Sources
- [Spec Kit GitHub Repository](https://github.com/github/spec-kit) - Official documentation and source code.
- [Cursor Documentation](https://docs.cursor.com/) - Guide to using Cursor's AI features.
- [System Design Primer](https://github.com/donnemartin/system-design-primer) - Concepts to reference during the "Plan" phase.
