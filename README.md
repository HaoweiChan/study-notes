# Study Notes

A structured knowledge base with an interactive web interface. Built for efficient learning, quick reference, and interview preparation.

**Live Demo:** [haoweichan.github.io/study-notes](https://haoweichan.github.io/study-notes/)

## What's Inside

**📚 Knowledge Base** — Notes on Algorithms, System Design, Machine Learning (AdTech focus), and Agentic AI.

**🧠 Active Learning** — 120+ Flashcards and Quizzes with real-time search, category filtering, dark mode, bookmarks, and spaced repetition.

**⚡ Clean Architecture** — Markdown-first notes with auto-extracted flashcards/quizzes, automated linting, and GitHub Pages deployment via React/Vite.

## Repository Structure

```
study-notes/
├── web/                     # React + Vite Web Application
│   ├── src/                 # Frontend source code
│   └── public/              # Static assets
├── docs/                    # Deployment target (GitHub Pages)
│   ├── artifacts/           # Auto-generated JSON data
│   └── notes/               # Copied markdown content
├── notes/                   # Source Markdown notes
│   ├── algorithm/           # Algorithms & data structures
│   ├── devops/              # DevOps & infrastructure
│   ├── system-design/       # System architecture
│   ├── machine-learning/    # ML (AdTech, RecSys)
│   ├── leetcode/            # Coding problem solutions
│   └── agentic/             # AI Agents, RAG, GenAI
├── templates/               # Note templates
├── scripts/                 # Automation scripts
│   ├── new_note.sh          # Create new notes
│   ├── deploy_web.sh        # Build & Deploy script
│   ├── export_notes_index.py # Index notes for WebUI
│   └── ...
└── .github/workflows/       # CI/CD for linting
```

## Usage

### Creating New Notes

Use the provided helper script to create new notes from the template:

```bash
./scripts/new_note.sh "Title of note" <category> [slug]
```

**Available Categories:**
- `algorithm` — Algorithms & data structures
- `devops` — Infrastructure & DevOps
- `system-design` — System architecture
- `machine-learning` — ML & AI
- `leetcode` — Coding problems
- `agentic` — AI agents & automation

**Examples:**
```bash
./scripts/new_note.sh "Binary Search Tree" algorithm
./scripts/new_note.sh "Design Real-Time Bidding" system-design
./scripts/new_note.sh "RAG Architectures" agentic
```

### Adding Content

**Flashcards:**
```markdown
- What is the time complexity of binary search? ::: O(log n)
```

**Quizzes:**
```markdown
## Quizzes
Q: Question?
Options:
- A) Option 1
- B) Option 2
Answers: A
Explanation: ...
```

### Note Format

All notes should follow this structure:
- **Location**: `notes/{category}/short-slug.md`
- **Frontmatter**: Required fields are `title`, `date`, and `category`
- **Sections**: Include `Summary`, `Details`, `Flashcards`, `Quizzes`

### Local Development & Deployment

**1. Install Dependencies**
```bash
# Python dependencies for generation scripts
pip install python-frontmatter

# Web dependencies
cd web
npm install
```

**2. Deploy Locally (Build & Copy)**
```bash
# Builds the React app and generates artifacts into docs/
./scripts/deploy_web.sh
```

**3. Serve Locally**
```bash
# Verify the build
python -m http.server 8000 -d docs
```

### Linting

The repository includes multiple layers of linting:
1. **GitHub Actions**: Run linting on all pull requests and pushes
2. **Manual checking**: Run `npx remark .` and `npx markdownlint "**/*.md"` locally
