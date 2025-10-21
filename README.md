# Study Notes

A structured knowledge base with an interactive flashcard interface. Built for efficient learning and quick reference.

## What's Inside

**Flashcard Study System** — 120+ cards with real-time search, category filtering, dark mode, bookmarks, and spaced repetition.

**Clean Architecture** — Markdown-first notes with auto-extracted flashcards, automated linting, and GitHub Pages deployment.

## Repository Structure

```
study-notes/
├── docs/
│   ├── index.html           # Flashcard web interface
│   ├── styles.css           # Design system from Figma
│   └── artifacts/
│       └── flashcards.json  # Auto-generated flashcard data
├── notes/
│   ├── algorithm/           # Algorithm & data structure notes
│   ├── devops/              # DevOps & infrastructure notes
│   ├── system-design/       # System architecture notes
│   ├── machine-learning/    # ML notes
│   ├── leetcode/            # Coding problem solutions
│   └── agentic/             # AI agent & automation notes
├── templates/
│   ├── default-template.md
│   ├── leetcode-template.md
│   ├── system-design-template.md
│   ├── devops-template.md
│   └── agentic-template.md
├── scripts/
│   ├── new_note.sh          # Create new notes
│   ├── export_flashcards_json.py  # Extract flashcards
│   └── categories/          # Category-specific scripts
└── .github/workflows/
    └── lint.yml             # CI/CD for linting
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
./scripts/new_note.sh "Docker Networking" devops
./scripts/new_note.sh "Design Twitter" system-design
./scripts/new_note.sh "Two Sum" leetcode
```

### Study Flashcards

🎯 **Interactive Flashcard Study System**

Access the flashcard study interface at: **[Study Flashcards](https://haoweichan.github.io/study-notes/)**

**Features:**
- 120+ flashcards across Algorithm and DevOps topics
- Spaced repetition learning system
- Progress tracking and deck selection
- Mobile-responsive design

**Adding Flashcards:**
Flashcards are automatically extracted from notes. Add them using any of these formats in your notes:

```markdown
## Flashcards

- What is the time complexity of binary search? ::: O(log n)
- What data structure is used for BFS? ::: Queue

```yaml
---
flashcards:
  - q: "Question?"
    a: "Answer!"
---
```

```flashcard
Q: Question?
A: Answer!

---

Q: Another question?
A: Another answer!
```


### Note Format

All notes should follow this structure:

- **Location**: `notes/{category}/short-slug.md`
- **Frontmatter**: Required fields are `title`, `date`, and `category`
- **Sections**: Include `Summary` and `Details` sections
- **Code blocks**: Use fenced code blocks with language tags
- **Tags**: Use `tags: []` for searchability

### Linting

The repository includes multiple layers of linting:

1. **Pre-commit hooks**: Automatically run markdownlint and prettier on changed files
2. **GitHub Actions**: Run linting on all pull requests and pushes
3. **Manual checking**: Run `npx remark .` and `npx markdownlint "**/*.md"` locally

### Cursor Integration

The `.cursor/rules.mdc` file contains best-effort rules for Cursor IDE integration. These rules help ensure consistent note formatting.

## Development

### Setup

1. Clone the repository
2. Install pre-commit hooks:
   ```bash
   pip install pre-commit
   pre-commit install
   ```

### Local Linting

```bash
# Install dependencies
npm init -y
npm install --no-audit --no-fund remark-cli remark-frontmatter remark-preset-lint-recommended markdownlint-cli

# Run linters
npx remark . --frail
npx markdownlint "**/*.md"
```