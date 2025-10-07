# Personal Notes Repo

A personal repository for organized study notes and knowledge sharing.

## Repository Structure

```
notes-repo/
├── README.md                 # This file
├── templates/
│   └── note-template.md      # Template for new notes
├── notes/                    # Your notes organized by category
│   ├── ml/                   # Machine learning notes
│   ├── algo/                 # Algorithms & problem solving
│   └── sd/                   # System design notes
├── .cursor/
│   └── rules.mdc            # Cursor rules for note generation/validation
├── .github/
│   └── workflows/
│       └── lint.yml         # GitHub Actions for linting
├── .markdownlint.json       # Markdown linting configuration
├── .remarkrc.cjs           # Remark linting configuration
├── .pre-commit-config.yaml  # Pre-commit hooks
└── scripts/
    └── new_note.sh         # Helper script to create new notes
```

## Usage

### Creating New Notes

Use the provided helper script to create new notes from the template:

```bash
./scripts/new_note.sh "Title of note" <category> [slug]
```

**Categories (with aliases):**
- `ml` (machine-learning, ml-research) — Machine learning
- `algo` (algorithms, algo-problems) — Algorithms & problem solving
- `sd` (system-design, system, sys) — System design

**Examples:**
```bash
# Default category (ML)
./scripts/new_note.sh "Kalman hedge ratio"

# Explicit categories
./scripts/new_note.sh "Longest Increasing Subsequence DP" algo lis-dp
./scripts/new_note.sh "High-throughput video ingestion" sd

# Using aliases
./scripts/new_note.sh "Neural Network Architectures" machine-learning
./scripts/new_note.sh "Load Balancer Design" system-design
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

## Contributing

This is a personal study notes repository. Feel free to use this structure for your own notes!

## License

This repository is for personal use and learning purposes.
