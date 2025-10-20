# Study Notes Flashcards

This is a simple flashcard study interface for the study notes repository.

## Usage

1. Visit the flashcards page at: `https://[username].github.io/study-notes/`
2. Select a deck from the dropdown (All, ML, Algo, SD)
3. Click "Show Answer" to reveal the answer
4. Rate your recall:
   - **Again**: Card needs more review (shown soon)
   - **Good**: Card recalled correctly (shown later)
   - **Easy**: Card mastered (shown much later)
5. Use navigation buttons to move between cards

## Adding New Flashcards

Flashcards are automatically extracted from your notes in the following formats:

### Frontmatter Format
```yaml
---
title: "Note Title"
flashcards:
  - q: "Question?"
    a: "Answer!"
  - q: "Another question?"
    a: "Another answer!"
---
```

### Fenced Block Format
```markdown
```flashcard
Q: What is this?
A: This is a flashcard example.

---

Q: Another question?
A: Another answer!
```
```

### List Format in Flashcards Section
```markdown
## Flashcards

- What is X? ::: This is X
- What is Y? ::: This is Y
```

## Technical Details

- Progress is saved locally in your browser
- Cards are filtered by category (machine-learning, algorithm, system-design)
- Simple spaced repetition algorithm (rudimentary scheduling)
- Works offline once loaded
