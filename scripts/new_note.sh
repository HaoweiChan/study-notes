#!/usr/bin/env bash
set -e

TITLE="$1"
CATEGORY_ARG="$2"   # user input
SLUG_ARG="$3"

if [ -z "$TITLE" ]; then
  echo "Usage: ./scripts/new_note.sh \"Title of note\" <category> [slug]"
  echo "Accepted categories (aliases): ml, algo, sd, system-design, system, sys"
  exit 1
fi

# default category
if [ -z "$CATEGORY_ARG" ]; then
  CATEGORY_ARG="ml"
fi

# normalize aliases to canonical directory names
case "$(echo "$CATEGORY_ARG" | tr '[:upper:]' '[:lower:]')" in
  ml|machine-learning|ml-research)
    CATEGORY="ml"
    ;;
  algo|algorithms|algo-problems)
    CATEGORY="algo"
    ;;
  sd|system-design|system|sys)
    CATEGORY="sd"
    ;;
  *)
    echo "Invalid category: $CATEGORY_ARG"
    echo "Valid categories/aliases: ml, algo, sd (you can also use system-design, sys, machine-learning, algorithms)"
    exit 1
    ;;
esac

DATE=$(date '+%Y-%m-%d')

if [ -n "$SLUG_ARG" ]; then
  SLUG="$SLUG_ARG"
else
  SLUG=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g' | sed -E 's/^-|-$//g')
fi

TARGET_DIR="notes/${CATEGORY}"
mkdir -p "$TARGET_DIR" templates

TARGET_FILE="${TARGET_DIR}/${SLUG}.md"

# Ensure template exists (create minimal template if missing)
if [ ! -f templates/note-template.md ]; then
  cat > templates/note-template.md <<'TEMPLATE'
---
title: "{{TITLE}}"
date: "{{DATE}}"
tags: []
related: []
slug: "{{SLUG}}"
category: "{{CATEGORY}}"
---

# {{TITLE}}

## Summary
Short 1–3 line summary.

## Details
Write your notes here.

## Examples / snippets
```python
# small code snippet (language-tagged)

Links
	•	reference links
TEMPLATE
fi

# Replace placeholders safely
esc_title=$(printf '%s' "$TITLE" | sed 's|/|\\/|g' | sed 's|&|\\\&|g')
sed -e "s/{{TITLE}}/${esc_title}/" \
    -e "s/{{DATE}}/${DATE}/" \
    -e "s/{{SLUG}}/${SLUG}/" \
    -e "s/{{CATEGORY}}/${CATEGORY}/" templates/note-template.md > "$TARGET_FILE"

echo "Created $TARGET_FILE"
