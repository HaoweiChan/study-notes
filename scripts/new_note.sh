#!/usr/bin/env bash
set -e

TITLE="$1"
CATEGORY_ARG="$2"   # user input
SLUG_ARG="$3"

if [ -z "$TITLE" ]; then
  echo "Usage: ./scripts/new_note.sh \"Title of note\" <category> [slug]"
  echo "Accepted categories (aliases): ml, algo, sd, leetcode, system-design, system, sys"
  echo "For LeetCode: ./scripts/new_note.sh \"Problem Title\" leetcode [problem-number]"
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
  leetcode)
    CATEGORY="leetcode"
    ;;
  *)
    echo "Invalid category: $CATEGORY_ARG"
    echo "Valid categories/aliases: ml, algo, sd, leetcode (you can also use system-design, sys, machine-learning, algorithms)"
    exit 1
    ;;
esac

DATE=$(date '+%Y-%m-%d')

# Special handling for LeetCode problems
PROBLEM_NUMBER=""
if [ "$CATEGORY" = "leetcode" ]; then
  if [ -n "$SLUG_ARG" ]; then
    # Check if SLUG_ARG is a number (problem number) or a slug
    if echo "$SLUG_ARG" | grep -qE '^[0-9]+$'; then
      PROBLEM_NUMBER="$SLUG_ARG"
      SLUG_ARG=""  # Reset SLUG_ARG since it was the problem number
    fi
  fi

  # Prompt for problem number if not provided
  if [ -z "$PROBLEM_NUMBER" ]; then
    echo -n "Enter LeetCode problem number: "
    read PROBLEM_NUMBER
    if [ -z "$PROBLEM_NUMBER" ]; then
      echo "Problem number is required for LeetCode notes"
      exit 1
    fi
  fi

  # Validate problem number
  if ! echo "$PROBLEM_NUMBER" | grep -qE '^[0-9]+$'; then
    echo "Problem number must be numeric"
    exit 1
  fi
fi

if [ -n "$SLUG_ARG" ]; then
  SLUG="$SLUG_ARG"
else
  SLUG=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g' | sed -E 's/^-|-$//g')
fi

# For LeetCode, prefix filename with problem number
if [ "$CATEGORY" = "leetcode" ]; then
  FINAL_SLUG="${PROBLEM_NUMBER}-${SLUG}"
else
  FINAL_SLUG="$SLUG"
fi

TARGET_DIR="notes/${CATEGORY}"
mkdir -p "$TARGET_DIR" templates

TARGET_FILE="${TARGET_DIR}/${FINAL_SLUG}.md"

# Choose template based on category
TEMPLATE_FILE="templates/note-template.md"
if [ "$CATEGORY" = "leetcode" ]; then
  TEMPLATE_FILE="templates/leetcode-template.md"
fi

# Ensure template exists (create minimal template if missing)
if [ ! -f "$TEMPLATE_FILE" ]; then
  if [ "$CATEGORY" = "leetcode" ]; then
    cat > templates/leetcode-template.md <<'TEMPLATE'
---
title: "{{TITLE}}"
date: "{{DATE}}"
tags: []
related: []
slug: "{{SLUG}}"
category: "{{CATEGORY}}"
leetcode_url: "{{LEETCODE_URL}}"
leetcode_difficulty: "{{DIFFICULTY}}"
leetcode_topics: []
---

# {{TITLE}}

## Summary
{{SUMMARY}}

## Problem Description
{{PROBLEM_DESCRIPTION}}

## Solution Approach
{{APPROACH}}

## Time & Space Complexity
- **Time Complexity:** {{TIME_COMPLEXITY}}
- **Space Complexity:** {{SPACE_COMPLEXITY}}

## Key Insights
{{INSIGHTS}}

## Examples / snippets

### Solution Code
```python
{{SOLUTION_CODE}}
```

### Example Walkthrough
{{WALKTHROUGH}}

## Edge Cases & Validation
{{EDGE_CASES}}

## Related Problems
{{RELATED_PROBLEMS}}
TEMPLATE
  else
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
  - reference links
TEMPLATE
  fi
fi

# Replace placeholders safely
esc_title=$(printf '%s' "$TITLE" | sed 's|/|\\/|g' | sed 's|&|\\\&|g')
sed -e "s/{{TITLE}}/${esc_title}/" \
    -e "s/{{DATE}}/${DATE}/" \
    -e "s/{{SLUG}}/${FINAL_SLUG}/" \
    -e "s/{{CATEGORY}}/${CATEGORY}/" "$TEMPLATE_FILE" > "$TARGET_FILE"

echo "Created $TARGET_FILE"
