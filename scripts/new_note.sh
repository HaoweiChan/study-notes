#!/usr/bin/env bash
set -e

TITLE="$1"
CATEGORY_ARG="$2"   # user input
SLUG_ARG="$3"

if [ -z "$TITLE" ]; then
  echo "Usage: ./scripts/new_note.sh \"Title of note\" <category> [slug]"
  echo "Accepted categories: machine-learning, algorithm, system-design, agentic, leetcode"
  echo "Aliases: ml/machine-learning, algo/algorithm, sd/system-design"
  echo "For LeetCode: ./scripts/new_note.sh \"Problem Title\" leetcode [problem-number]"
  exit 1
fi

# Default category
if [ -z "$CATEGORY_ARG" ]; then
  CATEGORY_ARG="algorithm"
fi

# Normalize aliases to canonical directory names
case "$(echo "$CATEGORY_ARG" | tr '[:upper:]' '[:lower:]')" in
  ml|machine-learning|ml-research)
    CATEGORY="machine-learning"
    ;;
  algo|algorithms|algo-problems|algorithm)
    CATEGORY="algorithm"
    ;;
  sd|system-design|system|sys)
    CATEGORY="system-design"
    ;;
  agentic)
    CATEGORY="agentic"
    ;;
  leetcode)
    CATEGORY="leetcode"
    ;;
  *)
    echo "Invalid category: $CATEGORY_ARG"
    echo "Valid categories: machine-learning, algorithm, system-design, agentic, leetcode"
    exit 1
    ;;
esac

# Get current date
DATE=$(date '+%Y-%m-%d')

# Generate slug if not provided
if [ -n "$SLUG_ARG" ]; then
  SLUG="$SLUG_ARG"
else
  SLUG=$(echo "$TITLE" | tr '[:upper:]' '[:lower:]' | sed -E 's/[^a-z0-9]+/-/g' | sed -E 's/^-|-$//g')
fi

# Delegate to category-specific handler or default handler
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CATEGORY_SCRIPT="$SCRIPT_DIR/categories/${CATEGORY}.sh"
DEFAULT_SCRIPT="$SCRIPT_DIR/categories/default.sh"

if [ -f "$CATEGORY_SCRIPT" ]; then
  # Use specific handler for this category
  source "$CATEGORY_SCRIPT"
  handle_${CATEGORY}_category "$TITLE" "$SLUG" "$DATE"
else
  # Use default handler
  source "$DEFAULT_SCRIPT"
  handle_default_category "$TITLE" "$SLUG" "$DATE" "$CATEGORY"
fi
