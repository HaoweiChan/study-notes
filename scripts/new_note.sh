#!/usr/bin/env bash
set -e

TITLE="$1"
CATEGORY_ARG="$2"   # user input
SLUG_ARG="$3"

# Function to discover available categories from notes/ directory
discover_categories() {
  local script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
  local notes_dir="$script_dir/../notes"

  # Find all subdirectories in notes/ and extract category names
  if [ -d "$notes_dir" ]; then
    find "$notes_dir" -maxdepth 1 -type d -not -path "$notes_dir" | sed 's|.*/||' | sort
  else
    echo "Notes directory not found: $notes_dir"
    exit 1
  fi
}

# Get available categories
AVAILABLE_CATEGORIES=$(discover_categories)

if [ -z "$TITLE" ]; then
  echo "Usage: ./scripts/new_note.sh \"Title of note\" <category> [slug]"
  echo "Available categories (auto-discovered from notes/): $(echo "$AVAILABLE_CATEGORIES" | tr '\n' ', ' | sed 's/,$//')"
  echo ""
  echo "Examples:"
  echo "  ./scripts/new_note.sh \"Binary Search\" algorithm"
  echo "  ./scripts/new_note.sh \"Neural Networks\" machine-learning"
  echo "  ./scripts/new_note.sh \"Docker Guide\" devops"
  echo ""
  echo "For LeetCode problems:"
  echo "  ./scripts/new_note.sh \"Two Sum\" leetcode"
  echo "  ./scripts/new_note.sh \"Valid Parentheses\" leetcode 20"
  exit 1
fi

# Default category
if [ -z "$CATEGORY_ARG" ]; then
  CATEGORY_ARG="algorithm"
fi

# Validate category exists in notes/ directory
CATEGORY_FOUND=false
for category in $AVAILABLE_CATEGORIES; do
  if [ "$category" = "$CATEGORY_ARG" ]; then
    CATEGORY_FOUND=true
    break
  fi
done

if [ "$CATEGORY_FOUND" = false ]; then
  echo "Invalid category: $CATEGORY_ARG"
  echo "Available categories: $(echo "$AVAILABLE_CATEGORIES" | tr '\n' ', ' | sed 's/,$//')"
  exit 1
fi

CATEGORY="$CATEGORY_ARG"

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
  # Convert category name to function name (replace hyphens with underscores)
  function_name="handle_$(echo "$CATEGORY" | tr '-' '_')_category"
  $function_name "$TITLE" "$SLUG" "$DATE" "$CATEGORY"
else
  # Use default handler
  source "$DEFAULT_SCRIPT"
  handle_default_category "$TITLE" "$SLUG" "$DATE" "$CATEGORY"
fi
